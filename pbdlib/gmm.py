import numpy as np

from .functions import *
from .model import *


import math
from numpy.linalg import inv, pinv, norm, det
import sys


class GMM(Model):
	def __init__(self, nb_states=1, nb_dim=None, nb_features=1):
		Model.__init__(self, nb_states)
		self.nb_dim = nb_dim
		self.nb_features = nb_features
		self.features_dyn = None
		self.publish_init = False  # flag to indicate that publishing was not init



	def ros_publish(self, model_publisher):
		from baxter_learning.msg import GMM_msgs

		msg = GMM_msgs()
		msg.nb_dim = self.nb_dim
		msg.nb_features = self.nb_features
		msg.nb_states = self.nb_states

		msg.Mu = self.Mu.flatten()
		msg.Priors = self.Priors.flatten()
		msg.Sigma = self.Sigma.flatten()
		msg.Lambda = self.Lambda.flatten()

		model_publisher.publish(msg)

	def ros_cb(self, GMM_msg):
		self.nb_dim = GMM_msg.nb_dim
		self.nb_features = GMM_msg.nb_features
		self.nb_states = GMM_msg.nb_states

		self.Priors = GMM_msg.Priors
		self.Mu = GMM_msg.Mu.reshape(self.nb_dim, self.nb_states)
		self.Sigma = GMM_msg.Sigma.reshape(self.nb_dim, self.nb_dim, self.nb_states)
		self.Lambda = GMM_msg.Lambda.reshape(self.nb_dim, self.nb_dim, self.nb_states)

	def lintrans(self, A, b, Sigma_b=None):

		dim = self.nb_dim/self.nb_features

		gmmt = GMM(self.nb_states,self.nb_dim)
		if self.nb_features == 1:
			gmmt.Mu = (A.dot(self.Mu) + b[:,np.newaxis])
		else:
			gmmt.Mu = np.empty(self.Mu.shape)
			for k, relatif in enumerate(self.features_dyn):
				if relatif:
					gmmt.Mu[k*dim:(k+1)*dim] = (A.dot(self.Mu[k*dim:(k+1)*dim]))
				else:
					gmmt.Mu[k*dim:(k+1)*dim] = (A.dot(self.Mu[k*dim:(k+1)*dim]) + b[:,np.newaxis])
		nb_dim = b.shape[0]
		# self.nb_dim = nb_dim

		gmmt.Sigma = np.zeros(self.Sigma.shape)
		gmmt.Lambda = np.zeros(self.Sigma.shape)

		if self.nb_features == 1:
			if Sigma_b is None:
				for i in range(self.nb_states):
					gmmt.Sigma[:,:,i] = A.dot(self.Sigma[:,:,i].dot(A.T))
					gmmt.Lambda[:,:,i] = A.dot(self.Lambda[:,:,i].dot(A.T))
			else:
				for i in range(self.nb_states):
					gmmt.Sigma[:, :, i] = A.dot((self.Sigma[:, :, i]+Sigma_b).dot(A.T))
					gmmt.Lambda[:, :, i] = np.linalg.inv(gmmt.Sigma[:, :, i])
		else:
			if Sigma_b is None:
				for i in range(self.nb_states):
					for k, relatif in enumerate(self.features_dyn):

						gmmt.Sigma[k * dim:(k + 1) * dim,k * dim:(k + 1) * dim, i] = \
							A.dot(self.Sigma[k * dim:(k + 1) * dim,k * dim:(k + 1) * dim, i].dot(A.T))

					gmmt.Lambda[:, :, i] = np.linalg.inv(gmmt.Sigma[:, :, i])
			else:
				for i in range(self.nb_states):
					gmmt.Sigma[:, :, i] = A.dot((self.Sigma[:, :, i] + Sigma_b).dot(A.T))
					gmmt.Lambda[:, :, i] = np.linalg.inv(gmmt.Sigma[:, :, i])


		return gmmt

	def update_precision_matrix(self):
		if self.Lambda is None:
			dim = self.Sigma.shape[0]
			self.Lambda = np.empty((self.nb_dim, self.nb_dim, self.nb_states), order='C')

		for i in range(self.nb_states):
			self.Lambda[:, :, i] = inv(self.Sigma[:, :, i])

	# Parameters in standard form. Call with capital for PBD form



	def em(self, data, reg=1e-8, maxiter=100, minstepsize=1e-5):

		if self.Sigma is None:
			self.Sigma = np.rollaxis(
				np.array([np.cov(data.T)/self.nb_states for i in range(self.nb_states)]), 0, 3)


		if self.Priors is None:
			self.Priors = np.ones(self.nb_states)/self.nb_states

		nb_min_steps = 5  # min num iterations
		nb_max_steps = maxiter  # max iterations
		max_diff_ll = minstepsize # max log-likelihood increase

		min_sigma = reg * np.eye(self.nb_dim)

		nb_samples = data.shape[0]

		from sklearn.cluster import KMeans
		km_init = KMeans(n_clusters=self.nb_states)
		km_init.fit(data)

		self.Mu = km_init.cluster_centers_.swapaxes(0, 1)

		data = data.T

		LL = np.zeros(nb_max_steps)
		for it in range(nb_max_steps):

			# E - step
			L = np.zeros((self.nb_states, nb_samples))
			L_log = np.zeros((self.nb_states, nb_samples))

			for i in range(self.nb_states):
				L_log[i, :] = np.log(self.Priors[i]) + multi_variate_normal(data.T, self.Mu[:, i],
												   self.Sigma[:, :, i], log=True)

				# L[i, :] = self.Priors[i] * multi_variate_normal(data,
				# 															self.Mu[:, i],
				# 															self.Sigma[:, :,
				# 															i])

			L = np.exp(L_log)

			GAMMA = L / np.sum(L, axis=0)
			GAMMA2 = GAMMA / np.sum(GAMMA, axis=1)[:, np.newaxis]


			# M-step
			self.Mu = np.einsum('ac,ic->ia', GAMMA2,
									data)  # a states, c sample, i dim

			dx = data[:, None] - self.Mu[:, :, None]  # nb_dim, nb_states, nb_samples

			self.Sigma = np.einsum('acj,iac->ija', np.einsum('iac,ac->aci', dx, GAMMA2),
									   dx)  # a states, c sample, i-j dim

			for i in range(self.nb_states):
				self.Sigma[:, :, i] = self.Sigma[:, :, i] + min_sigma

			# print self.Sigma[:, :, i]

			# Update initial state probablility vector
			self.Priors = np.mean(GAMMA, axis=1)

			# print GAMMA2[i]
			LL[it] = 0

			LL[it] -= np.sum(L_log)
			LL[it] = LL[it] / nb_samples
			# Check for convergence
			if it > nb_min_steps:
				if LL[it] - LL[it - 1] < max_diff_ll:
					return GAMMA

		print "GMM did not converge before reaching max iteration. Consider augmenting the number of max iterations."
		return GAMMA

	def __mul__(self, other):

		Sigma = np.zeros((self.nb_dim,self.nb_dim,self.nb_states))
		Lambda = np.zeros((self.nb_dim,self.nb_dim,self.nb_states))
		Mu  = np.zeros((self.nb_dim,self.nb_states))

		if self.Lambda is None:
			self.update_precision_matrix()
		if other.Lambda is None:
			other.update_precision_matrix()

		for i in range(self.nb_states):
			# Compute precision matrices
			# prec_s = np.linalg.inv(self.Sigma[:,:,i])
			# prec_o = np.linalg.inv(other.Sigma[:,:,i])
			prec_s = self.Lambda[:, :, i]
			prec_o = other.Lambda[:, :, i]
			# Compute covariance of p	roduct:
			Lambda[:,:,i] = prec_s + prec_o
			Sigma[:,:,i] = np.linalg.inv(Lambda[:,:,i])
			# Sigma[:,:,i] = np.linalg.inv(prec_s + prec_o)

			# Compute mean of product:
			Mu[:,i] = Sigma[:,:,i].dot(prec_s.dot(self.Mu[:,i]) + prec_o.dot(other.Mu[:,i]))

		# Create GMM of product
		prodgmm = GMM(self.nb_states, self.nb_dim)
		prodgmm.Mu = Mu
		prodgmm.Sigma = Sigma
		prodgmm.Lambda = Lambda


		return prodgmm
