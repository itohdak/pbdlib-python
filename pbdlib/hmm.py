import numpy as np

from .functions import *
from .model import *
from .gmm import *
from .gmr import *

import math
from numpy.linalg import inv, pinv, norm, det
import sys


class HMM(GMM):
	def __init__(self, nb_states, nb_dim=2):
		Model.__init__(self, nb_states, nb_dim)

	def init_hmm_kbins(self, demos, dep=None, reg=1e-8):
		"""
		Init HMM by splitting each demos in K bins along time. Each K states of the HMM will
		be initialized with one of the bin. It corresponds to a left-to-right HMM.

		:param demos:	[list of np.array([nb_timestep, nb_dim])]
		:param dep:
		:param reg:		[float]
		:return:
		"""

		# delimit the cluster bins for first demonstration
		self.nb_dim = demos[0].shape[1]

		self.Priors = np.zeros(self.nb_states)
		self.Mu = np.zeros((self.nb_dim, self.nb_states))
		self.Sigma = np.zeros((self.nb_dim, self.nb_dim, self.nb_states))
		t_sep = []

		for demo in demos:
			t_sep += [map(int, np.round(np.linspace(0, demo.shape[0], self.nb_states + 1)))]

		# print t_sep
		for i in range(self.nb_states):
			data_tmp = np.empty((0, self.nb_dim))
			inds = []
			states_nb_data = 0   # number of datapoints assigned to state i

			# Get bins indices for each demonstration
			for n, demo in enumerate(demos):
				inds = range(t_sep[n][i], t_sep[n][i+1])

				data_tmp = np.concatenate([data_tmp, demo[inds]], axis=0)
				states_nb_data += t_sep[n][i+1]-t_sep[n][i]

			self.Priors[i] = states_nb_data
			self.Mu[:, i] = np.mean(data_tmp, axis=0)

			if dep is None:
				self.Sigma[:, :, i] = np.cov(data_tmp.T) + np.eye(self.nb_dim) * reg
			else:
				for d in dep:
					dGrid = np.ix_(d, d, [i])
					self.Sigma[dGrid] = (np.cov(data_tmp[:, d].T) + np.eye(
						len(d)) * reg)[:, :, np.newaxis]
				# print self.Sigma[:,:,i]

		# normalize priors
		self.Priors = self.Priors / np.sum(self.Priors)

		# Hmm specific init
		self.Trans = np.ones((self.nb_states, self.nb_states)) * 0.01

		nb_data = np.mean([d.shape[0] for d in demos])

		for i in range(self.nb_states - 1):
			self.Trans[i, i] = 1.0 - float(self.nb_states) / nb_data
			self.Trans[i, i + 1] = float(self.nb_states) / nb_data

		self.Trans[-1, -1] = 1.0
		self.StatesPriors = np.ones(self.nb_states) * 1./self.nb_states


	def viterbi(self, demo):
		"""
		Compute most likely sequence of state given observations

		:param demo: 	[np.array([nb_timestep, nb_dim])]
		:return:
		"""

		nb_data, dim = demo.shape

		demo = demo.T
		logB = np.zeros((self.nb_states, nb_data))
		logDELTA = np.zeros((self.nb_states, nb_data))
		PSI = np.zeros((self.nb_states, nb_data)).astype(int)

		for i in range(self.nb_states):
			logB[i, :] = multi_variate_normal(demo.T, self.Mu[:, i], self.Sigma[:, :, i], log=True)

		# forward pass
		logDELTA[:, 0] = np.log(self.StatesPriors + realmin) + logB[:, 0]

		for t in range(1, nb_data):
			for i in range(self.nb_states):
				# get index of maximum value : most probables
				PSI[i, t] = np.argmax(logDELTA[:, t - 1] + np.log(self.Trans[:, i] + realmin))
				logDELTA[i, t] = np.max(logDELTA[:, t - 1] + np.log(self.Trans[:, i] + realmin)) + \
								 logB[i, t]

		# backtracking
		q = [0 for i in range(nb_data)]
		q[-1] = np.argmax(logDELTA[:, -1])
		for t in range(nb_data - 2, -1, -1):
			q[t] = PSI[q[t + 1], t + 1]

		return q

	def compute_messages(self, demo, dep=None, table=None, marginal=None):
		"""

		:param demo: 	[np.array([nb_timestep, nb_dim])]
		:param dep: 	[A x [B x [int]]] A list of list of dimensions
			Each list of dimensions indicates a dependence of variables in the covariance matrix
			E.g. [[0],[1],[2]] indicates a diagonal covariance matrix
			E.g. [[0, 1], [2]] indicates a full covariance matrix between [0, 1] and no
			covariance with dim [2]
		:param table: 	np.array([nb_states, nb_demos]) - composed of 0 and 1
			A mask that avoid some demos to be assigned to some states
		:param marginal: [slice(dim_start, dim_end)]
			If not None, compute messages with marginals probabilities
			(can be used for time-series regression)
		:return:
		"""
		sample_size = demo.shape[0]

		# emission probabilities
		B = np.zeros((self.nb_states, sample_size))

		for i in range(self.nb_states):
			Mu, Sigma = (self.Mu, self.Sigma)

			if marginal is not None:
				Mu, Sigma = self.get_marginal(marginal)

			if dep is None :
				B[i, :] = multi_variate_normal(demo,
											   Mu[:, i],
											   Sigma[:, :, i], log=False)
			else:  # block diagonal computation
				B[i, :] = 1.0
				for d in dep:
					dGrid = np.ix_(d, d, [i])
					B[[i], :] *= multi_variate_normal(demo, Mu[d, i],
													  Sigma[dGrid][:, :, 0], log=False)

		if table is not None:
			B *= table[:, [n]]

		self._B = B

		# forward variable alpha (rescaled)
		alpha = np.zeros((self.nb_states, sample_size))
		alpha[:, 0] = self.StatesPriors * B[:, 0]
		c = np.zeros(sample_size)
		c[0] = 1.0 / np.sum(alpha[:, 0] + realmin)
		alpha[:, 0] = alpha[:, 0] * c[0]

		for t in range(1, sample_size):
			alpha[:, t] = mul([alpha[:, t - 1], self.Trans]) * B[:, t]
			# Scaling to avoid underflow issues
			c[t] = 1.0 / np.sum(alpha[:, t] + realmin)
			alpha[:, t] = alpha[:, t] * c[t]

		# backward variable beta (rescaled)
		beta = np.zeros((self.nb_states, sample_size))
		beta[:, -1] = np.ones(self.nb_states) * c[-1]  # Rescaling
		for t in range(sample_size - 2, -1, -1):
			beta[:, t] = np.dot(self.Trans, beta[:, t + 1]) * B[:, t + 1]
			beta[:, t] = np.minimum(beta[:, t] * c[t], realmax)

		# Smooth node marginals, gamma
		gamma = (alpha * beta) / np.tile(np.sum(alpha * beta, axis=0) + realmin,
										 (self.nb_states, 1))

		# Smooth edge marginals. zeta (fast version, considers the scaling factor)
		zeta = np.zeros((self.nb_states, self.nb_states, sample_size - 1))

		for i in range(self.nb_states):
			for j in range(self.nb_states):
				zeta[i, j, :] = self.Trans[i, j] * alpha[i, 0:-1] * B[j, 1:] * beta[
																			   j,
																			   1:]

		return alpha, beta, gamma, zeta, c

	def em(self, demos, dep=None, reg=1e-8, table=None, end_cov=False, cov_type='full'):
		"""

		:param demos:	[list of np.array([nb_timestep, nb_dim])]
		:param dep:		[A x [B x [int]]] A list of list of dimensions
			Each list of dimensions indicates a dependence of variables in the covariance matrix
			E.g. [[0],[1],[2]] indicates a diagonal covariance matrix
			E.g. [[0, 1], [2]] indicates a full covariance matrix between [0, 1] and no
			covariance with dim [2]
		:param reg:		[float] or list [nb_dim x float] for different regularization in different dimensions
			Regularization term used in M-step for covariance matrices
		:param table:		np.array([nb_states, nb_demos]) - composed of 0 and 1
			A mask that avoid some demos to be assigned to some states
		:param end_cov:	[bool]
			If True, compute covariance matrix without regularization after convergence
		:param cov_type: 	[string] in ['full', 'diag', 'spherical']
		:return:
		"""
		nb_min_steps = 5  # min num iterations
		nb_max_steps = 50  # max iterations
		max_diff_ll = 1e-4  # max log-likelihood increase

		nb_samples = len(demos)
		data = np.concatenate(demos).T
		nb_data = data.shape[0]

		s = [{} for d in demos]
		# stored log-likelihood
		LL = np.zeros(nb_max_steps)

		# create regularization matrix
		if isinstance(reg, float):
			min_sigma = np.eye(self.nb_dim) * reg
		else:
			min_sigma = np.diag(reg)

		for it in range(nb_max_steps):

			for n, demo in enumerate(demos):
				s[n]['alpha'], s[n]['beta'], s[n]['gamma'], s[n]['zeta'], s[n]['c'] = self.compute_messages(demo, dep, table)

			# concatenate intermediary vars
			gamma = np.hstack([s[i]['gamma'] for i in range(nb_samples)])
			zeta = np.dstack([s[i]['zeta'] for i in range(nb_samples)])
			gamma_init = np.hstack([s[i]['gamma'][:, 0:1] for i in range(nb_samples)])
			gamma_trk = np.hstack([s[i]['gamma'][:, 0:-1] for i in range(nb_samples)])

			gamma2 = gamma / np.tile(np.sum(gamma, axis=1).reshape(-1, 1) + realmin,
									 (1, gamma.shape[1]))

			# M-step
			for i in range(self.nb_states):
				# Update centers
				self.Mu[:, i] = np.einsum('a,ia->i',gamma2[i], data)

				# Update covariances
				Data_tmp = data - self.Mu[:, [i]]
				self.Sigma[:, :, i] = np.einsum('ij,jk->ik',
												np.einsum('ij,j->ij', Data_tmp,
														  gamma2[i, :]), Data_tmp.T)
				# Regularization
				self.Sigma[:, :, i] = self.Sigma[:, :, i] + min_sigma

				if cov_type == 'diag':
					self.Sigma[:, :, i] *= np.eye(self.Sigma.shape[0])
				
			# Update initial state probablility vector
			self.StatesPriors = np.mean(gamma_init, axis=1)

			# Update transition probabilities
			self.Trans = np.sum(zeta, axis=2) / np.tile(
				np.sum(gamma_trk, axis=1).reshape(-1, 1) + realmin, (1, self.nb_states))
			# print self.Trans

			# Compute avarage log-likelihood using alpha scaling factors
			LL[it] = 0
			for n in range(nb_samples):
				LL[it] -= sum(np.log(s[n]['c']))
			LL[it] = LL[it] / nb_samples

			# Check for convergence
			if it > nb_min_steps:
				if LL[it] - LL[it - 1] < max_diff_ll:
					if end_cov:
						for i in range(self.nb_states):
							# recompute covariances without regularization
							Data_tmp = data - self.Mu[:, [i]]
							self.Sigma[:, :, i] = np.einsum('ij,jk->ik',
												np.einsum('ij,j->ij', Data_tmp,
														  gamma2[i, :]), Data_tmp.T)

					self.update_precision_matrix()
					# print "EM converged after " + str(it) + " iterations"
					# print LL[it]
					return LL[it]

		print "EM did not converge"
		print LL
		return LL

	def score(self, demos):
		"""

		:param demos:	[list of np.array([nb_timestep, nb_dim])]
		:return:
		"""
		ll = []
		for n, demo in enumerate(demos):
			_, _, _, _, c = self.compute_messages(demo)
			ll += [np.sum(np.log(c))]

		return ll

	def condition(self, data_in, dim_in, dim_out, h=None, gmm=False):
		if gmm:
			return super(HMM, self).condition(data_in, dim_in, dim_out)
		else:
			a, _, _, _, _ = self.compute_messages(data_in, marginal=dim_in)

			return super(HMM, self).condition(data_in, dim_in, dim_out, h=a)
