import numpy as np
from .functions import *

class Model(object):
	''' Placeholder for a model'''

	def __init__(self, nb_states, nb_dim=2):
		self._mu = None
		self._sigma = None
		self._lmbda = None

		self.Priors = None
		self.Mu = None
		self.Sigma = None  # covariance matrix
		self.Lambda = None  # Precision matrix
		self.nb_states = nb_states

		self.nb_dim = nb_dim


	@property
	def mu(self):
		if self._mu is None:
			self._mu = self.Mu.swapaxes(0, 1)

		return self._mu

	@mu.setter
	def mu(self, value):
		self._mu = value
		self.Mu = np.rollaxis(self._mu, 0, 1)


	@property
	def sigma(self):
		if self._sigma is None:
			self._sigma = np.rollaxis(self.Sigma, 2, 0)

		return self._sigma

	@sigma.setter
	def sigma(self, value):
		self._sigma = value
		self.Sigma = np.rollaxis(self._sigma, 0, 3)

	@property
	def lmbda(self):
		if self._lmbda is None:
			self._lmbda = np.linalg.inv(self.sigma)

		return self._lmbda

	@lmbda.setter
	def lmbda(self, value):
		self._lmbda = value
		self.Lambda = np.rollaxis(self._lmbda, 0, 3)

	def condition(self, data_in, dim_in, dim_out, h=None):
		"""

		:param data_in:		[np.array([nb_timestep, nb_dim])
		:param dim_in:
		:param dim_out:
		:param h:
		:return:
		"""
		sample_size = data_in.shape[0]


		# compute responsabilities
		mu_in, sigma_in = self.get_marginal(dim_in)

		if h is None:
			h = np.zeros((self.nb_states, sample_size))
			for i in range(self.nb_states):
				h[i, :] = multi_variate_normal(data_in,
											   mu_in[:, i],
											   sigma_in[:, :, i], log=False)

		h /= np.sum(h, axis=0)[None, :]

		mu_out, sigma_out = self.get_marginal(dim_out)
		mu_est, sigma_est = ([], [])

		inv_sigma_in_in, inv_sigma_out_in = ([], [])

		_, sigma_in_out = self.get_marginal(dim_in, dim_out)

		for i in range(self.nb_states):
			inv_sigma_in_in += [np.linalg.inv(sigma_in[:, :, i])]
			inv_sigma_out_in += [sigma_in_out[:, :, i].T.dot(inv_sigma_in_in[-1])]

			mu_est += [mu_out[:, i] + np.einsum('ij,aj->ai',
												inv_sigma_out_in[-1], data_in - mu_in[:,i])]

			# mu_est += [mu_out[:, i] + 0. * np.einsum('ij,aj->ai',
			# 									inv_sigma_out_in[-1], data_in - mu_in[:,i])]

			sigma_est += [sigma_out[:, :, i] - inv_sigma_out_in[-1].dot(sigma_in_out[:, :, i])]

		mu_est, sigma_est = (np.asarray(mu_est), np.asarray(sigma_est))

		# return np.mean(mu_est, axis=0)
		return np.sum(h[:, :, None] * mu_est, axis=0), np.sum(h[:,:,None, None] * sigma_est[:, None] ,axis=0)

	def get_marginal(self, dim, dim_out=None):
		mu, sigma = (self.Mu, self.Sigma)

		if isinstance(dim, list):
			if dim_out is not None:
				dGrid = np.ix_(dim, dim_out)
			else:
				dGrid = np.ix_(dim, dim)

			mu, sigma = (mu[dim], sigma[dGrid])
		elif isinstance(dim, slice):
			if dim_out is not None:
				mu, sigma = (mu[dim], sigma[dim, dim_out])
			else:
				mu, sigma = (mu[dim], sigma[dim, dim])

		return mu, sigma

