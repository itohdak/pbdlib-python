import numpy as np
from .gmm import GMM, MVN
from functions import multi_variate_normal, multi_variate_t
from utils import gaussian_moment_matching

class MTMM(GMM):
	"""
	Multivariate t-distribution mixture
	"""

	def __init__(self, *args, **kwargs):
		GMM.__init__(self, *args, **kwargs)

		self._nu = None
		self._k = None

	def __add__(self, other):
		if isinstance(other, MVN):
			gmm = MTMM(nb_dim=self.nb_dim, nb_states=self.nb_states)

			gmm.nu = self.nu
			gmm.k = self.k
			gmm.priors = self.priors
			gmm.mu = self.mu + other.mu[None]
			gmm.sigma = self.sigma + other.sigma[None]

			return gmm

		else:
			raise NotImplementedError

	@property
	def k(self):
		return self._k

	@k.setter
	def k(self, value):
		self._k = value

	@property
	def nu(self):
		return self._nu

	@nu.setter
	def nu(self, value):
		self._nu = value

	def condition_gmm(self, data_in, dim_in, dim_out):
		sample_size = data_in.shape[0]

		# compute responsabilities
		mu_in, sigma_in = self.get_marginal(dim_in)


		h = np.zeros((self.nb_states, sample_size))
		for i in range(self.nb_states):
			h[i, :] = multi_variate_t(data_in[None], self.nu[i],
									  mu_in[i],
									  sigma_in[i])

		h += np.log(self.priors)[:, None]
		h = np.exp(h).T
		h /= np.sum(h, axis=1, keepdims=True)
		h = h.T

		mu_out, sigma_out = self.get_marginal(dim_out)
		mu_est, sigma_est = ([], [])

		inv_sigma_in_in, inv_sigma_out_in = ([], [])

		_, sigma_in_out = self.get_marginal(dim_in, dim_out)

		for i in range(self.nb_states):
			inv_sigma_in_in += [np.linalg.inv(sigma_in[i])]
			inv_sigma_out_in += [sigma_in_out[i].T.dot(inv_sigma_in_in[-1])]
			dx = data_in[None] - mu_in[i]
			mu_est += [mu_out[i] + np.einsum('ij,aj->ai',
											 inv_sigma_out_in[-1], dx)]

			s = np.sum(np.einsum('ai,ij->aj', dx, inv_sigma_in_in[-1]) * dx, axis=1)
			a = (self.nu[i] + s) / (self.nu[i] + mu_in.shape[1])

			sigma_est += [a[:, None, None] *
						  (sigma_out[i] - inv_sigma_out_in[-1].dot(sigma_in_out[i]))[None]]

		mu_est, sigma_est = (np.asarray(mu_est)[:, 0], np.asarray(sigma_est)[:, 0])


		gmm_out = MTMM(nb_states=self.nb_states, nb_dim=mu_out.shape[1])
		gmm_out.nu = self.nu + gmm_out.nb_dim
		gmm_out.mu = mu_est
		gmm_out.sigma = sigma_est
		gmm_out.priors = h[:, 0]

		return gmm_out


	def condition(self, data_in, dim_in, dim_out, h=None, return_gmm=False):
		"""
		[1] M. Hofert, 'On the Multivariate t Distribution,' R J., vol. 5, pp. 129-136, 2013.
		:return:

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
				# h[i, :] = multi_variate_normal(data_in,
				# 							   mu_in[i],
				# 							   sigma_in[i])
				h[i, :] = multi_variate_t(data_in, self.nu[i],
											   mu_in[i],
											   sigma_in[i])

			h += np.log(self.priors)[:, None]
			h = np.exp(h).T
			h /= np.sum(h, axis=1, keepdims=True)
			h = h.T

		self._h = h
		mu_out, sigma_out = self.get_marginal(dim_out)
		mu_est, sigma_est = ([], [])

		inv_sigma_in_in, inv_sigma_out_in = ([], [])

		_, sigma_in_out = self.get_marginal(dim_in, dim_out)

		for i in range(self.nb_states):
			inv_sigma_in_in += [np.linalg.inv(sigma_in[i])]
			inv_sigma_out_in += [sigma_in_out[i].T.dot(inv_sigma_in_in[-1])]
			dx = data_in - mu_in[i]
			mu_est += [mu_out[i] + np.einsum('ij,aj->ai',
											 inv_sigma_out_in[-1], dx)]

			s = np.sum(np.einsum('ai,ij->aj', dx, inv_sigma_in_in[-1]) * dx, axis=1)
			a = (self.nu[i] + s)/(self.nu[i] + mu_in.shape[1])

			sigma_est += [a[:, None, None] *
						  (sigma_out[i] - inv_sigma_out_in[-1].dot(sigma_in_out[i]))[None]]

		mu_est, sigma_est = (np.asarray(mu_est), np.asarray(sigma_est))

		if return_gmm:
			return mu_est, sigma_est
		# return np.mean(mu_est, axis=0)
		else:

			return gaussian_moment_matching(mu_est, sigma_est, h.T)
