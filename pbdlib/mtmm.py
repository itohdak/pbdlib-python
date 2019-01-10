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


	def condition(self, data_in, dim_in, dim_out, h=None, return_gmm=False, reg_in=1e-20):
		"""
		[1] M. Hofert, 'On the Multivariate t Distribution,' R J., vol. 5, pp. 129-136, 2013.

		Conditional probabilities in a Joint Multivariate t Distribution Mixture Model

		:param data_in:		[np.array([nb_data, nb_dim])
				Observed datapoints x_in
		:param dim_in:		[slice] or [list of index]
				Dimension of input space e.g.: slice(0, 3), [0, 2, 3]
		:param dim_out:		[slice] or [list of index]
				Dimension of output space e.g.: slice(3, 6), [1, 4]
		:param h:			optional - [np.array([nb_states, nb_data])]
				Overrides marginal probability of states given input dimensions
		:return:
		"""

		sample_size = data_in.shape[0]

		# compute marginal probabilities of states given observation p(k|x_in)
		mu_in, sigma_in = self.get_marginal(dim_in)

		if h is None:
			h = np.zeros((self.nb_states, sample_size))
			for i in range(self.nb_states):
				h[i, :] = multi_variate_t(data_in, self.nu[i],
											   mu_in[i],
											   sigma_in[i])

			h += np.log(self.priors)[:, None]
			h = np.exp(h).T
			h /= np.sum(h, axis=1, keepdims=True)
			h = h.T

		self._h = h # storing value

		mu_out, sigma_out = self.get_marginal(dim_out)  # get marginal distribution of x_out
		mu_est, sigma_est = ([], [])

		# get conditional distribution of x_out given x_in for each states p(x_out|x_in, k)
		inv_sigma_in_in, inv_sigma_out_in = ([], [])

		_, sigma_in_out = self.get_marginal(dim_in, dim_out)

		for i in range(self.nb_states):
			inv_sigma_in_in += [np.linalg.inv(sigma_in[i] + reg_in * np.eye(sigma_in.shape[-1]))]
			inv_sigma_out_in += [sigma_in_out[i].T.dot(inv_sigma_in_in[-1])]
			dx = data_in - mu_in[i]
			mu_est += [mu_out[i] + np.einsum('ij,aj->ai',
											 inv_sigma_out_in[-1], dx)]

			s = np.sum(np.einsum('ai,ij->aj', dx, inv_sigma_in_in[-1]) * dx, axis=1)
			a = (self.nu[i] + s)/(self.nu[i] + mu_in.shape[1])

			sigma_est += [a[:, None, None] *
						  (sigma_out[i] - inv_sigma_out_in[-1].dot(sigma_in_out[i]))[None]]

		mu_est, sigma_est = (np.asarray(mu_est), np.asarray(sigma_est))

		nu = self.nu + mu_in.shape[1]
		# the conditional distribution is now a still a mixture

		if return_gmm:
			return mu_est, sigma_est * nu/(nu-2.)
		else:
			# apply moment matching to get a single MVN for each datapoint
			return gaussian_moment_matching(mu_est, sigma_est * (nu/(nu-2.))[:, None, None, None], h.T)

	def get_pred_post_uncertainty(self, data_in, dim_in, dim_out):
		"""
		[1] M. Hofert, 'On the Multivariate t Distribution,' R J., vol. 5, pp. 129-136, 2013.

		Conditional probabilities in a Joint Multivariate t Distribution Mixture Model

		:param data_in:		[np.array([nb_data, nb_dim])
				Observed datapoints x_in
		:param dim_in:		[slice] or [list of index]
				Dimension of input space e.g.: slice(0, 3), [0, 2, 3]
		:param dim_out:		[slice] or [list of index]
				Dimension of output space e.g.: slice(3, 6), [1, 4]
		:param h:			optional - [np.array([nb_states, nb_data])]
				Overrides marginal probability of states given input dimensions
		:return:
		"""

		sample_size = data_in.shape[0]

		# compute marginal probabilities of states given observation p(k|x_in)
		mu_in, sigma_in = self.get_marginal(dim_in)

		h = np.zeros((self.nb_states, sample_size))
		for i in range(self.nb_states):
			h[i, :] = multi_variate_t(data_in, self.nu[i],
										   mu_in[i],
										   sigma_in[i])

		h += np.log(self.priors)[:, None]
		h = np.exp(h).T
		h /= np.sum(h, axis=1, keepdims=True)
		h = h.T

		self._h = h # storing value

		mu_out, sigma_out = self.get_marginal(dim_out)  # get marginal distribution of x_out
		mu_est, sigma_est = ([], [])

		# get conditional distribution of x_out given x_in for each states p(x_out|x_in, k)
		inv_sigma_in_in, inv_sigma_out_in = ([], [])

		_, sigma_in_out = self.get_marginal(dim_in, dim_out)

		_as = []

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

			_as += [a]

		mu_est, sigma_est = (np.asarray(mu_est), np.asarray(sigma_est))

		a = np.einsum('ia,ia->a', h, _as)

		_, _covs = gaussian_moment_matching(mu_est, sigma_est, h.T)

		# return a
		return np.linalg.det(_covs)

		# the conditional distribution is now a still a mixture


class VBayesianGMM(MTMM):
	def __init__(self, sk_parameters, *args, **kwargs):
		"""
		self.model = tff.VBayesianGMM(
			{'n_components':5, 'n_init':4, 'reg_covar': 0.006 ** 2,
         'covariance_prior': 0.02 ** 2 * np.eye(12),'mean_precision_prior':1e-9})

		:param sk_parameters:
		:param args:
		:param kwargs:
		"""
		MTMM.__init__(self, *args, **kwargs)

		from sklearn import mixture

		self._training_data = None
		self._posterior_predictive = None


		self._sk_model = mixture.BayesianGaussianMixture(**sk_parameters)
		self._posterior_samples = None

	@property
	def model(self):
		return self._sk_model

	@property
	def posterior_samples(self):
		return self._posterior_samples

	def make_posterior_samples(self, nb_samples=10):
		from scipy.stats import wishart
		from .gmm import GMM
		self._posterior_samples = []

		m = self._sk_model

		nb_states = m.means_.shape[0]

		for i in range(nb_samples):
			_gmm = GMM()

			_gmm.lmbda = np.array(
				[wishart.rvs(m.degrees_of_freedom_[i],
							 np.linalg.inv(m.covariances_[i] * m.degrees_of_freedom_[i]))
				 for i in range(nb_states)])

			_gmm.mu = np.array(
				[np.random.multivariate_normal(
					m.means_[i], np.linalg.inv(m.mean_precision_[i] * _gmm.lmbda[i])
				)
				for i in range(nb_states)])

			_gmm.priors = m.weights_

			self._posterior_samples += [_gmm]

	def posterior(self, data, mean_scale=10., cov=None, dp=True):

		self.nb_dim = data.shape[1]

		self._sk_model.fit(data)

		states = np.where(self._sk_model.weights_ > -5e-2)[0]

		self.nb_states = states.shape[0]
		# see [1] K. P. Murphy, 'Conjugate Bayesian analysis of the Gaussian distribution,' vol. 0, no. 7, 2007. par 9.4
		# or [1] E. Fox, 'Bayesian nonparametric learning of complex dynamical phenomena,' 2009, p

		# m.covariances_ = W_k_^-1/m.degrees_of_freedom_
		m = self._sk_model

		self.priors = np.copy(m.weights_[states])
		self.mu = np.copy(m.means_[states])
		self.k = np.copy(m.mean_precision_[states])

		self.nu = np.copy(m.degrees_of_freedom_[states]) - self.nb_dim + 1


		w_k = np.linalg.inv(m.covariances_ * m.degrees_of_freedom_[:, None, None])
		l_k = ((m.degrees_of_freedom_[:, None, None] + 1 - self.nb_dim) * m.mean_precision_[:, None, None])/ \
			  (1. + m.mean_precision_[:, None, None]) * w_k

		self.sigma = np.copy(np.linalg.inv(l_k))[states]

		# self.sigma = np.copy(self._sk_model.covariances_[states]) * (
		# self.k[:, None, None] + 1) * self.nu[:, None, None] \
		# 			 / (self.k[:, None, None] * (self.nu[:, None, None] - self.nb_dim + 1))

		# add new state, base measure TODO make not heuristic

		if dp:
			self.priors = np.concatenate([self.priors, 0.02 * np.ones((1,))], 0)
			self.priors /= np.sum(self.priors)

			self.mu = np.concatenate([self.mu, np.zeros((1, self.nb_dim))], axis=0)
			if cov is None:
				cov = mean_scale ** 2 * np.eye(self.nb_dim)

			self.sigma = np.concatenate([self.sigma, cov[None]], axis=0)

			self.k = np.concatenate([self.k, np.ones((1, ))], axis=0)

			_nu_p = self._sk_model.degrees_of_freedom_prior_

			self.nu = np.concatenate([self.nu, _nu_p * np.ones((1, ))], axis=0)

			self.nb_states = states.shape[0] + 1

		# add new state, base measure TODO make not heuristic

		if dp:
			self.priors = np.concatenate([self.priors, 0.02 * np.ones((1,))], 0)
			self.priors /= np.sum(self.priors)

			self.mu = np.concatenate([self.mu, np.zeros((1, self.nb_dim))], axis=0)
			if cov is None:
				cov = mean_scale ** 2 * np.eye(self.nb_dim)

			self.sigma = np.concatenate([self.sigma, cov[None]], axis=0)

			self.k = np.concatenate([self.k, np.ones((1, ))], axis=0)
			self.nu = np.concatenate([self.nu, np.ones((1, ))], axis=0)

			self.nb_states = states.shape[0] + 1

	def condition(self, *args, **kwargs):
		"""
		[1] M. Hofert, 'On the Multivariate t Distribution,' R J., vol. 5, pp. 129-136, 2013.

		Conditional probabilities in a Joint Multivariate t Distribution Mixture Model

		:param data_in:		[np.array([nb_data, nb_dim])
				Observed datapoints x_in
		:param dim_in:		[slice] or [list of index]
				Dimension of input space e.g.: slice(0, 3), [0, 2, 3]
		:param dim_out:		[slice] or [list of index]
				Dimension of output space e.g.: slice(3, 6), [1, 4]
		:param h:			optional - [np.array([nb_states, nb_data])]
				Overrides marginal probability of states given input dimensions
		:return:
		"""
		if not kwargs.get('samples', False):
			return MTMM.condition(self, *args, **kwargs)
		kwargs.pop('samples')

		mus, sigmas = [], []

		for _gmm in self.posterior_samples:
			mu, sigma = _gmm.condition(*args, **kwargs)
			mus += [mu]; sigmas += [sigma]

		mus, sigmas = np.array(mus), np.array(sigmas)
		# moment matching
		mu = np.mean(mus, axis=0)
		dmu = mu[None] - mus
		sigma = np.mean(sigmas, axis=0) + \
				np.einsum('aki,akj->kij', dmu, dmu) / len(self.posterior_samples)

		return mu, sigma

class VMBayesianGMM(VBayesianGMM):
	def __init__(self, n, sk_parameters, *args, **kwargs):
		"""
		Multimodal posterior approximation using a several training

		self.model = tff.VMBayesianGMM(
			{'n_components':5, 'n_init':4, 'reg_covar': 0.006 ** 2,
         'covariance_prior': 0.02 ** 2 * np.eye(12),'mean_precision_prior':1e-9})

		:param n:  	number of evaluations
		:param sk_parameters:
		:param args:
		:param kwargs:
		"""

		self.models = [VBayesianGMM(sk_parameters, *args, **kwargs) for i in range(n)]
		self.n = n
		self._training_data = None

	def posterior(self, data, *args, **kwargs):
		for model in self.models:
			model.posterior(data, *args, **kwargs)

	def condition(self, *args, **kwargs):
		params = []

		for model in self.models:
			params += [model.condition(*args, **kwargs)]

		# TODO check how to compute priors in a good way
		params = zip(*params)
		mu, sigma = gaussian_moment_matching(np.array(params[0]),
													   np.array(params[1]))

		return mu, sigma

	@property
	def nb_states(self):
		return [model.nb_states for model in self.models]

	def plot(self, *args, **kwargs):
		import matplotlib
		cmap = kwargs.pop('cmap', 'viridis')

		colors = matplotlib.cm.get_cmap(cmap, self.n)

		for i, model in enumerate(self.models):
			color = colors(i)
			kwargs['color'] = [color[i] for i in range(3)]
			model.plot(*args, **kwargs)