import numpy as np
from .gmm import GMM, MVN
from .hmm import HMM
from functions import multi_variate_normal, multi_variate_t
from utils import gaussian_moment_matching
from scipy.special import gamma, gammaln, logsumexp

class MTMM(GMM):
	"""
	Multivariate t-distribution mixture
	"""

	def __init__(self, *args, **kwargs):
		self._nu = kwargs.pop('nu', None)
		GMM.__init__(self, *args, **kwargs)

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

	def marginal_model(self, dims):
		mtmm = MTMM(nb_dim=dims.stop - dims.start, nb_states=self.nb_states)
		mtmm.priors = self.priors
		mtmm.mu = self.mu[:, dims]
		mtmm.sigma = self.sigma[:, dims, dims]
		mtmm.nu = self.nu

		return mtmm

	def get_matching_gmm(self):
		return GMM(mu=self.mu, sigma=self.sigma * (self.nu/(self.nu-2.))[:, None, None],
				   priors=self.priors)

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

	def log_prob(self, x):
		return logsumexp(self.log_prob_components(x) + np.log(self.priors)[:, None], axis=0)

	def log_prob_components(self, x):
		dx = self.mu[:, None] - x[None]  # [nb_states, nb_samples, nb_dim]

		# slower
		# s = np.sum(np.einsum('kij,kai->kaj', self.lmbda, dx) * dx, axis=2) # [nb_states, nb_samples]

		# faster
		s = np.sum(np.matmul(self.lmbda[:, None], dx[:, :, :, None])[:, :, :, 0] * dx, axis=2) # [nb_states, nb_samples]

		log_norm = self.log_normalization[:, None]
		return log_norm + (-(self.nu + self.nb_dim) / 2)[:, None] * np.log(1 + s/ self.nu[:, None])

	def obs_likelihood(self, demo=None, dep=None, marginal=None, *args, **kwargs):
		B = self.log_prob_components(demo)
		return np.exp(B), B
	@property
	def log_normalization(self):
		if self._log_normalization is None:
			self._log_normalization = gammaln((self.nu + self.nb_dim) / 2) + 0.5 * np.linalg.slogdet(self.lmbda)[1] - \
				gammaln(self.nu / 2) - self.nb_dim / 2. * (np.log(self.nu) + np.log(np.pi))

		return self._log_normalization

	# @profile
	def condition(self, data_in, dim_in, dim_out, h=None, return_gmm=False, reg_in=1e-20,
				  concat=True, return_linear=False, tmp=False):
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

		if data_in.ndim == 1:
			data_in = data_in[None]
			was_not_batch = True
		else:
			was_not_batch = False

		sample_size = data_in.shape[0]

		if tmp and hasattr(self, '_tmp_slices') and not self._tmp_slices == (dim_in, dim_out):
			del self._tmp_inv_sigma_out_in, self._tmp_inv_sigma_in_in, self._tmp_slices, self._tmp_marginal_model

		# compute marginal probabilities of states given observation p(k|x_in)
		mu_in, sigma_in = self.get_marginal(dim_in)

		if tmp and hasattr(self, '_tmp_marginal_model'):
			marginal_model = self._tmp_marginal_model
		else:
			marginal_model = self.marginal_model(dim_in)
			if tmp:
				self._tmp_marginal_model = marginal_model

		if h is None:
			h = marginal_model.log_prob_components(data_in)
			h += np.log(self.priors)[:, None]
			h = np.exp(h).T
			h /= np.sum(h, axis=1, keepdims=True)

		#[nb_samples, nb_states]
		self._h = h # storing value

		mu_out, sigma_out = self.get_marginal(dim_out)  # get marginal distribution of x_out

		# get conditional distribution of x_out given x_in for each states p(x_out|x_in, k)

		_, sigma_in_out = self.get_marginal(dim_in, dim_out)

		if not concat: # faster when more datapointsS
			mu_est, sigma_est = ([], [])
			inv_sigma_in_in, inv_sigma_out_in = ([], [])

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
		else:
			# test if slices change and reset

			if tmp and hasattr(self, '_tmp_inv_sigma_in_in'):
				inv_sigma_in_in = self._tmp_inv_sigma_in_in
				inv_sigma_out_in = self._tmp_inv_sigma_out_in
			else:
				inv_sigma_in_in = np.linalg.inv(sigma_in + reg_in * np.eye(sigma_in.shape[-1])[None])
				inv_sigma_out_in = np.einsum('aji,ajk->aik', sigma_in_out, inv_sigma_in_in)

			if tmp and not hasattr(self, '_tmp_inv_sigma_in_in'):
				self._tmp_inv_sigma_in_in = inv_sigma_in_in
				self._tmp_inv_sigma_out_in = inv_sigma_out_in
				self._tmp_slices = (dim_in, dim_out)

			# [nb_states, nb_sample, nb_dim]
			dx = data_in[None] - mu_in[:, None]

			# mu_est = mu_out[:, None] + np.einsum('aij,abj->abi', inv_sigma_out_in, dx)
			mu_est = mu_out[:, None] + np.matmul(inv_sigma_out_in[:, None], dx[:, :, :, None])[:, :, :, 0]

			s = np.sum(np.matmul(inv_sigma_in_in[:, None], dx[:, :, :, None])[:, :, :, 0] * dx,
					   axis=2)
			# s = np.sum(np.einsum('kij,kai->kaj',inv_sigma_in_in, dx) * dx, axis=2)

			a = (self.nu[:, None] + s) / (self.nu[:, None] + mu_in.shape[1])

			# sigma_est = a[:, :, None, None] * (sigma_out - np.einsum('aij,ajk->aik', inv_sigma_out_in, sigma_in_out))[:, None]
			sigma_est = a[:, :, None, None] * (sigma_out - np.matmul(inv_sigma_out_in, sigma_in_out))[:, None]

		nu = self.nu + mu_in.shape[1]
		# the conditional distribution is now a still a mixture

		if return_gmm:
			return h, mu_est, sigma_est * (nu/(nu-2.))[:, None, None, None]
		elif return_linear:
			As = inv_sigma_out_in
			bs = mu_out - np.matmul(inv_sigma_out_in, mu_in[:, :, None])[:, :, 0]
			A = np.einsum('ak,kij->aij', h, As)
			b = np.einsum('ak,ki->ai', h, bs)
			if was_not_batch:
				return A[0], b[0], gaussian_moment_matching(mu_est, sigma_est * (nu/(nu-2.))[:, None, None, None], h)[1][0]
			else:
				return A, b, gaussian_moment_matching(mu_est, sigma_est * (nu/(nu-2.))[:, None, None, None], h)[1]
		else:
			# apply moment matching to get a single MVN for each datapoint
			return gaussian_moment_matching(mu_est, sigma_est * (nu/(nu-2.))[:, None, None, None], h)

	def get_pred_post_uncertainty(self, data_in, dim_in, dim_out, log=False):
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
		if log:
			return np.linalg.slogdet(_covs)[1]
		else:
			return np.linalg.det(_covs)
		# the conditional distribution is now a still a mixture

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
				[wishart.rvs(m.degrees_of_freedom_[i] + 1.,
							 np.linalg.inv(m.covariances_[i] * m.degrees_of_freedom_[i]))
				 for i in range(nb_states)])

			_gmm.mu = np.array(
				[np.random.multivariate_normal(
					m.means_[i], np.linalg.inv(m.mean_precision_[i] * _gmm.lmbda[i])
				)
				for i in range(nb_states)])

			_gmm.priors = m.weights_

			self._posterior_samples += [_gmm]

	def get_used_states(self):
		keep = self.nu + self.nb_dim - 1.01 > self.nu_prior
		return MTMM(mu=self.mu[keep], lmbda=self.lmbda[keep],
					sigma=self.sigma[keep], nu=self.nu[keep], priors=self.priors[keep])

	def posterior(self, data, mean_scale=10., cov=None, dp=True):

		self.nb_dim = data.shape[1]

		self._sk_model.fit(data)

		self.nb_states = self._sk_model.weights_.shape[0]
		# see [1] K. P. Murphy, 'Conjugate Bayesian analysis of the Gaussian distribution,' vol. 0, no. 7, 2007. par 9.4
		# or [1] E. Fox, 'Bayesian nonparametric learning of complex dynamical phenomena,' 2009, p

		# m.covariances_ = W_k_^-1/m.degrees_of_freedom_
		m = self._sk_model

		self.priors = np.copy(m.weights_)
		self.mu = np.copy(m.means_)
		self.k = np.copy(m.mean_precision_)

		self.nu = np.copy(m.degrees_of_freedom_) - self.nb_dim + 1

		self.nu_prior = m.degrees_of_freedom_prior

		w_k = np.linalg.inv(m.covariances_ * m.degrees_of_freedom_[:, None, None])
		l_k = ((m.degrees_of_freedom_[:, None, None] + 1 - self.nb_dim) * m.mean_precision_[:, None, None])/ \
			  (1. + m.mean_precision_[:, None, None]) * w_k

		self.sigma = np.copy(np.linalg.inv(l_k))

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
			kwargs.pop('return_samples', True)
			return MTMM.condition(self, *args, **kwargs)
		kwargs.pop('samples')
		return_samples = kwargs.pop('return_samples', False)
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

		if return_samples:
			return mu, sigma, mus
		else:
			return mu, sigma

class VBayesianHMM(VBayesianGMM, HMM):
	def __init__(self, *args, **kwargs):
		VBayesianGMM.__init__(self, *args, **kwargs)
		self._trans = None
		self._init_priors = None

	def obs_likelihood(self, demo=None, dep=None, marginal=None, *args, **kwargs):
		return VBayesianGMM.obs_likelihood(self, demo=demo, dep=dep, marginal=marginal)

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