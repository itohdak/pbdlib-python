import numpy as np
prec_min = 1e-15

class MVN(object):
	def __init__(self, mu=None, sigma=None, lmbda=None, lmbda_ns=None, sigma_cv=None):
		"""

		:param mu:		np.array([nb_dim])
			Mean vector
		:param sigma: 	np.array([nb_dim, nb_dim])
			Covariance matrix
		:param lmbda: 	np.array([nb_dim, nb_dim])
			Precision matrix
		:param lmbda_ns:
		:param sigma_cv:
		"""

		self.mu = mu
		self.sigma = sigma

		if sigma is None and lmbda is not None:
			self.sigma = np.linalg.inv(lmbda)
			self.lmbda = lmbda
		elif lmbda is None and sigma is not None:
			self.lmbda = np.linalg.inv(self.sigma)
		else:
			self.lmbda = lmbda

		#
		self.lmbda_ns = lmbda_ns
		self.sigma_cv = sigma_cv

		self.nb_dim = self.mu.shape[0]

	def transform(self, A, b, dA=None, db=None):
		if dA is None:
			return MVN(mu=A.dot(self.mu) + b, sigma=A.dot(self.sigma).dot(A.T))

		else:
			if self.nb_dim == 2:
				Sr = dA * np.array([[0, -1.], [1., 0.]])
			elif self.nb_dim == 3:
				Sr = np.array([[0., -1., 1.],[1., 0., -1.],[-1., 1., 0.]])
			else:
				print "Did you just invented 4 dimensional space ???"
				raise ValueError


			ds = Sr.dot(A).dot(self.mu)
			ds = np.outer(ds, ds)

			return MVN(mu=A.dot(self.mu) + b,
					   sigma=A.dot(self.sigma).dot(A.T) + ds + db * np.eye(2))

	def transf(self, A, b, min_precision=False):
		self.mu = np.linalg.pinv(A).dot(self.mu - b)
		self.lmbda = A.T.dot(self.lmbda).dot(A)

		if min_precision:
			self.sigma = np.linalg.inv(self.lmbda + prec_min * np.eye(self.lmbda.shape[0]))
		else:
			self.sigma = np.linalg.inv(self.lmbda)

		if self.lmbda_ns is not None:
			self.lmbda_ns = A.T.dot(self.lmbda_ns).dot(A)

		if self.sigma_cv is not None:
			self.sigma_cv = A.T.dot(self.sigma_cv).dot(A)

		if min_precision is not None:
			self.lmbda += np.eye(self.lmbda.shape[0]) * min_precision

			if self.lmbda_ns is not None:
				self.lmbda_ns += np.eye(self.lmbda.shape[0]) * min_precision



	def __mul__(self, other):
		"""
		Standart product of MVN
		:param other:
		:return:
		"""
		assert all([self.lmbda is not None, other.lmbda is not None]), "Precision not defined"


		prod = MVN(self.mu)
		prod.mu = self.lmbda.dot(self.mu) + other.lmbda.dot(other.mu)
		prod.lmbda = self.lmbda + other.lmbda
		prod.sigma = np.linalg.inv(prod.lmbda)

		prod.mu = prod.sigma.dot(prod.mu)
		# prod.mu = np.linalg.solve(prod.lmbda, prod.mu)

		return prod

	def __rmul__(self, other):
		"""
		Standart product of MVN
		:param other:
		:return:
		"""
		return self.__mul__(other, self)

	def __pow__(self, other):
		"""
		Product of MVN with bluffing experts
		:param power:
		:param modulo:
		:return:
		"""
		assert all([self.lmbda_ns is not None, other.lmbda_ns is not None]), "Bluffing precision not defined"

		prod = MVN(self.mu)

		prod.mu = self.lmbda_ns.dot(self.mu) + other.lmbda_ns.dot(other.mu)
		prod.lmbda_ns = self.lmbda_ns + other.lmbda_ns

		prod.lmbda = self.lmbda + other.lmbda
		prod.sigma = np.linalg.inv(prod.lmbda)

		prod.mu = np.linalg.inv(prod.lmbda_ns).dot(prod.mu)


		return prod

	def __mod__(self, other):
		"""
		Product of MVN with bluffing experts
		:param power:
		:param modulo:
		:return:
		"""
		assert all([self.sigma_cv is not None, other.sigma_cv is not None]), "Bluffing precision not defined"

		prod = MVN(self.mu)

		prod.mu = (self.sigma_cv + self.lmbda).dot(self.mu) + (other.sigma_cv + other.lmbda).dot(other.mu)
		prod.lmbda_cv = self.lmbda + self.sigma_cv + other.lmbda + other.sigma_cv

		prod.sigma_cv = self.sigma_cv + other.sigma_cv

		# prod.lmbda = self.lmbda + other.lmbda
		prod.lmbda = prod.lmbda_cv
		prod.sigma = np.linalg.inv(prod.lmbda)

		prod.mu = np.linalg.inv(prod.lmbda_cv).dot(prod.mu)


		return prod
