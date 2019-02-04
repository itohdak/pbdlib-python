import numpy as np
from utils.utils import lifted_transfer_matrix
import pbdlib as pbd

class LQR(object):
	def __init__(self, A=None, B=None, nb_dim=2, dt=0.01, horizon=50):
		self._horizon = horizon
		self.A = A
		self.B = B
		self.dt = dt

		self.nb_dim = nb_dim

		self._s_xi, self._s_u = None, None
		self._x0 = None

		self._gmm_xi, self._gmm_u = None, None
		self._mvn_sol_xi, self._mvn_sol_u = None, None

		self._seq_xi, self._seq_u = None, None

		self._S, self._v, self._K, self._Kv, self._ds, self._cs , self._Q = \
			None, None, None, None, None, None, None

	@property
	def K(self):
		assert self._K is not None, "Solve Ricatti before"

		return self._K

	@property
	def Q(self):
		assert self._Q is not None, "Solve Ricatti before"

		return self._Q

	@property
	def cs(self):
		"""
		Return c list where control command u is
			u = -K x + c

		:return:
		"""
		if self._cs is None:
			self._cs = self.get_feedforward()

		return self._cs

	@property
	def ds(self):
		"""
		Return c list where control command u is
			u = K(d - x)

		:return:
		"""
		if self._ds is None:
			self._ds = self.get_target()

		return self._ds

	@property
	def horizon(self):
		return self._horizon

	@horizon.setter
	def horizon(self, value):
		self.reset_params()

		self._horizon = value

	@property
	def u_dim(self):
		"""
		Number of dimension of input
		:return:
		"""
		if self.B is not None:
			return self.B.shape[1]
		else:
			return self.nb_dim

	@property
	def xi_dim(self):

		"""
		Number of dimension of state
		:return:
		"""
		if self.A is not None:
			return self.A.shape[0]
		else:
			return self.nb_dim * 2


	@property
	def gmm_xi(self):
		"""
		Distribution of state
		:return:
		"""
		return self._gmm_xi

	@gmm_xi.setter
	def gmm_xi(self, value):
		"""
		:param value 		[pbd.GMM] or [(pbd.GMM, list)]
		"""
		# resetting solution
		self._mvn_sol_xi = None
		self._mvn_sol_u = None
		self._seq_u = None
		self._seq_xi = None

		self._gmm_xi = value

	@property
	def gmm_u(self):
		"""
		Distribution of control input
		:return:
		"""
		return self._gmm_u

	@gmm_u.setter
	def gmm_u(self, value):
		"""
		:param value 		[float] or [pbd.MVN] or [pbd.GMM] or [(pbd.GMM, list)]
		"""
		# resetting solution
		self._mvn_sol_xi = None
		self._mvn_sol_u = None
		self._seq_u = None
		self._seq_xi = None

		if isinstance(value, float):
			self._gmm_u = pbd.MVN(
				mu=np.zeros(self.u_dim), lmbda=10 ** value * np.eye(self.u_dim))
		else:
			self._gmm_u = value

	@property
	def x0(self):
		return self._x0

	@x0.setter
	def x0(self, value):
		# resetting solution
		self._mvn_sol_xi = None
		self._mvn_sol_u = None

		self._x0 = value

	def get_Q_z(self, t):
		"""
		get Q and target z for time t
		:param t:
		:return:
		"""
		if isinstance(self._gmm_xi, tuple):
			gmm, seq = self._gmm_xi
			return gmm.lmbda[seq[t]], gmm.mu[seq[t]]
		elif isinstance(self._gmm_xi, pbd.GMM):
			return self._gmm_xi.lmbda[t], self._gmm_xi.mu[t]
		elif isinstance(self._gmm_xi, pbd.MVN):
			return self._gmm_xi.lmbda, self._gmm_xi.mu
		else:
			raise ValueError, "Not supported gmm_xi"

	def get_R(self, t):
		if isinstance(self._gmm_u, pbd.MVN):
			return self._gmm_u.lmbda
		elif isinstance(self._gmm_u, tuple):
			gmm, seq = self._gmm_u
			return gmm.lmbda[seq[t]]
		elif isinstance(self._gmm_u, pbd.GMM):
			return self._gmm_u.lmbda[t]
		else:
			raise ValueError, "Not supported gmm_u"

	def ricatti(self):
		"""
		http://web.mst.edu/~bohner/papers/tlqtots.pdf
		:return:
		"""

		Q, z = self.get_Q_z(-1)
		#
		_S = [None for i in range(self._horizon)]
		_v = [None for i in range(self._horizon)]
		_K = [None for i in range(self._horizon-1)]
		_Kv = [None for i in range(self._horizon-1)]
		_Q = [None for i in range(self._horizon-1)]
		# _S = np.empty((self._horizon, self.xi_dim, self.xi_dim))
		# _v = np.empty((self._horizon, self.xi_dim))
		# _K = np.empty((self._horizon-1, self.u_dim, self.xi_dim))
		# _Kv = np.empty((self._horizon-1, self.u_dim, self.xi_dim))

		_S[-1] = Q
		_v[-1] = Q.dot(z)

		for t in range(self.horizon-2, -1, -1):
			Q, z = self.get_Q_z(t)
			R = self.get_R(t)

			_Q[t] = np.linalg.inv(R + self.B.T.dot(_S[t+1]).dot(self.B))
			_Kv[t] = _Q[t].dot(self.B.T)
			_K[t] = _Kv[t].dot(_S[t+1]).dot(self.A)

			AmBK = self.A - self.B.dot(_K[t])

			_S[t] = self.A.T.dot(_S[t+1]).dot(AmBK) + Q
			_v[t] = AmBK.T.dot(_v[t+1]) + Q.dot(z)

		self._S = _S
		self._v = _v
		self._K = _K
		self._Kv = _Kv
		self._Q = _Q

		self._ds = None
		self._cs = None

	def get_target(self):
		ds = []

		for t in range(0, self.horizon-1):
			ds += [np.linalg.inv(self._S[t].dot(self.A)).dot(self._v[t])]

		return np.array(ds)

	def get_feedforward(self):
		cs = []

		for t in range(0, self.horizon-1):
			cs += [self._Kv[t].dot(self._v[t+1])]

		return np.array(cs)

	def get_seq(self, xi0, return_target=False):
		xis = [xi0]
		us = [-self._K[0].dot(xi0) + self._Kv[0].dot(self._v[0])]

		ds = []

		for t in range(1, self.horizon-1):
			xis += [self.A.dot(xis[-1]) + self.B.dot(us[-1])]

			if return_target:
				d = np.linalg.inv(self._S[t].dot(self.A)).dot(self._v[t])
				ds += [d]

				us += [self._K[t].dot(d-xis[-1])]
			else:
				us += [-self._K[t].dot(xis[-1]) + self._Kv[t].dot(self._v[t+1])]

		if return_target:
			return np.array(xis), np.array(us), np.array(ds)
		else:
			return np.array(xis), np.array(us)

class GMMLQR(LQR):
	"""
	LQR with a GMM cost on the state, approximation to be checked
	"""

	def __init__(self, *args, **kwargs):
		self._full_gmm_xi = None
		LQR.__init__(self, *args, **kwargs)

	@property
	def full_gmm_xi(self):
		"""
		Distribution of state
		:return:
		"""
		return self._full_gmm_xi

	@full_gmm_xi.setter
	def full_gmm_xi(self, value):
		"""
		:param value 		[pbd.GMM] or [(pbd.GMM, list)]
		"""
		self._full_gmm_xi = value

	def ricatti(self, x0, n_best=None):
		costs = []

		if isinstance(self._full_gmm_xi, pbd.MTMM):
			full_gmm = self.full_gmm_xi.get_matching_gmm()
		else:
			full_gmm = self.full_gmm_xi

		if n_best is not None:
			log_prob_components = self.full_gmm_xi.log_prob_components(x0)
			a = np.sort(log_prob_components, axis=0)[-n_best - 1][0]

		for i in range(self.full_gmm_xi.nb_states):
			if n_best is not None and log_prob_components[i] <a:
				costs += [-np.inf]
			else:
				self.gmm_xi = full_gmm, [i for j in range(self.horizon)]
				LQR.ricatti(self)
				xis, us = self.get_seq(x0)
				costs += [np.sum(self.gmm_u.log_prob(us) + self.full_gmm_xi.log_prob(xis))]

		max_lqr = np.argmax(costs)
		self.gmm_xi = full_gmm, [max_lqr for j in range(self.horizon)]
		LQR.ricatti(self)



class PoGLQR(LQR):
	"""
	Implementation of LQR with Product of Gaussian as described in

		http://calinon.ch/papers/Calinon - HFR2016.pdf

	"""

	def __init__(self, A=None, B=None, nb_dim=2, dt=0.01, horizon=50):
		self._horizon = horizon
		self.A = A
		self.B = B
		self.nb_dim = nb_dim
		self.dt = dt

		self._s_xi, self._s_u = None, None
		self._x0 = None

		self._mvn_xi, self._mvn_u = None, None
		self._mvn_sol_xi, self._mvn_sol_u = None, None

		self._seq_xi, self._seq_u = None, None

	@property
	def A(self):
		return self._A

	@A.setter
	def A(self, value):
		self.reset_params() # reset params
		self._A = value

	@property
	def B(self):
		return self._B

	@B.setter
	def B(self, value):
		self.reset_params() # reset params
		self._B = value

	@property
	def mvn_u_dim(self):
		"""
		Number of dimension of input sequence lifted form
		:return:
		"""
		if self.B is not None:
			return self.B.shape[1] * self.horizon
		else:
			return self.nb_dim * self.horizon

	@property
	def mvn_xi_dim(self):

		"""
		Number of dimension of state sequence lifted form
		:return:
		"""
		if self.A is not None:
			return self.A.shape[0] * self.horizon
		else:
			return self.nb_dim * self.horizon * 2


	@property
	def mvn_sol_u(self):
		"""
		Distribution of control input after solving LQR
		:return:
		"""
		assert self.x0 is not None, "Please specify a starting state"
		assert self.mvn_xi is not None, "Please specify a target distribution"
		assert self.mvn_u is not None, "Please specify a control input distribution"

		if self._mvn_sol_u is None:
			self._mvn_sol_u =  self.mvn_xi.inv_trans_s(
				self.s_u, self.s_xi.dot(self.x0)) % self.mvn_u

		return self._mvn_sol_u

	@property
	def seq_xi(self):
		if self._seq_xi is None:
			self._seq_xi =  self.mvn_sol_xi.mu.reshape(self.horizon, self.xi_dim)

		return self._seq_xi

	@property
	def seq_u(self):
		if self._seq_u is None:
			self._seq_u = self.mvn_sol_u.mu.reshape(self.horizon, self.u_dim)

		return self._seq_u


	@property
	def mvn_sol_xi(self):
		"""
		Distribution of state after solving LQR
		:return:
		"""
		if self._mvn_sol_xi is None:
			self._mvn_sol_xi= self.mvn_sol_u.transform(
				self.s_u, self.s_xi.dot(self.x0))

		return self._mvn_sol_xi

	@property
	def mvn_xi(self):
		"""
		Distribution of state
		:return:
		"""
		return self._mvn_xi

	@mvn_xi.setter
	def mvn_xi(self, value):
		"""
		:param value 		[float] or [pbd.MVN]
		"""
		# resetting solution
		self._mvn_sol_xi = None
		self._mvn_sol_u = None
		self._seq_u = None
		self._seq_xi = None

		self._mvn_xi = value

	@property
	def mvn_u(self):
		"""
		Distribution of control input
		:return:
		"""
		return self._mvn_u

	@mvn_u.setter
	def mvn_u(self, value):
		"""
		:param value 		[float] or [pbd.MVN]
		"""
		# resetting solution
		self._mvn_sol_xi = None
		self._mvn_sol_u = None
		self._seq_u = None
		self._seq_xi = None

		if isinstance(value, pbd.MVN):
			self._mvn_u = value
		else:
			self._mvn_u = pbd.MVN(
				mu=np.zeros(self.mvn_u_dim), lmbda=10 ** value * np.eye(self.mvn_u_dim))


	@property
	def xis(self):
		return self.mvn_sol_xi.mu.reshape(self.horizon, self.xi_dim/self.horizon)


	@property
	def k(self):
		# return self.mvn_sol_u.sigma.dot(self.s_u.T.dot(self.mvn_xi.lmbda)).dot(self.s_xi).reshape(
		return self.mvn_sol_u.sigma.dot(self.s_u.T.dot(self.mvn_xi.lmbda)).dot(self.s_xi).reshape(
			(self.horizon, self.mvn_u_dim/self.horizon, self.mvn_xi_dim/self.horizon))

	@property
	def s_u(self):
		if self._s_u is None:
			self._s_xi, self._s_u = lifted_transfer_matrix(self.A, self.B,
				horizon=self.horizon, dt=self.dt, nb_dim=self.nb_dim)
		return self._s_u
	@property
	def s_xi(self):
		if self._s_xi is None:
			self._s_xi, self._s_u = lifted_transfer_matrix(self.A, self.B,
				horizon=self.horizon, dt=self.dt, nb_dim=self.nb_dim)

		return self._s_xi

	def reset_params(self):
		# reset everything
		self._s_xi, self._s_u = None, None
		self._x0 = None
		# self._mvn_xi, self._mvn_u = None, None
		self._mvn_sol_xi, self._mvn_sol_u = None, None
		self._seq_xi, self._seq_u = None, None

	@property
	def horizon(self):
		return self._horizon

	@horizon.setter
	def horizon(self, value):
		self.reset_params()


		self._horizon = value