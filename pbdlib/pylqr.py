import numpy as np
from scipy.special import factorial
import numpy.linalg as ln
from enum import Enum

class LQR_type(Enum):
	standard = 0
	canonical = 1
	open_loop = 2

class PyLQR(object):
	def __init__(self, A=None, B=None, dt=0.01, canonical=False, nb_dim=2, rFactor=-6,
				 horizon=50, nb_features=1, rozo=False, jerk=True, discrete=False, nb_ctrl=None, R=None):
		"""

		:param A:			[np.array()]
		:param B:			[np.array()]
			Dynamical system parameters, if you want to provide personalized ones.
		:param dt:
		:param canonical: 	[bool]
			Automatically set dynamic system to canonical
		:param nb_dim:		[int]
			Number of dimension of the space (2 or 3)
		:param rFactor:		[float]
			Control cost
		:param horizon:		[int]
			Number of timestep for horizon
		:param nb_features:	[int]
			Number of features used for reproduction (1 pos only, 2 pos-vel, 3 pos-vel-acc)
		:param jerk:
		:param discrete:	[bool]
			Use LQR in discrete timestep
		:param nb_ctrl:
		:param R:
		"""
		# let the user choose his A model  and B control matrix
		self.discrete = discrete
		self.dt = dt

		if canonical:
			self.type = LQR_type.canonical

		if A is not None and B is not None:
			self.A = A
			self.B = B
			self.nb_var = self.A.shape[0]
			self.r = R

			self.horizon = horizon

			self.target = np.zeros((A.shape[0], horizon))

			self.force_target = np.zeros((B.shape[1], horizon))

			self.nb_dim = A.shape[0]
			self.nb_ctrl = B.shape[1]

			self.rData = np.zeros((self.nb_dim, self.horizon))
			return

		if canonical:
			if not discrete:
				A1d = np.zeros((2, 2))
				A1d[0, 1] = 1.0
				B1d = np.array([0., 1.])[:, np.newaxis]

				self.A = np.kron(A1d, np.eye(nb_dim))
				self.B = np.kron(B1d, np.eye(nb_dim))
			else:
				nb_deriv = 2
				A1d = np.zeros((nb_deriv, nb_deriv))

				for i in range(nb_deriv):
					A1d += np.diag(np.ones(nb_deriv-i),i) * np.power(self.dt, i) / factorial(i)

				B1d = np.zeros((nb_deriv, 1))
				for i in range(1, nb_deriv+1):
					B1d[nb_deriv-i] = np.power(self.dt, i) / factorial(i)

				self.A = np.kron(A1d, np.eye(nb_dim))
				self.B = np.kron(B1d, np.eye(nb_dim))

				if nb_ctrl is not None:
					Btmp = np.zeros((nb_dim*nb_features, nb_ctrl))
					Btmp[:nb_dim*nb_features, 0:3] = self.B
					self.B = Btmp

				# self.B = self.B.T
			# print self.B.shape

			self.nb_var = self.A.shape[0]
			self.r = np.eye(nb_dim) * 10**rFactor
			self.r_factor = rFactor


			self.horizon = horizon

			# initialize targets
			self.target = np.zeros((nb_dim*2, horizon))
			self.force_target = np.zeros((nb_dim, horizon)) # only used if you want to track a force, by default a zero force is tracked

			self.nb_dim = nb_dim
			self.nb_features = nb_features

			nb_rdata = self.nb_dim

			if nb_rdata < 4:
				nb_rdata *= 2

			self.rData = np.zeros((nb_rdata, self.horizon))

			if nb_features == 1:
				self.type = LQR_type.canonical
			else:
				self.type = LQR_type.open_loop

	def set_r_factor(self, r_factor):
		self.r_factor = r_factor
		self.r = np.eye(self.nb_dim) * 10 ** r_factor

	# @profile
	def evaluate_gains_finiteHorizon(self, use_python=False):
		self.use_python = use_python

		self.S = np.zeros((self.horizon, self.nb_var, self.nb_var))
		self.L = np.zeros((self.horizon, self.B.shape[1], self.nb_var))
		self.d = np.zeros((self.horizon, self.nb_var))

		self.S[-1,:,:] = self.Qp[-1]

		for t in range(self.horizon-2, -1, -1):
			Q = self.Qp[t]
			self.S[t,:,:] = Q - self.A.T.dot((
				self.S[t + 1, :, :]).dot(self.B).dot(
				ln.inv(self.B.T.dot(self.S[t+1,:,:]).dot(self.B) + self.r)).dot(
				self.B.T).dot(self.S[t+1,:,:]) - self.S[t+1,:,:]).dot(self.A)

		for t in range(self.horizon):
			self.L[t,:,:] = ln.inv(self.B.T.dot(self.S[t,:,:]).dot(self.B) + self.r).dot(
				self.B.T).dot(self.S[t,:,:]).dot(self.A)


	def evaluate_gains_infiniteHorizon(self, one_step=False, use_scipy=False, use_python=False):
		self.use_python = use_python + use_scipy

		invR = ln.inv(self.r)
		self.S = np.zeros((self.horizon, self.nb_var, self.nb_var))
		self.L = np.zeros((self.horizon, self.B.shape[1], self.nb_var))
		self.d = np.zeros((self.horizon, self.nb_var))

		for i in range(1):
			r = self.r
			self.S[i] = self.solve_algebraic_riccati(self.A, self.B, self.Qp[i], r) # is P in report
			self.L[i] = invR.dot(self.B.T).dot(self.S[i])

			if self.type is LQR_type.open_loop:
				self.d[i] = ln.inv(self.A.T - self.S[i].dot(self.B).dot(invR).dot(self.B.T))\
					.dot(self.S[i]).dot(self.B).dot(self.force_target[:,i])

	@staticmethod
	def solve_algebraic_riccati(A, B, Q, R):
		"""
		Solve algebraic riccati for discrete case
		:param A:
		:param B:
		:param Q:
		:param R:
		:return:
		"""
		n = A.shape[0]

		Z = np.empty((2*n, 2*n))
		G = (B.dot(ln.inv(R))).dot(B.T)

		Z[0:n, 0:n] = A
		Z[n:2*n, 0:n] = -Q
		Z[0:n, n:2*n] = -G
		Z[n:2*n, n:2*n] = -A.T

		U = np.empty((2*n, n), dtype=complex)

		dd, V = ln.eig(Z)
		i=0
		for j in range(2*n):
			if dd[j].real < 0:
				U[:, i] = V[:, j]
				i += 1

		try:
			Sc = ln.inv(U[0:n, 0:n].T).dot(U[n:2*n, 0:n].T)
			# print Sc.real
			return Sc.real
		except:
			print "Singular matrix"


	def set_hmm_problem(self, model, q, horizon=None, force_model=None, use_force=True):
		"""
		Set the LQR problem given a model and a sequence of states

		:param model: an HMM or HSMM model, if you want to use a TP-GMM you either
			have to make the product before or use a higher dimension that will give you
			gains relative to each reference frame
		:param q: np.array((nb_timestep,))
			vector of state indicator
		:return:
		"""
		Qtmp = np.zeros((self.nb_dim*2, self.nb_dim*2))

		if horizon is None:
			horizon = self.horizon

		if self.type in [LQR_type.canonical, LQR_type.standard]:
			self.Qp = np.zeros((self.horizon, self.nb_dim * 2, self.nb_dim * 2))

			for i in range(horizon):
				self.target[0:self.nb_dim,i] = model.Mu[0:self.nb_dim, q[i]]
				Qtmp[0:self.nb_dim, 0:self.nb_dim] = model.Lambda[0:self.nb_dim, 0:self.nb_dim, q[i]]
				self.Qp[i] = np.copy(Qtmp)

	def solve_hmm_problem(self, start_point, start_speed=None):
		"""
		Compute a trajectory from the LQR solution
		:param start_point: 	[np.array(nb_dim)]
		:param start_speed: 	[np.array(nb_dim)] assumed as 0 if not given
		:return:
		"""
		x_s = []

		if start_speed is None:
			start_speed = np.zeros(start_point.shape)

		x = np.concatenate([start_point, start_speed])

		for t in range(self.horizon):
			self.rData[:, t] = x

			u = self.L[t].dot(self.target[:, t] - x)
			if t == 0:
				u_0 = np.copy(u)

			if not self.discrete:
				x += (self.A.dot(x) + self.B.dot(u)) * self.dt
			else:
				x = (self.A.dot(x) + self.B.dot(u))

			x_s += [x]

		return np.copy(self.rData), u_0, np.asarray(x_s)

