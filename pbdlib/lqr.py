import numpy as np
import sys, os
import numpy.linalg as ln
import scipy

cpp_path = os.path.abspath(__file__)[:-14]
sys.path.append(cpp_path + "/pbdlib_cpp_wrapper/build/bin")

import spbdlibpy as pbdc

M = lambda x: np.asfortranarray(x)

class LQR(pbdc.LQR):
	def __init__(self, A, B, dt=0.01, rFactor=-6,
				 horizon=50, discrete=False, R=None):
		"""

		:param A:			[np.array()]
		:param B:			[np.array()]
			Dynamical system parameters, if you want to provide personalized ones.
		:param dt:
		:param rFactor:		[float]
			Control cost
		:param horizon:		[int]
			Number of timestep for horizon
		:param discrete:	[bool]
			Use LQR in discrete timestep
		:param R:
		"""
		# let the user choose his A model  and B control matrix
		self.discrete = discrete
		self.dt = dt


		self.A = A
		self.B = B
		self.nb_var = self.A.shape[0]
		self.r = R if R is not None else np.eye(B.shape[1]) * rFactor

		pbdc.LQR.__init__(self, M(self.A), M(self.B), dt)

		self.horizon = horizon
		self.Q = pbdc.Vmat(horizon)

		self.target = np.zeros((A.shape[0], horizon))

		# self.force_target = np.zeros((B.shape[1], horizon))

		self.nb_dim = A.shape[0]
		self.nb_ctrl = B.shape[1]

	def set_r_factor(self, r_factor):
		self.r_factor = r_factor
		self.r = np.eye(self.nb_dim) * 10 ** r_factor

	def reset_model(self, A, B):
		self.A = A
		self.B = B
		pbdc.LQR.__init__(self, M(A), M(B), self.dt)

	# @profile
	def evaluate_gains_finiteHorizon(self, use_python=False):
		self.use_python = use_python

		if use_python:
			invR = ln.inv(self.r)
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
		else:
			if self.discrete:
				super(LQR, self).evaluate_gains_finiteHorizon_discrete()
			else:
				raise not NotImplementedError
				# super(LQR, self).evaluate_gains_finiteHorizon(M(final_cost), M(final_target))
				pass

	def evaluate_gains_infiniteHorizon(self, one_step=False, use_scipy=False, use_python=False):
		self.use_python = use_python + use_scipy

		if one_step and not self.use_python:
			super(LQR, self).evaluate_gains_infiniteHorizon_step()
		elif use_python:
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

		elif use_scipy:
			if self.discrete:
				S = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q[0], self.r)
				self.L = ln.inv(self.r).dot(self.B.T).dot(S)

			else:
				S = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q[0], self.r)
				self.L = ln.inv(self.r).dot(self.B.T).dot(S)
		else:
			super(LQR, self).evaluate_gains_infiniteHorizon()

	def solve_algebraic_riccati(self, A, B, Q, R):
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


	def setProblem(self, r, q, target):
		"""
		Set the LQR problem

		:param r: 		np.array((DxD))
			Control cost
		:param q:  		[np.array((DxD)) x nb_timestep]
			List of state cost
		:param target:	np.array((Dxnb_timestep))
			Target vector for all time step
		:return: bool
		"""

		# create vector of control cost
		q_vec = pbdc.Vmat(len(q))
		for i in range(len(q)):
			q_vec[i] = M(q[i])

		self.target = target
		return pbdc.LQR.setProblem(self, M(r), q_vec, M(target))

	def predict(self, xi_0):
		xi_s = [xi_0]

		for t in range(self.horizon):
			u = self.getGains(t).dot(self.target[:, t] - xi_s[-1])

			xi_s += [np.copy(self.A.dot(xi_s[-1]) + self.B.dot(u))]

		return np.array(xi_s)


	def get_target_gain(self, t=0, get_force=False):
		"""
		Get the attractor target and gains at time step t
		:param t:
		:return:
		"""
		if self.use_python:
			if get_force:
				return np.copy(self.target[:, t]), self.L, self.force_target[:, t]
			else:
				return np.copy(self.target[:, t]), self.L
		else:
			if get_force:
				return np.copy(self.target[:,t]), self.getGains(t), self.force_target[:,t]
			else:
				return np.copy(self.target[:,t]), self.getGains(t)
