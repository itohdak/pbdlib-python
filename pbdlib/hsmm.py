import numpy as np

from hmm import *
from .functions import *
from .model import *

class OnlineForwardVariable():
	def __init__(self):
		self.nbD = None
		self.bmx = None
		self.ALPHA = None
		self.S = None
		self.h = None
		pass


class HSMM(HMM):
	def __init__(self, nb_states):
		HMM.__init__(self, nb_states)

		# transition matrix for forward variable computation : you can feed with the one you want
		self.Trans_Fw = np.zeros((nb_states, nb_states))

	def compute_duration(self, s=None, dur_reg=2.0, data_dim=None, sequ=None, last=True):
		"""

		:param s:
		:return:
		"""
		# reformat transition matrix: By removing self transition
		# self.Trans_Pd = self.Trans - np.diag(np.diag(self.Trans)) + realmin
		# self.Trans_Pd /= colvec(np.sum(self.Trans_Pd, axis=1))

		if data_dim is None and s is not None:
			data_dim = range(s[0]['Data'].shape[0])

		# init duration components
		self.Mu_Pd = np.zeros(self.nb_states)
		self.Sigma_Pd = np.zeros(self.nb_states)

		# reconstruct sequence of states from all demonstrations
		state_seq = []

		trans_list = np.zeros((self.nb_states, self.nb_states))# create a table to count the transition

		# reformat transition matrix by counting the transition
		if s is None:
			s = sequ


		for j, d in enumerate(s):
			if sequ is None:
				state_seq_tmp = self.viterbi(d['Data'][data_dim])
			else:
				state_seq_tmp = d.tolist()
			prev_state = 0
			for i, state in enumerate(state_seq_tmp):

				if i == 0:	# first state of sequence :
					pass
				elif i == len(state_seq_tmp)-1 and last:	# last state of sequence
					trans_list[state][state] += 1.0
				elif state != prev_state:	# transition detected
					trans_list[prev_state][state] += 1.0

				prev_state = state
			# make a list of state to compute the durations
			state_seq += state_seq_tmp

		self.Trans_Pd = trans_list
		# make sum to one
		for i in range(self.nb_states):
			sum = np.sum(self.Trans_Pd[i,:])
			if sum > realmin:
				self.Trans_Pd[i,:] /= sum

		#print state_seq

		# list of duration
		stateDuration = [[] for i in range(self.nb_states)]

		currState = state_seq[0]
		cnt = 1

		for i,state in enumerate(state_seq):
			if i == len(state_seq)-1: # last state of sequence
				stateDuration[currState] += [cnt]
			elif state == currState:
				cnt += 1
			else:
				stateDuration[currState] += [cnt]
				cnt = 1
				currState = state

		#print stateDuration
		for i in range(self.nb_states):
			self.Mu_Pd[i] = np.mean(stateDuration[i])
			if len(stateDuration[i])>1:
				self.Sigma_Pd[i] = np.std(stateDuration[i]) + dur_reg
			else:
				self.Sigma_Pd[i] = dur_reg


	def compute_tp_trans_matrix(self, s , i_in, min_sigma = 1e-9, dur_reg=2.0, data_dim=None):
		"""

		:param s: 		List of of dict ['Data'] = np.array((nb_dim, nb_timestep))

		:param i_in:	List of int
			Dimension of the parameters of the task-parametrized transition matrix
		:return:
		"""

		if data_dim is None:
			data_dim = range(s[0]['Data'].shape[0])

		nb_dim = len(i_in)
		self.tp_trans = pbd.TP_TRANSITION(nb_states=self.nb_states,nb_dim=nb_dim)

		# reformat transition matrix
		self.Trans_Pd = self.Trans - np.diag(np.diag(self.Trans)) + realmin
		self.Trans_Pd /= colvec(np.sum(self.Trans_Pd, axis=1))
		# np.set_printoptions(precision=2,infstr='inf',linewidth=120)
		# print self.Trans_Pd

		# init duration components
		self.Mu_Pd = np.zeros(self.nb_states)
		self.Sigma_Pd = np.zeros(self.nb_states)

		# reconstruct sequence of states from all demonstrations
		state_seq = []

		trans_param_list = [[[] for i in range(self.nb_states)]
					for j in range(self.nb_states)]

		for d in s:
			state_seq_tmp = self.viterbi(d['Data'][data_dim])
			prev_state = 0
			for i, state in enumerate(state_seq_tmp):

				if i == 0:	# first state of sequence :
					pass
				elif i == len(state_seq_tmp)-1:	# last state of sequence
					trans_param_list[state][state] += [d['Data'][data_dim][i_in,i]]
				elif state != prev_state:	# transition detected
					trans_param_list[prev_state][state] += [d['Data'][data_dim][i_in,i]]

				prev_state = state

			state_seq += state_seq_tmp

		# compute model of transition parameters : here a gaussian distribution
		for i in range(self.nb_states):
			for j in range(self.nb_states):
				# if no transition detected
				nb_trans = len(trans_param_list[i][j])
				if nb_trans == 0:
					self.tp_trans.Prior_Trans[i, j] = 0.0
					self.tp_trans.Mu_Trans[:, i, j] = np.zeros(nb_dim)
					self.tp_trans.Sigma_Trans[:, :, i, j] = np.eye( nb_dim)

				else:
					self.tp_trans.Prior_Trans[i, j] = nb_trans
					self.tp_trans.Mu_Trans[:, i, j] = np.mean(trans_param_list[i][j],axis=0)
					if nb_trans == 1:
						self.tp_trans.Sigma_Trans[:, :, i, j] = min_sigma * \
																np.eye(nb_dim)
					else:
						self.tp_trans.Sigma_Trans[:, :, i, j] =\
							np.cov(np.array(trans_param_list[i][j]).T) + min_sigma * \
																np.eye(nb_dim)

			if np.sum(self.tp_trans.Prior_Trans[i, :]) > realmin:
				self.tp_trans.Prior_Trans[i, :] /= np.sum(self.tp_trans.Prior_Trans[i, :])

		self.Trans_Pd = self.tp_trans.Prior_Trans
		# list of duration
		stateDuration = [[] for i in range(self.nb_states)]

		currState = state_seq[0]
		cnt = 1

		for i, state in enumerate(state_seq):
			if i == len(state_seq) - 1:  # last state of sequence
				stateDuration[currState] += [cnt]
			elif state == currState:
				cnt += 1
			else:
				stateDuration[currState] += [cnt]
				cnt = 1
				currState = state
		#
		# #print stateDuration
		# for i in range(self.nb_states):
		# 	self.Mu_Pd[i] = np.mean(stateDuration[i])
		# 	self.Sigma_Pd[i] = np.std(stateDuration[i])
		# print stateDuration
		for i in range(self.nb_states):
			self.Mu_Pd[i] = np.mean(stateDuration[i])
			if len(stateDuration[i]) > 1:
				self.Sigma_Pd[i] = np.std(stateDuration[i]) + dur_reg
			else:
				self.Sigma_Pd[i] = dur_reg

	def _update_transition_matrix(self, tr_param):
		for i in range(self.nb_states):
			for j in range(self.nb_states):
				if self.tp_trans.Prior_Trans[i,j] > realmin:
					self.Trans_Fw[i, j] = self.tp_trans.Prior_Trans[i,j] *\
							  multi_variate_normal(tr_param.reshape(-1,1),
							   self.tp_trans.Mu_Trans[:,i,j], self.tp_trans.Sigma_Trans[:,:,i,j])
				else :
					self.Trans_Fw[i, j] = 0

			# rescale to sum to one
			if np.sum(self.Trans_Fw[i, :]) > realmin:
				self.Trans_Fw[i, :] /= np.sum(self.Trans_Fw[i, :])

	def forward_variable_ts(self, n_step, trans_type='default', statesPriors=None):
		"""
		To compute a forward variable for HSMM based online on time.

		:param n_step: 			int
			Number of step for forward variable computation
		:param trans_type:		'default' or 'tp_trans'
		:param start_priors: 	np.array((N,))
			Priors for localizing at first step
		:return:
		"""
		if trans_type == 'default':
			self.Trans_Fw = self.Trans_Pd
		elif trans_type == 'tp_trans':
			# self.Trans_Fw = self.Trans_Pd
			pass

		nbD = np.round(2 * n_step/self.nb_states)

		self.Pd = np.zeros((self.nb_states, nbD))

		# Precomputation of duration probabilities
		for i in range(self.nb_states):
			self.Pd[i, :] = multi_variate_normal(np.arange(nbD), self.Mu_Pd[i], self.Sigma_Pd[i])
			self.Pd[i, :] = self.Pd[i, :] / np.sum(self.Pd[i, :])

		h = np.zeros((self.nb_states, n_step))

		ALPHA, S, h[:, 0] = self._fwd_init_ts(nbD, statesPriors=statesPriors)

		for i in range(1, n_step):
			ALPHA, S, h[:, i] = self._fwd_step_ts(ALPHA, S, nbD)

		h /= np.sum(h, axis=0)

		return h

	def forward_variable_priors(self, n_step, priors, tp_param=None, start_priors=None):
		"""
		Compute the forward variable with some priors over the states

		:param n_step: 			[int]
			Number of step for forward variable computation
		:param priors: 			[np.array((N,))]
			Priors over the states
		:param start_priors: 	[np.array((N,))]
			Priors for localizing at first step

		:return:
		"""
		if tp_param is None:
			try:
				self.Trans_Fw = self.tp_trans.Prior_Trans
			except:
				# print "No task-parametrized transition matrix : normal transition matrix will be used"
				self.Trans_Fw = self.Trans_Pd
		else: # compute the transition matrix for current parameters
			self._update_transition_matrix(tp_param)


		# nbD = np.round(2 * n_step/self.nb_states)
		nbD = np.round(2 * n_step)

		self.Pd = np.zeros((self.nb_states, nbD))

		# Precomputation of duration probabilities
		for i in range(self.nb_states):
			self.Pd[i, :] = multi_variate_normal(np.arange(nbD), self.Mu_Pd[i], self.Sigma_Pd[i])
			if np.sum(self.Pd[i,:])< 1e-50:
				self.Pd[i,:] = 1.0 / self.Pd[i, :].shape[0]
			else:
				self.Pd[i, :] = self.Pd[i, :] / np.sum(self.Pd[i, :])

		h = np.zeros((self.nb_states, n_step))

		priors = colvec(priors)
		priors /= np.sum(priors)

		bmx, ALPHA, S, h[:, [0]] = self._fwd_init_priors(nbD, priors, start_priors=start_priors)

		for i in range(1, n_step):
			bmx, ALPHA, S, h[:, [i]] = self._fwd_step_priors(bmx, ALPHA, S, nbD, priors)

		h /= np.sum(h, axis=0)

		return h

	def online_forward_variable_prob(self, n_step, priors, tp_param=None, start_priors=None, nb_sum=None):
		"""
		To compute an online forward variable for HSMM. You can use it for example for localization
		of state while reproducing.

		:param n_step:			[int]
		:param priors: 			[np.array((nb_states,))]
		:param tp_param: 		[np.array((nb_input_dim,))]
		:param start_priors:	[np.array((nb_states,))]
		:return:
		"""
		if tp_param is None:
			try:
				# self.Trans_Fw = self.tp_trans.Prior_Trans
				self.Trans_Fw = self.Trans_Pd
				# print self.Trans_Fw
			except:
				print "No task-parametrized transition matrix : normal transition matrix will be used"
				self.Trans_Fw = self.Trans_Pd
				# print self.Trans_Fw
		else:  # compute the transition matrix for current parameters
			self._update_transition_matrix(tp_param)

		self.ol = OnlineForwardVariable()

		# self.ol.nbD = np.round(2 * n_step / self.nb_states)
		if nb_sum is None:
			self.ol.nbD = np.round(2 * n_step)
		else:
			self.ol.nbD = nb_sum

		self.Pd = np.zeros((self.nb_states, self.ol.nbD))

		# Precomputation of duration probabilities
		for i in range(self.nb_states):
			self.Pd[i, :] = multi_variate_normal(np.arange(self.ol.nbD), self.Mu_Pd[i],
												 self.Sigma_Pd[i])
			if np.sum(self.Pd[i, :]) < 1e-50:
				self.Pd[i, :] = 1.0 / self.Pd[i, :].shape[0]
			else:
				self.Pd[i, :] = self.Pd[i, :] / np.sum(self.Pd[i, :])

		h = np.zeros((self.nb_states, n_step))

		self.ol.h = np.zeros((self.nb_states, 1))

		priors = colvec(priors)
		priors /= np.sum(priors)

		self.ol.bmx, self.ol.ALPHA, self.ol.S, self.ol.h = self._fwd_init_priors(self.ol.nbD, priors,
														 start_priors=start_priors)

		# for i in range(1, n_step):
		# 	bmx, ALPHA, S, h[:, [i]] = self._fwd_step_priors(bmx, ALPHA, S, self.ol.nbD, priors)
		#
		# h /= np.sum(h, axis=0)

		return self.ol.h

	def online_forward_variable_prob_step(self, priors):
		"""
		Single step to compute an online forward variable for HSMM.

		:param priors: 			[np.array((nb_states,))]
		:return:
		"""

		priors = colvec(priors)
		try:
			self.ol.bmx, self.ol.ALPHA, self.ol.S, self.ol.h = \
				self._fwd_step_priors(self.ol.bmx, self.ol.ALPHA, self.ol.S, self.ol.nbD, priors,
									  trans_reg=0.00, trans_diag=0.00)
			return self.ol.h
		except:
			# traceback.print_exc(file=sys.stdout)
			return None


	def online_forward_variable_prob_predict(self, n_step, priors):
		"""
		Compute prediction for n_step timestep on the current online forward variable.

		:param priors: 			[np.array((nb_states,))]
		:return:
		"""

		h = np.zeros((self.nb_states, n_step))

		priors = colvec(priors)
		priors /= np.sum(priors)

		# bmx, ALPHA, S, h[:, [0]] = self._fwd_init_priors(nbD, priors, start_priors=start_priors)
		h[:,[0]] = self.ol.h
		bmx = self.ol.bmx
		ALPHA = self.ol.ALPHA
		S = self.ol.S

		try:
			for i in range(1, n_step):
				bmx, ALPHA, S, h[:, [i]] = self._fwd_step_priors(bmx, ALPHA, S, self.ol.nbD, priors)

		except:
			h = np.tile(self.ol.h, (1, n_step))

			# traceback.print_exc(file=sys.stdout)

		h /= np.sum(h, axis=0)

		return h

	def forward_variable_hsum(self, n_step, Data, tp_param=None):
		"""
		Compute the forward variable with observation

		:param n_step: 		int
			Number of step for forward variable computation
		:param priors: 		np.array((N,))
			Priors over the states

		:return:
		"""
		if tp_param is None:
			# self.Trans_Fw = self.tp_trans.Prior_Trans
			self.Trans_Fw = self.Trans_Pd
		else: # compute the transition matrix for current parameters
			self._update_transition_matrix(tp_param)

		# nbD = np.round(2 * n_step/self.nb_states)
		nbD = np.round(4 * n_step)

		self.Pd = np.zeros((self.nb_states, nbD))

		# Precomputation of duration probabilities
		for i in range(self.nb_states):
			self.Pd[i, :] = multi_variate_normal(np.arange(nbD), self.Mu_Pd[i], self.Sigma_Pd[i])
			self.Pd[i, :] = self.Pd[i, :] / np.sum(self.Pd[i, :])

		if np.isnan(self.Pd).any():
			print "Problem of duration probabilities"
			return

		h = np.zeros((self.nb_states, n_step))

		bmx, ALPHA, S, h[:, [0]] = self._fwd_init_hsum(nbD, Data[:,1])
		for i in range(1, n_step):
			bmx, ALPHA, S, h[:, [i]] = self._fwd_step_hsum(bmx, ALPHA, S, nbD, Data[:,i])

		h /= np.sum(h, axis=0)

		return h

	def _fwd_init_ts(self, nbD, statesPriors=None):
		"""
		Initiatize forward variable computation based only on duration (no observation)
		:param nbD: number of time steps
		:return:
		"""
		if statesPriors is None:
			ALPHA = np.tile(colvec(self.StatesPriors), [1, nbD]) * self.Pd
		else:
			ALPHA = np.tile(colvec(statesPriors), [1, nbD]) * self.Pd

		S = np.dot(self.Trans_Fw.T, ALPHA[:, [0]]) # use [idx] to keep the dimension

		return ALPHA, S, np.sum(ALPHA, axis=1)

	def _fwd_step_ts(self, ALPHA, S, nbD):
		"""
		Step of forward variable computation based only on duration (no observation)
		:return:
		"""
		ALPHA = np.concatenate((S[:, [-1]] * self.Pd[:, 0:nbD-1] + ALPHA[:, 1:nbD],
								S[:, [-1]] * self.Pd[:, [nbD-1]]), axis=1)

		S = np.concatenate((S, np.dot(self.Trans_Fw.T, ALPHA[:, [0]])), axis=1)

		return ALPHA, S, np.sum(ALPHA, axis=1)

	def _fwd_init_priors(self, nbD, priors,start_priors=None):
		"""

		:param nbD:
		:return:
		"""
		bmx = np.zeros((self.nb_states, 1))

		Btmp = priors

		if start_priors is None:
			ALPHA = np.tile(colvec(self.StatesPriors), [1, nbD]) * self.Pd
		else:
			ALPHA = np.tile(colvec(start_priors), [1, nbD]) * self.Pd
		# r = Btmp.T * np.sum(ALPHA, axis=1)
		r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))

		bmx[:, [0]] = Btmp / r
		E = bmx * ALPHA[:, [0]]
		S = np.dot(self.Trans_Fw.T, E) # use [idx] to keep the dimension

		return bmx, ALPHA, S, Btmp * colvec(np.sum(ALPHA, axis=1))

	def _fwd_step_priors(self, bmx, ALPHA, S, nbD, priors, trans_reg=0.0, trans_diag=0.0):
		"""

		:param bmx:
		:param ALPHA:
		:param S:
		:param nbD:
		:return:
		"""

		Btmp = priors

		ALPHA = np.concatenate((S[:, [-1]] * self.Pd[:, 0:nbD-1] + bmx[:,[-1]] * ALPHA[:, 1:nbD],
								S[:, [-1]] * self.Pd[:, [nbD-1]]), axis=1)

		r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))
		bmx = np.concatenate((bmx, Btmp / r), axis=1)
		E = bmx[:, [-1]] * ALPHA[:, [0]]

		S = np.concatenate((S, np.dot(self.Trans_Fw.T + np.eye(self.nb_states) * trans_diag + trans_reg, ALPHA[:, [0]])), axis=1)
		alpha = Btmp * colvec(np.sum(ALPHA, axis=1))
		alpha /= np.sum(alpha)
		return bmx, ALPHA, S, alpha

	def _fwd_init_hsum(self, nbD, Data):
		"""

		:param nbD:
		:return:
		"""
		bmx = np.zeros((self.nb_states, 1))

		Btmp = np.zeros((self.nb_states, 1))

		for i in range(self.nb_states):
			Btmp[i] = multi_variate_normal(Data.reshape(-1,1), self.Mu[:,i], self.Sigma[:,:,i]) + 1e-12

		Btmp /= np.sum(Btmp)

		ALPHA = np.tile(colvec(self.StatesPriors), [1, nbD]) * self.Pd
		# r = Btmp.T * np.sum(ALPHA, axis=1)
		r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))

		bmx[:, [0]] = Btmp / r
		E = bmx * ALPHA[:, [0]]
		S = np.dot(self.Trans_Fw.T, E) # use [idx] to keep the dimension

		return bmx, ALPHA, S, Btmp * colvec(np.sum(ALPHA, axis=1))

	def _fwd_step_hsum(self, bmx, ALPHA, S, nbD, Data):
		"""

		:param bmx:
		:param ALPHA:
		:param S:
		:param nbD:
		:return:
		"""

		Btmp = np.zeros((self.nb_states, 1))

		for i in range(self.nb_states):
			Btmp[i] = multi_variate_normal(Data.reshape(-1,1), self.Mu[:,i], self.Sigma[:,:,i]) + 1e-12

		Btmp /= np.sum(Btmp)

		ALPHA = np.concatenate((S[:, [-1]] * self.Pd[:, 0:nbD-1] + bmx[:,[-1]] * ALPHA[:, 1:nbD],
								S[:, [-1]] * self.Pd[:, [nbD-1]]), axis=1)

		r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))
		bmx = np.concatenate((bmx, Btmp / r), axis=1)
		E = bmx[:, [-1]] * ALPHA[:, [0]]

		S = np.concatenate((S, np.dot(self.Trans_Fw.T, ALPHA[:, [0]])), axis=1)
		alpha = Btmp * colvec(np.sum(ALPHA, axis=1))
		alpha /= np.sum(alpha)
		return bmx, ALPHA, S, alpha