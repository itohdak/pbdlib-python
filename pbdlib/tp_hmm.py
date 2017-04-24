import numpy as np
from .hmm import *
from .hsmm import *
from .gmm import *
from .functions import *
from .model import *
from .gmr import *

import math
from numpy.linalg import inv, pinv, norm, det
import sys


class TP_HMM(HSMM):
	def __init__(self, nb_states, nb_frames, nb_dim, nb_features=1, features_dyn=[True]):
		"""

		:param nb_states: 		[int]
			number of HMM or HSMM states
		:param nb_frames: 		[int]
		:param nb_dim: 			[int] in [1,2,3]
			number of dimensions of the space
		:param nb_features:
			number of features, such as position, velocity, forces, acceleration
		:param features_dyn:
			if the feature is absolute as position [False] or relative as velocity [True]
		"""
		HSMM.__init__(self, nb_states)

		self.gmm_prod = None

		self.nb_features = nb_features
		self.features_dyn = features_dyn
		self.nb_axis = nb_dim
		self.nb_frames = nb_frames

		self.gmms = []

	def prodgmm(self, TPs, data_in=None, i_in=None, error=False, sensori_reg=None, reactive=False):
		"""
		Get the product of GMM given the reference frame TPs, optionally, evidences could be
		given if the GMM also encodes sensory information

		:param TPs: 		list of dict() {'A': np.array((N,N)),'b': np.array((N,))}
		:param data_in : 	np.array((D,))
			evidence
		:param error :		[bool]
			use a Gaussian error on 'b' of task parameters, should be available as TPs[m]['Sigma_b']
		:param i_in:		list of D indexes. ex: [0,1]
			dimensions of evidencesy
		:return:			[Model]
		"""

		if data_in is not None and not reactive:
			# search for indexes of trajectory
			i_out = range(self.Mu.shape[0])
			for dim in i_in:
				i_out.remove(dim)

			if sensori_reg is not None:
				self.regress(data_in, i_in, i_out, reg=sensori_reg)
			else:
				self.regress(data_in, i_in, i_out)

			for i in range(self.nb_frames):
				self.gmms.append(GMM(self.nb_states, self.nb_axis))

				self.gmms[i].Mu = self.MuR[i*self.nb_axis:(i + 1) * self.nb_axis, :]
				# self.gmms[i].Priors = self.PriorsR
				self.gmms[i].Priors = self.PriorsR/self.Priors
				self.gmms[i].Sigma = self.SigmaR[i*self.nb_axis:(i + 1) * self.nb_axis,
												i*self.nb_axis:(i + 1) * self.nb_axis, :]
				self.gmms[i].update_precision_matrix()

		elif data_in is not None and reactive:
			self.gmms = []
			i_out = range(self.Mu.shape[0])
			for dim in i_in:
				i_out.remove(dim)

			if sensori_reg is not None:
				MuTmp, SigmaTmp = self.regress_reactive(data_in, i_in, i_out, reg=sensori_reg)
			else:
				MuTmp, SigmaTmp = self.regress_reactive(data_in, i_in, i_out)

			MuTmp = MuTmp[:, np.newaxis]
			SigmaTmp = SigmaTmp[:, :, np.newaxis]

			for i in range(self.nb_frames):
				self.gmms.append(GMM(1, self.nb_axis))

				self.gmms[i].Mu = MuTmp[i * self.nb_axis:(i + 1) * self.nb_axis, :]
				# self.gmms[i].Priors = self.PriorsR
				self.gmms[i].Priors = self.Priors
				self.gmms[i].Sigma = SigmaTmp[
									 i * self.nb_axis:(i + 1) * self.nb_axis,
									 i * self.nb_axis:(i + 1) * self.nb_axis, :]
				self.gmms[i].update_precision_matrix()

		else:
			# if self.gmms == []:
			self.gmms = []
			for i in range(self.nb_frames):
				self.gmms.append(GMM(self.nb_states, self.nb_axis * self.nb_features,
									 nb_features=self.nb_features))

				dim = range(i * self.nb_axis, (i + 1) * self.nb_axis)

				fs = i * self.nb_axis * self.nb_features  # start of frame i
				fe = (i + 1) * self.nb_axis * self.nb_features  # end of frame i
				self.gmms[i].Mu = self.Mu[fs:fe, :]
				self.gmms[i].Priors = self.Priors
				self.gmms[i].Sigma = self.Sigma[fs:fe, fs:fe, :]
				self.gmms[i].features_dyn = self.features_dyn
				self.gmms[i].update_precision_matrix()

		if not error:
			self.gmm_prod = self.gmms[0].lintrans(TPs[0]['A'], TPs[0]['b'])

			for m in range(1, self.nb_frames):
				self.gmm_prod *= self.gmms[m].lintrans(TPs[m]['A'], TPs[m]['b'])
		else:
			self.gmm_prod = self.gmms[0].lintrans(TPs[0]['A'], TPs[0]['b'], Sigma_b=TPs[0]['Sigma_b'])

			for m in range(1, self.nb_frames):
				self.gmm_prod *= self.gmms[m].lintrans(TPs[m]['A'], TPs[m]['b'], Sigma_b=TPs[m]['Sigma_b'])

		self.gmm_prod.Priors = self.gmms[0].Priors

		try:
			self.gmm_prod.Trans = self.Trans
			self.gmm_prod.Trans_Pd = self.Trans_Pd
			self.gmm_prod.Mu_Pd = self.Mu_Pd
		except:
			pass

		return self.gmm_prod

	def prodgmm_priors(self, TPs, data_in=None, i_in=None):
		"""
		Get the product of GMM given the reference frame TPs, optionally, evidences could be
		given if the GMM also encodes sensory information

		:param TPs: 		list of dict() {'A': np.array((N,N)),'b': np.array((N,))}
		:param data_in : 	np.array((D,))
			evidence
		:param i_in:		list of D indexes. ex: [0,1]
			dimensions of evidences
		:return:
		"""

		if data_in is not None:
			# search for indexes of trajectory
			i_out = range(self.Mu.shape[0])
			for dim in i_in:
				i_out.remove(dim)

			self.regress(data_in, i_in, i_out)

			for i in range(self.nb_frames):
				self.gmms.append(GMM(self.nb_states, self.nb_axis))

				self.gmms[i].Mu = self.MuR[i*self.nb_axis:(i + 1) * self.nb_axis, :]
				# self.gmms[i].Priors = self.PriorsR
				self.gmms[i].Priors = self.PriorsR/self.Priors
				self.gmms[i].Sigma = self.SigmaR[i*self.nb_axis:(i + 1) * self.nb_axis,
												i*self.nb_axis:(i + 1) * self.nb_axis, :]
				self.gmms[i].update_precision_matrix()

		else:
			# if self.gmms == []:
			for i in range(self.nb_frames):
				self.gmms.append(GMM(self.nb_states, self.nb_axis))

				self.gmms[i].Mu = self.Mu[i*self.nb_axis:(i + 1) * self.nb_axis, :]
				self.gmms[i].Priors = self.Priors
				self.gmms[i].Sigma = self.Sigma[i*self.nb_axis:(i + 1) * self.nb_axis,
												i*self.nb_axis:(i + 1) * self.nb_axis, :]
				self.gmms[i].update_precision_matrix()


		weights = np.zeros((self.nb_frames, self.nb_states))

		for i in range(self.nb_frames):
			for j in range(self.nb_states):
				weights[i, j] = np.exp(-np.sum(self.gmms[i].Mu[:,j]**2)/8000)

		weights /= np.sum(weights, axis=0)

		for i in range(self.nb_frames):
			for j in range(self.nb_states):
				self.gmms[i].Lambda[:,:,j] *= weights[i,j]

		self.gmm_prod = self.gmms[0].lintrans(TPs[0]['A'], TPs[0]['b'])


		for m in range(1, self.nb_frames):
			self.gmm_prod *= self.gmms[m].lintrans(TPs[m]['A'], TPs[m]['b'])
			# self.gmm_prod = self.gmm_prod.weighted_product(self.gmms[m].lintrans(TPs[m]['A'], TPs[m]['b']),weights=weights[m,:])

		self.gmm_prod.Priors = self.gmms[0].Priors

		try:
			self.gmm_prod.Trans = self.Trans
			self.gmm_prod.Trans_Pd = self.Trans_Pd
			self.gmm_prod.Mu_Pd = self.Mu_Pd
		except:
			pass

		return self.gmm_prod