import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat # loading data from matlab
import os
import pbdlib as pbd
import pbdlib.plot
from pbdlib.utils.jupyter_utils import *
np.set_printoptions(precision=2)

nb_states = 5  # choose the number of states in HMM or clusters in GMM

letter_in = 'X' # INPUT LETTER: choose a letter in the alphabet
letter_out = 'C' # OUTPUT LETTER: choose a letter in the alphabet

datapath = os.path.dirname(pbd.__file__) + '/data/2Dletters/'

# data_in = loadmat(datapath + '%s.mat' % letter_in)
# data_out = loadmat(datapath + '%s.mat' % letter_out)

# demos_in = [d['pos'][0][0].T for d in data_in['demos'][0]] # cleaning matlab data
# demos_out = [d['pos'][0][0].T for d in data_out['demos'][0]] # cleaning matlab data

# demos = [np.concatenate([d_in, d_out], axis=1)
#          for d_in, d_out in zip(demos_in, demos_out)]

# fig, ax = plt.subplots(ncols=2)
# fig.set_size_inches(5., 2.5)

# [ax[i].set_title(s) for i, s in enumerate(['input', 'output'])]

# for p_in, p_out in zip(demos_in, demos_out):
#     ax[0].plot(p_in[:, 0], p_in[:, 1])
#     ax[1].plot(p_out[:, 0], p_out[:, 1])

# # creating models
# gmm = pbd.GMM(nb_states=nb_states)
# hmm = pbd.HMM(nb_states=nb_states)
# hsmm = pbd.HSMM(nb_states=nb_states)

# # initializing model by splitting the demonstrations in k bins
# [model.init_hmm_kbins(demos) for model in [gmm, hmm, hsmm]]

# # EM to train model
# gmm.em(np.concatenate(demos), reg=1e-8)
# hmm.em(demos, reg=1e-8)
# hsmm.em(demos, reg=1e-8)

# # plotting demos
# fig, ax = plt.subplots(ncols=3)
# fig.set_size_inches(7.5, 2.8)

# for p_in, p_out in zip(demos_in, demos_out):
#     ax[0].plot(p_in[:, 0], p_in[:, 1])
#     ax[1].plot(p_out[:, 0], p_out[:, 1])

# [ax[i].set_title(s, fontsize=16)
#      for i, s in enumerate([r'$\{\mu_{IN}, \Sigma_{IN}\}$',
#                             '$\{\mu_{OUT}, \Sigma_{OUT}\}$', r'$A$'])]

# plt.tight_layout()
# # use dim for selecting dimensions of GMM to plot
# pbd.plot_gmm(gmm.mu, gmm.sigma, ax=ax[0], dim=[0, 1]);
# pbd.plot_gmm(gmm.mu, gmm.sigma, ax=ax[1], dim=[2, 3]);

# # plotting transition matrix
# ax[2].imshow(np.log(hmm.Trans+1e-10), interpolation='nearest',
#              vmin=-5, cmap='viridis');
# plt.tight_layout()

# n = 0

# resp_gmm = gmm.compute_resp(demos_in[n], marginal=slice(0, 2))

# alpha_hmm, _, _, _, _ = hmm.compute_messages(
#     demos_in[n], marginal=slice(0, 2))

# alpha_hsmm, _, _, _, _ = hsmm.compute_messages(
#     demos_in[n], marginal=slice(0, 2))


# fig, ax = plt.subplots(nrows=3)
# fig.set_size_inches(7.5,3.6)

# ax[0].plot(resp_gmm.T , lw=1);
# ax[1].plot(alpha_hmm.T, lw=1);
# ax[2].plot(alpha_hsmm.T, lw=1);

# [ax[i].set_ylim([-0.2, 1.2]) for i in range(3)]
# plt.xlabel('timestep');

# mu_est_gmm, sigma_est_gmm = gmm.condition(
#     demos_in[1], dim_in=slice(0, 2), dim_out=slice(2, 4))
# mu_est_hmm, sigma_est_hmm = hmm.condition(
#     demos_in[1], dim_in=slice(0, 2), dim_out=slice(2, 4))
# mu_est_hsmm, sigma_est_hsmm = hsmm.condition(
#     demos_in[1], dim_in=slice(0, 2), dim_out=slice(2, 4))

# fig, ax = plt.subplots(ncols=3)
# fig.set_size_inches(7.5, 2.5)

# for p_in, p_out in zip(demos_in, demos_out):
#     ax[0].plot(p_out[:, 0], p_out[:, 1],'k', alpha=0.1)
#     ax[1].plot(p_out[:, 0], p_out[:, 1],'k', alpha=0.1)
#     ax[2].plot(p_out[:, 0], p_out[:, 1],'k', alpha=0.1)

# [ax[i].set_title(s, fontsize=16)
#      for i, s in enumerate(['GMM', 'HMM', 'HSMM'])]

# ax[0].plot(mu_est_gmm[:, 0], mu_est_gmm[:,1 ], 'r--', lw=3)
# ax[1].plot(mu_est_hmm[:, 0], mu_est_hmm[:,1 ], 'r--', lw=3)
# ax[2].plot(mu_est_hsmm[:, 0], mu_est_hsmm[:,1 ], 'r--', lw=3)

# pbd.plot_gmm(
#     mu_est_gmm[::5], sigma_est_gmm[::5], ax=ax[0], swap=True, alpha=0.05)
# pbd.plot_gmm(
#     mu_est_hmm[::5], sigma_est_hmm[::5], ax=ax[1], swap=True, alpha=0.05)
# pbd.plot_gmm(
#     mu_est_hsmm[::5], sigma_est_hsmm[::5], ax=ax[2], swap=True, alpha=0.05)

letter_out = 'C' # choose a letter in the alphabet

data_out = loadmat(datapath + '%s.mat' % letter_out)

demos_out = [d['pos'][0][0].T for d in data_out['demos'][0]]
demos_in = [np.random.multivariate_normal(np.zeros(2), np.eye(2), d.shape[0])
            for d in demos_out]

demos = [np.concatenate([d_in, d_out], axis=1)
         for d_in, d_out in zip(demos_in, demos_out)]

gmm = pbd.GMM(nb_states=nb_states)
hmm = pbd.HMM(nb_states=nb_states)
hsmm = pbd.HSMM(nb_states=nb_states)

# initializing model
[model.init_hmm_kbins(demos) for model in [gmm, hmm, hsmm]]

# EM to train model
gmm.em(np.concatenate(demos), reg=1e-3)
hmm.em(demos, reg=1e-3)
hsmm.em(demos, reg=1e-3)

# plotting demos
fig, ax = plt.subplots(ncols=2)
fig.set_size_inches(5., 2.5)

for p_in, p_out in zip(demos_in, demos_out):
    ax[0].plot(p_in[:, 0], p_in[:, 1])
    ax[1].plot(p_out[:, 0], p_out[:, 1])

# use dim for selecting dimensions of GMM to plot
pbd.plot_gmm(gmm.mu, gmm.sigma, ax=ax[0], dim=[0, 1]);
pbd.plot_gmm(gmm.mu, gmm.sigma, ax=ax[1], dim=[2, 3]);

plt.show()

n = 0

resp_gmm = gmm.compute_resp(demos_in[n][:10], marginal=slice(0, 2))
alpha_hsmm, _, _, _, _ = hsmm.compute_messages(
    demos_in[n][:10], marginal=slice(0, 2))
alpha_hmm, _, _, _, _ = hmm.compute_messages(
    demos_in[n][:10], marginal=slice(0, 2))


fig, ax = plt.subplots(nrows=3)
fig.set_size_inches(7.5,3.6)

ax[0].plot(resp_gmm.T , lw=1);
ax[1].plot(alpha_hmm.T, lw=1);
ax[2].plot(alpha_hsmm.T, lw=1);

[ax[i].set_ylim([-0.2, 1.2]) for i in range(3)]
plt.xlabel('timestep');

plt.show()

mu_est_gmm, sigma_est_gmm = gmm.condition(
    demos_in[1], dim_in=slice(0, 2), dim_out=slice(2, 4))
mu_est_hmm, sigma_est_hmm = hmm.condition(
    demos_in[1], dim_in=slice(0, 2), dim_out=slice(2, 4))
mu_est_hsmm, sigma_est_hsmm = hsmm.condition(
    demos_in[1], dim_in=slice(0, 2), dim_out=slice(2, 4))

fig, ax = plt.subplots(ncols=3)
fig.set_size_inches(7.5, 2.5)

for p_in, p_out in zip(demos_in, demos_out):
    ax[0].plot(p_out[:, 0], p_out[:, 1],'k', alpha=0.1)
    ax[1].plot(p_out[:, 0], p_out[:, 1],'k', alpha=0.1)
    ax[2].plot(p_out[:, 0], p_out[:, 1],'k', alpha=0.1)


ax[0].plot(mu_est_gmm[:, 0], mu_est_gmm[:,1 ], 'r--', lw=3)
ax[1].plot(mu_est_hmm[:, 0], mu_est_hmm[:,1 ], 'r--', lw=3)
ax[2].plot(mu_est_hsmm[:, 0], mu_est_hsmm[:,1 ], 'r--', lw=3)

pbd.plot_gmm(
    mu_est_gmm[::5], sigma_est_gmm[::5], ax=ax[0], swap=True, alpha=0.05)
pbd.plot_gmm(
    mu_est_hmm[::5], sigma_est_hmm[::5], ax=ax[1], swap=True, alpha=0.05)
pbd.plot_gmm(
    mu_est_hsmm[::5], sigma_est_hsmm[::5], ax=ax[2], swap=True, alpha=0.05)

plt.show()
