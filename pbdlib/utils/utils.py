from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')


def gu_pinv(A, rcond=1e-15):
    I = A.shape[0]
    J = A.shape[1]
    return np.array([[np.linalg.pinv(A[i, j]) for j in range(J)] for i in range(I)])

def align_trajectories(data):
	"""

	:param data: 		[list of np.array([nb_timestep, M, N, ...])]
	:return:
	"""
	from dtw import dtw

	ls = np.argmax([d.shape[0] for d in data])  # select longest as basis

	data_warp = []

	for d in data:
		dist, cost, acc, path = dtw(data[ls], d,
									dist=lambda x, y: np.linalg.norm(x - y, ord=1))

		data_warp += [d[path[1]][:data[ls].shape[0]]]

	return data_warp



def angle_to_rotation(theta):
	return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

def feature_to_slice(nb_dim=2, nb_frames=None, nb_attractor=2,
		features=None):
	# type: (int, list of int, int, list of list of string) -> object
	index = []
	l = 0
	for i, nb_frame, feature in zip(range(nb_attractor), nb_frames, features):
		index += [[]]
		for m in range(nb_frame):
			index[i] += [{}]
			for f in feature:
				index[i][m][f] = slice(l, l + nb_dim)
				l += nb_dim

	return index


def dtype_to_index(dtype):
	last_idx = 0
	idx = {}
	for name in dtype.names:
		idx[name] = range(last_idx,last_idx+dtype[name].shape[0])
		last_idx += dtype[name].shape[0]

	return idx



def gu_pinv(A, rcond=1e-15):
	I = A.shape[0]
	J = A.shape[1]
	return np.array([[np.linalg.pinv(A[i, j]) for j in range(J)] for i in range(I)])


#
# def gu_pinv(a, rcond=1e-15):
#     a = np.asarray(a)
#     swap = np.arange(a.ndim)
#     swap[[-2, -1]] = swap[[-1, -2]]
#     u, s, v = np.linalg.svd(a)
#     cutoff = np.maximum.reduce(s, axis=-1, keepdims=True) * rcond
#     mask = s > cutoff
#     s[mask] = 1. / s[mask]
#     s[~mask] = 0
#
#     return np.einsum('...uv,...vw->...uw',
#                      np.transpose(v, swap) * s[..., None, :],
#                      np.transpose(u, swap))

def plot_model_time(model, demos,figsize=(10, 2), dim_idx=[1], demo_idx=0):

	nb_dim = len(dim_idx)
	nb_samples = len(demos)

	fig = plt.figure(3, figsize=(figsize[0], figsize[1] * nb_dim))
	# fig.suptitle("Reproduction", fontsize=14, fontweight='bold')
	nb_plt = nb_dim
	ax = []  # subplots
	label_size = 15
	### specify subplots ###
	gs = gridspec.GridSpec(nb_dim, 1)

	for j in range(nb_plt):  # [0, 2, 1, 3, ...]
		ax.append(fig.add_subplot(gs[j]))
	for a in ax:
		a.set_axis_bgcolor('white')

	fig.suptitle("Demonstration", fontsize=14, fontweight='bold')

	idx = np.floor(np.linspace(1, 255, model.nb_states)).astype(int)
	color = cmap.viridis(range(256))[idx, 0:3]  # for states

	state_sequ = []
	for d in demos:
		state_sequ += [model.viterbi(d['Data'])]

	d = demos[demo_idx]
	s = state_sequ[demo_idx]

	for dim, a in zip(dim_idx, ax):
		a.plot(d['Data'][dim,:])

		for x_s, x_e, state in zip([0] + np.where(np.diff(s))[0].tolist(),  # start step
						np.where(np.diff(s))[0].tolist() + [len(s)],   # end step
						np.array(s)[[0] + (np.where(np.diff(s))[0] + 1).tolist()]):   # state idx
			a.axvline(x=x_e, ymin=0, ymax=1, c='k', lw=2, ls='--')

			mean = model.Mu[dim, state]
			var = np.sqrt(model.Sigma	[dim, dim, state])
			a.plot([x_s, x_e], [mean, mean], c='k', lw=2)

			a.fill_between([x_s, x_e], [mean+var, mean+var], [mean-var, mean-var],
							 alpha=0.5, color=color[state])

	plt.show()

def plot_demos_3d(demos, figsize=(15, 5), angle=[60, 45]):
	nb_samples = len(demos)
	fig = plt.figure(1, figsize=figsize)
	fig.suptitle("Demonstration", fontsize=14, fontweight='bold')
	nb_plt = 2
	ax = []
	label_size = 15

	idx = np.floor(np.linspace(1, 255, nb_samples)).astype(int)
	color_demo = cmap.viridis(range(256))[idx, 0:3]  # for states

	nb = 0
	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

	for j in [0, 1]:
		ax.append(fig.add_subplot(gs[j], projection='3d', axisbg='white'))

	ax[nb].set_title(r'$\mathrm{Skill\ A}$')
	for ax_ in ax:
		ax_.view_init(angle[0], angle[1])

	for i, c in zip(range(nb_samples), color_demo):
		a = 1
		ax[nb].plot(demos[i]['Data'][0, :], demos[i]['Data'][1, :], demos[i]['Data'][2, :],
					color=c, lw=1, alpha=a)
	# ax[nb].plot(demos[i]['Data'][7,:], demos[i]['Data'][8,:],'H',color=c,ms=10,alpha=a)

	nb += 1
	ax[nb].set_title(r'$\mathrm{Skill\ B}$')
	for i, c in zip(range(nb_samples), color_demo):
		a = 1
		ax[nb].plot(demos[i]['Data'][3, :], demos[i]['Data'][4, :], demos[i]['Data'][5, :],
					color=c, lw=1, alpha=a)


# ax[nb].plot(demos[i]['Data'][7,:], demos[i]['Data'][8,:],'H',color=c,ms=10,alpha=a)


def repro_plot(model, demos, save=False, tp_list=[], figsize=(3.5, 5)):
	nb_states = model.nb_states
	nb_tp = len(tp_list)
	idx = np.floor(np.linspace(1, 255, model.nb_states)).astype(int)
	color = cmap.viridis(range(256))[idx, 0:3]  # for states

	fig = plt.figure(3, figsize=(figsize[0] * nb_tp, figsize[1]))
	# fig.suptitle("Reproduction", fontsize=14, fontweight='bold')
	nb_plt = nb_tp * 2
	ax = []  # subplots
	label_size = 15
	t = 50  # timestep for reproduction

	# regress in first configuration
	i_in = [6, 7, 8]  # input dimension
	i_out = [0, 1, 2]  # output

	### specify subplots ###
	gs = gridspec.GridSpec(2, nb_tp, height_ratios=[4, 1])

	rn = []
	for i in range(nb_tp):
		rn += [i, i + nb_tp]

	for j in rn:  # [0, 2, 1, 3, ...]
		ax.append(fig.add_subplot(gs[j]))
	for a in ax:
		a.set_axis_bgcolor('white')

	for i in range(0, nb_tp * 2, 2):
		tp = tp_list[i / 2]
		data_in = tp[0]['b']
		model.regress(data_in - tp[1]['b'], i_in, i_out)
		prod_1 = model.prodgmm(tp)

		nb = i  # subplots counter
		ax[nb].set_title(r'$\mathrm{(a)}$')

		item_plt, = ax[nb].plot(data_in[0], data_in[1], '^', color=color[3], ms=12)

		pblt.plot_gmm(prod_1.Mu, prod_1.Sigma, dim=[0, 1], color=color,
					  alpha=model.PriorsR * nb_states, ax=ax[nb], nb=2)

		### plot state sequence ###
		nb = i + 1

		### get state sequence ###
		h = model.forward_variable_priors(t, model.PriorsR, start_priors=model.StatesPriors)

		for i in range(nb_states):
			ax[nb].plot(h[i, :], color=color[i])

	"""LEGEND, LABEL, ..."""
	for i in range(0, nb_plt, 2):
		# rob_plt, = ax[i].plot(40,40,'s',color=(1,0.4,0),ms=8,zorder=30)
		ax[i].set_aspect('equal', 'datalim')
		for j in [3, 4, 5, 6, 2]:
			demo_plt, = ax[i].plot(demos[j]['Glb'][0, :], demos[j]['Glb'][1, :], 'k:', lw=1,
								   alpha=1)

	for i in range(1, nb_plt, 2):
		#     ax[i].set_title(r'$\mathrm{forward\ variable}\, \alpha_t(z_n)$')
		ax[i].set_title(r'$\alpha_t(z_n)$', fontsize=16)
		ax[i].set_xlabel(r'$t\, \mathrm{[timestep]}$', fontsize=16)
		ax[i].set_ylim([-0.1, 1.1])
		ax[i].set_yticks(np.linspace(0, 1, 3))

	lgd = fig.legend([item_plt, demo_plt], ['obstacle position', 'Demonstrations']
					 , frameon=True, ncol=3,
					 bbox_to_anchor=(0.1, -0.01), loc='lower left', numpoints=1)
	frame = lgd.get_frame()
	# frame.set_facecolor('White')


	plt.tight_layout(pad=2.4, w_pad=0.9, h_pad=1.0)
	if save:
		plt.savefig('/home/idiap/epignat/thesis/paper/images/' + skill_name + '_repro.pdf',
					bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.show()


def plot_model(model, demos, figsize=(8, 3.5), skill_name='temp', save=False):
	nb_samples = len(demos)
	fig = plt.figure(2, figsize=figsize)
	# fig.suptitle("Model", fontsize=14, fontweight='bold')
	nb_plt = 3
	ax = []
	label_size = 15
	# plt.style.use('bmh')
	plt.style.use('ggplot')

	idx = np.floor(np.linspace(1, 255, model.nb_states)).astype(int)
	color = cmap.viridis(range(256))[idx, 0:3]  # for states

	nb = 0

	gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8])


	for j in range(nb_plt):
		ax.append(fig.add_subplot(gs[j]))
		ax[j].set_axis_bgcolor('white')

	ax[nb].set_title(r'$(a)\ j=1$')

	# for i in range(nb_samples):
	#    ax[nb].plot(demos[i]['Data'][4,0], demos[i]['Data'][5,0],'^',color=color[3],ms=10,alpha=0.5,zorder=30)


	for i in range(nb_samples):
		ax[nb].plot(demos[i]['Data'][0, :], demos[i]['Data'][1, :], 'k:', lw=1, alpha=1)

	pblt.plot_gmm(model.Mu, model.Sigma, dim=[0, 1], color=color, alpha=0.8, linewidth=1,
				  ax=ax[nb], nb=1)

	ax[nb].set_ylabel('z position [cm]')
	nb += 1

	ax[nb].set_title(r'$(b)\ j=2$')

	for i in range(nb_samples):
		demos_plt, = ax[nb].plot(demos[i]['Data'][3, :], demos[i]['Data'][4, :], 'k:', lw=1,
								 alpha=1)
	# ax[nb].plot(demos[i]['Data'][2,0], demos[i]['Data'][3,0],'H',color=c,ms=10)

	pblt.plot_gmm(model.Mu, model.Sigma, dim=[3, 4], color=color, alpha=0.8, linewidth=1,
				  ax=ax[nb], nb=1)

	nb += 1
	ax[nb].set_title(r'$(c)\ \mathrm{sensory}$')

	for i in range(nb_samples):
		sense_plt, = ax[nb].plot(demos[i]['Data'][6, 0], demos[i]['Data'][7, 0], '^',
								 color=color[3], ms=12, zorder=30)

	pblt.plot_gmm(model.Mu, model.Sigma, dim=[6, 7], color=color, alpha=0.5, ax=ax[nb],
				  nb=1)
	# ax[nb].set_xlim([-20,140])

	plt.tight_layout()

	lgd = fig.legend([demos_plt, sense_plt],
					 ['demonstrations', 'hand position'], frameon=True, ncol=2,
					 bbox_to_anchor=(0.4, -0.01), loc='lower left', numpoints=1)
	# frame = lgd.get_frame()
	# frame.set_facecolor('White')

	for i in range(nb_plt):
		# ax[i].plot(0, 0,'+',color='k',ms=20,zorder=30,lw=2)

		ax[i].set_xlabel('x position [cm]')

	plt.tight_layout(pad=2.8, w_pad=0.2, h_pad=-1.0)
	if save:
		plt.savefig('/home/idiap/epignat/thesis/paper/images/' + skill_name + '_model.pdf',
					bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_demos(demos, data_dim, figsize=(8,5)):
	nb_samples = len(demos)
	fig = plt.figure(2, figsize=figsize)
	# fig.suptitle("Model", fontsize=14, fontweight='bold')
	nb_plt = len(data_dim)
	ax = []
	label_size = 15
	# plt.style.use('bmh')
	plt.style.use('ggplot')
	nb = 0

	gs = gridspec.GridSpec(nb_plt, 1)

	for j in range(nb_plt):
		ax.append(fig.add_subplot(gs[j]))
		ax[j].set_axis_bgcolor('white')

	for j, dim in enumerate(data_dim):
		for i in range(nb_samples):
			ax[j].plot(demos[i]['Data'][dim,:].T)

def train_test(demos, demo_idx=0, nb_states=5, test=True, sensory=True, kbins=True,
			   hmmr=True,
			   nb_dim=3, nb_frames=2):
	demos_train = deepcopy(demos)
	nb_samples = len(demos)
	if test:
		demos_train.pop(demo_idx)
		nb_s = nb_samples - 1
	else:
		nb_s = nb_samples

	model = pbd.TP_HMM(nb_states, nb_dim=nb_dim, nb_frames=nb_frames)
	dep = [[0, 1], [2, 3], [4, 5]]

	Data_train = np.hstack([d['Data'] for d in demos_train])
	# model.init_hmm_kmeans(Data, nb_states, nb_samples, dep=dep)

	best = {'model': None, 'score': np.inf}
	for i in range(10):
		if sensory:
			model.init_hmm_gmm(Data_train, nb_states, nb_samples, dep=dep)
			scale = 8.
		else:
			if kbins:
				model.init_hmm_kbins(Data_train, nb_states, nb_s, dep=dep)
			else:
				model.init_hmm_kmeans(Data_train, nb_states, nb_samples, dep=dep,
									  dim_init=range(6))
			scale = 1e10

		if sensory:
			score = model.em_hmm(demos_train, dep=dep, reg=0.0002,
								 reg_diag=[1., 1., 1., 1., 1., 1., scale, scale, scale])
		else:
			score = model.em_hmm(demos_train, dep=dep, reg=0.0002,
								 reg_diag=[1., 1., 1., 1., 1., 1., scale, scale, scale],
								 end_cov=True)
		if score < best['score']:
			best['score'] = score

			best['model'] = deepcopy(model)

	print 'Best :', best['score']
	model = best['model']

	model.compute_duration(demos_train)

	# model.init_hmm_kbins(Data, nb_states, nb_samples, dep=dep)
	if hmmr:
		hmmr = pbd.hmmr.HMMR(model, nb_dim=3)

		min_dist = pow(5e-2, 3)
		hmmr.to_gmr(demos_train, mix_std=0.1, reg=min_dist, plot_on=False)
	else:
		hmmr = None

	return model, hmmr


def repro_demo(model, hmmr,demos,demo_idx=0,start_point=None,plot_on=False):
	nb_states = model.nb_states
	nb_samples = len(demos)


	t = 50  # timestep for reproduction
	# regress in first configuration
	i_in = [6, 7, 8]  # input dimension
	i_out = [0, 1, 2]  # output

	tp = deepcopy(demos[demo_idx]['TPs'])
	data_in = tp[0]['b']
	model.regress(data_in - tp[1]['b'], i_in, i_out, reg=0.01)
	prod_1 = model.prodgmm(tp)

	### get state sequence ###
	# print model.PriorsR
	h_1 = model.forward_variable_priors(t, model.PriorsR, start_priors=model.StatesPriors)

	hmmr.create_distribution_fwd(h_1, start_pos=None)  # 64.3 ms  ~1.5 ms per timestep
	prod_ph_1 = hmmr.prodgmm(tp)

	lqr = pbd.LQR(canonical=True, horizon=70, rFactor=-2.0, nb_dim=3)

	q = np.argmax(h_1, axis=0)
	# print q
	# make a rest at the end
	q = np.concatenate([q, np.ones(20) * q[-1]])

	lqr.set_hmm_problem(prod_ph_1, range(50) + [49] * 20)
	lqr.evaluate_gains_infiniteHorizon()

	plan, command = lqr.solve_hmm_problem(start_point)

	if plot_on:
		label_size = 15
		idx = np.floor(np.linspace(1, 255, 50)).astype(int)
		color_gmr = cmap.viridis(range(256))[idx, 0:3]  # for states
		idx = np.floor(np.linspace(1, 255, nb_states)).astype(int)
		color = cmap.viridis(range(256))[idx, 0:3]  # for states
		fig = plt.figure(3 + demo_idx, figsize=(5, 5))
		# fig.suptitle("Reproduction", fontsize=14, fontweight='bold')
		nb_plt = 2
		ax = []  # subplots
		### specify subplots ###
		gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[4, 1])

		for j in [0, 1]:
			ax.append(fig.add_subplot(gs[j]))
		for a in ax:
			a.set_axis_bgcolor('white')

		### plot regressed HMM ###
		nb = 0  # subplots counter
		ax[nb].set_title(r'$\mathrm{(a)}$')

		for j in range(nb_samples):
			demo_plt, = ax[0].plot(demos[j]['Glb'][0, :], demos[j]['Glb'][1, :], 'k:', lw=1,
								   alpha=1)

		ax[nb].plot(data_in[0], data_in[1], '^', color=color[-1], ms=12)

		pblt.plot_gmm(prod_1.Mu, prod_1.Sigma, dim=[0, 1], color=color,
					  alpha=model.PriorsR * nb_states, ax=ax[nb], nb=2)

		### plot state sequence ###
		nb += 1



		for i in range(nb_states):
			ax[nb].plot(h_1[i, :], color=color[i])

		ax[nb].set_ylim([-0.1, 1.1])

		pblt.plot_gmm(prod_ph_1.Mu, prod_ph_1.Sigma, dim=[0, 1], color=color_gmr, ax=ax[nb - 1],
					  nb=1)


		ax[0].plot(plan[0, :], plan[1, :], 'w', lw=2, zorder=50)
		ax[0].plot(demos[demo_idx]['Glb'][0, :], demos[demo_idx]['Glb'][1, :], 'k--', lw=3,
				   alpha=1, zorder=49)

	return np.copy(plan)