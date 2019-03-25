import numpy as np

def gaussian_moment_matching(mus, sigmas, h=None):
	"""

	:param mu:			[np.array([nb_states, nb_timestep, nb_dim])]
				or [np.array([nb_states, nb_dim])]
	:param sigma:		[np.array([nb_states, nb_timestep, nb_dim, nb_dim])]
				or [np.array([nb_states, nb_dim, nb_dim])]
	:param h: 			[np.array([nb_timestep, nb_states])]
	:return:
	"""

	if h is None:
		h = np.ones((mus.shape[1], mus.shape[0]))/ mus.shape[0]

	if h.ndim == 1:
		h = h[None]

	if mus.ndim == 3:
		mu = np.einsum('ak,kai->ai', h, mus)
		dmus = mus - mu[None]  # nb_timesteps, nb_states, nb_dim
		if sigmas.ndim == 4:
			sigma = np.einsum('ak,kaij->aij', h, sigmas) + \
				 np.einsum('ak,akij->aij', h, np.einsum('kai,kaj->akij', dmus, dmus))
		else:
			sigma = np.einsum('ak,kij->aij', h, sigmas) + \
				 np.einsum('ak,akij->aij', h, np.einsum('kai,kaj->akij', dmus, dmus))

		return mu, sigma
	else:
		mu = np.einsum('ak,ki->ai', h, mus)
		dmus = mus[None] - mu[:, None] # nb_timesteps, nb_states, nb_dim
		sigma = np.einsum('ak,kij->aij', h, sigmas) + \
				 np.einsum('ak,akij->aij',h , np.einsum('aki,akj->akij', dmus, dmus))

		return mu, sigma

def gaussian_conditioning(mu, sigma, data_in, dim_in, dim_out, reg=None):
	"""

	:param mu: 			[np.array([nb_timestep, nb_dim])]
	:param sigma: 		[np.array([nb_timestep, nb_dim, nb_dim])]
	:param data_in: 	[np.array([nb_timestep, nb_dim])]
	:param dim_in: 		[slice]
	:param dim_out: 	[slice]
	:return:
	"""
	if sigma.ndim == 2:

		if reg is None:
			inv_sigma_in_in = np.linalg.inv(sigma[dim_in, dim_in])
		else:
			reg = reg * np.eye(dim_in.stop - dim_in.start)
			inv_sigma_in_in = np.linalg.inv(sigma[dim_in, dim_in] + reg)

		inv_sigma_out_in = np.einsum('ji,jk->ik', sigma[dim_in, dim_out], inv_sigma_in_in)
		mu_cond = mu[dim_out] + np.einsum('ij,aj->ai', inv_sigma_out_in,
											 data_in - mu[dim_in])
		sigma_cond = sigma[dim_out, dim_out] - np.einsum('ij,jk->ik', inv_sigma_out_in,
															sigma[dim_in, dim_out])

	else:

		if reg is None:
			inv_sigma_in_in = np.linalg.inv(sigma[:, dim_in, dim_in])
		else:
			reg = reg * np.eye(dim_in.stop - dim_in.start)
			inv_sigma_in_in = np.linalg.inv(sigma[:, dim_in, dim_in] + reg)

		inv_sigma_out_in = np.einsum('aji,ajk->aik', sigma[:, dim_in, dim_out], inv_sigma_in_in)
		mu_cond = mu[:, dim_out] + np.einsum('aij,aj->ai', inv_sigma_out_in,
											 data_in - mu[:, dim_in])
		sigma_cond = sigma[:, dim_out, dim_out] - np.einsum('aij,ajk->aik', inv_sigma_out_in,
															sigma[:, dim_in, dim_out])

	return mu_cond, sigma_cond