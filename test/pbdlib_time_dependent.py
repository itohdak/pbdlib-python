import sys
sys.path.append('..')

import numpy as np
import os
import matplotlib.pyplot as plt
import pbdlib as pbd

np.set_printoptions(precision=2)

from scipy.io import loadmat # loading data from matlab
import argparse

def load_sample_data():
    datapath = os.path.dirname(pbd.__file__) + '/data/gui/'
    data = np.load(datapath + 'test_001.npy', allow_pickle=True)[()]

    demos_x = data['x']  #Position data
    demos_dx = data['dx'] # Velocity data
    demos_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(demos_x, demos_dx)] # Position-velocity
    return demos_x, demos_dx, demos_xdx

def show_sample_data():
    demos_x, demos_dx, demos_xdx = load_sample_data()
    for d in demos_x:
        plt.axes().set_prop_cycle(None)
        plt.plot(d)
    plt.show()

def show_aligned_trajectories():
    demos_x, demos_dx, demos_xdx = load_sample_data()
    demos_x, demos_dx, demos_xdx = pbd.utils.align_trajectories(demos_x, [demos_dx, demos_xdx])
    t = np.linspace(0, 100, demos_x[0].shape[0])

    fig, ax = plt.subplots(nrows=2)
    for d in demos_x:
        ax[0].set_prop_cycle(None)
        ax[0].plot(d)

    ax[1].plot(t)
    plt.show()

def augment_data():
    demos_x, demos_dx, demos_xdx = load_sample_data()
    demos_x, demos_dx, demos_xdx = pbd.utils.align_trajectories(demos_x, [demos_dx, demos_xdx])
    t = np.linspace(0, 100, demos_x[0].shape[0])

    demos = [np.hstack([t[:,None], d]) for d in demos_xdx]
    data = np.vstack([d for d in demos])

    model = pbd.GMM(nb_states=4, nb_dim=5)

    model.init_hmm_kbins(demos) # initializing model

    # EM to train model
    model.em(data, reg=[0.1, 1., 1., 1., 1.])

    # plotting
    fig, ax = plt.subplots(nrows=4)
    fig.set_size_inches(12,7.5)

    # position plotting
    for i in range(4):
        for p in demos:
            ax[i].plot(p[:, 0], p[:, i + 1])
        pbd.plot_gmm(model.mu, model.sigma, ax=ax[i], dim=[0, i + 1]);
    plt.show()

def synthesis():
    demos_x, demos_dx, demos_xdx = load_sample_data()
    demos_x, demos_dx, demos_xdx = pbd.utils.align_trajectories(demos_x, [demos_dx, demos_xdx])
    t = np.linspace(0, 100, demos_x[0].shape[0])

    demos = [np.hstack([t[:,None], d]) for d in demos_xdx]
    data = np.vstack([d for d in demos])

    model = pbd.GMM(nb_states=4, nb_dim=5)

    model.init_hmm_kbins(demos) # initializing model

    # EM to train model
    model.em(data, reg=[0.1, 1., 1., 1., 1.])

    mu, sigma = model.condition(t[:, None], dim_in=slice(0, 1), dim_out=slice(1, 5))

    pbd.plot_gmm(mu, sigma, dim=[0, 1], color='orangered', alpha=0.3)

    for d in demos_x:
        plt.plot(d[:, 0], d[:, 1])
    plt.show()

def import_data(letter='C'): # choose a letter in the alphabet
    datapath = os.path.dirname(pbd.__file__) + '/data/2Dletters/'
    data = loadmat(datapath + '%s.mat' % letter)
    demos_x = [d['pos'][0][0].T for d in data['demos'][0]] # Position data
    demos_dx = [d['vel'][0][0].T for d in data['demos'][0]] # Velocity data
    demos_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(demos_x, demos_dx)] # Position-velocity
    return demos_x, demos_dx, demos_xdx

def import_data_from_path(path):
    data = np.load(path, allow_pickle=True)
    demos_x = data.all()['x']
    demos_dx = data.all()['dx']
    demos_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(demos_x, demos_dx)] # Position-velocity
    return demos_x, demos_dx, demos_xdx

def hmm(letter='C', gui=True):
    if letter.isalpha() == 1:
        print('load alphabet')
        demos_x, demos_dx, demos_xdx = import_data(letter)
    else:
        print('load from path')
        demos_x, demos_dx, demos_xdx = import_data_from_path(letter)
    model = pbd.HMM(nb_states=4, nb_dim=4)

    model.init_hmm_kbins(demos_xdx) # initializing model

    # EM to train model
    model.em(demos_xdx, reg=1e-3)


    # plotting
    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(12,3.5)

    # position plotting
    ax[0].set_title('pos')
    for p in demos_x:
        ax[0].plot(p[:, 0], p[:, 1])

    pbd.plot_gmm(model.mu, model.sigma, ax=ax[0], dim=[0, 1]);

    # velocity plotting
    ax[1].set_title('vel')
    for p in demos_dx:
        ax[1].plot(p[:, 0], p[:, 1])

    pbd.plot_gmm(model.mu, model.sigma, ax=ax[1], dim=[2, 3]);


    # plotting transition matrix
    ax[2].set_title('transition')
    ax[2].imshow(np.log(model.Trans+1e-10), interpolation='nearest', vmin=-5, cmap='viridis');
    plt.tight_layout()
    if gui:
        plt.show()
    return demos_x, demos_dx, demos_xdx, model

def lqr(letter='C'):
    demos_x, demos_dx, demos_xdx, model = hmm(letter=letter, gui=False)
    demo_idx = 0
    sq = model.viterbi(demos_xdx[demo_idx])

    plt.figure(figsize=(5, 1))
    # plt.axis('off')
    plt.plot(sq, lw=3);
    plt.xlabel('timestep');
    plt.show()

def solve_lqr(letter='C', dt=0.01):
    demos_x, demos_dx, demos_xdx, model = hmm(letter=letter, gui=False)
    demo_idx = 3
    sq = model.viterbi(demos_xdx[demo_idx])

    lqr = pbd.PoGLQR(nb_dim=2, dt=dt, horizon=demos_xdx[demo_idx].shape[0])
    lqr.mvn_xi = model.concatenate_gaussian(sq)
    lqr.mvn_u = -4.
    lqr.x0 = demos_xdx[demo_idx][0]

    xi = lqr.seq_xi

    fig, ax = plt.subplots(ncols=2)

    fig.set_size_inches(8,3.5)


    # position plotting
    ax[0].set_title('position')
    for p in demos_x:
        ax[0].plot(p[:, 0], p[:, 1], alpha=0.4)

    ax[0].plot(xi[:, 0], xi[:, 1], 'b', lw=3)
    pbd.plot_gmm(model.mu, model.sigma, ax=ax[0], dim=[0, 1]);

    # velocity plotting
    ax[1].set_title('velocity')
    for p in demos_dx:
        ax[1].plot(p[:, 0], p[:, 1], alpha=0.4)

    ax[1].plot(xi[:, 2], xi[:, 3], 'b', lw=3, label='repro')

    plt.legend()
    pbd.plot_gmm(model.mu, model.sigma, ax=ax[1], dim=[2, 3]);
    plt.show()


if __name__ == '__main__':
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    parser.add_argument(
        '-f', '--filename', dest='filename', type=str,
        default='', help='filename for demos'
    )
    args = parser.parse_args()
    if args.filename == '':
        print('No file selected. Use alphabet C.')
        solve_lqr()
    else:
        solve_lqr('./data/' + args.filename + '.npy', dt=0.05)

    # hmm('./data/1.npy')
    # solve_lqr('./data/10.npy')
    # solve_lqr('./data/output.npy', dt=0.05)
    # solve_lqr('Z')

