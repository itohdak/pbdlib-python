import sys
sys.path.append('..')

import numpy as np
import os
import matplotlib.pyplot as plt
import pbdlib as pbd

np.set_printoptions(precision=2)

from scipy.io import loadmat # loading data from matlab

def import_data(letter='C'): # choose a letter in the alphabet
    datapath = os.path.dirname(pbd.__file__) + '/data/2Dletters/'
    data = loadmat(datapath + '%s.mat' % letter)
    demos = [d['pos'][0][0].T for d in data['demos'][0]] # cleaning awful matlab data
    return demos

def import_data_from_path(path):
    data = np.load(path, allow_pickle=True)
    data = data.all()['x']
    demos = data
    return data

def test_HMM(letter='C', score=False, nb_states=4):
    if letter.isalpha() == 1:
        print('load alphabet')
        demos = import_data(letter)
    else:
        print('load from path')
        demos = import_data_from_path(letter)

    # 3 dimentionalize
    demos = [np.concatenate([demo, np.zeros((demo.shape[0], 1))], axis=1) for demo in demos]

    model = pbd.HMM(nb_states=nb_states, nb_dim=2)

    model.init_hmm_kbins(demos) # initializing model

    # plotting
    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(5,2.5)
    if model.mu.shape[1] == 2:
        pbd.plot_gmm(model.mu, model.sigma, alpha=0.5, color='steelblue', ax=ax[0]); # plot after init only
    else:
        pbd.plot_gmm(model.mu[:,:2], model.sigma[:,:2,:2], alpha=0.5, color='steelblue', ax=ax[0]); # plot after init only

    # EM to train model
    model.em(demos, reg=1e-3)

    # plotting demos
    for p in demos:
        ax[0].plot(p[:, 0], p[:, 1])

    if model.mu.shape[1] == 2:
        pbd.plot_gmm(model.mu, model.sigma, ax=ax[0]);
    else:
        pbd.plot_gmm(model.mu[:,:2], model.sigma[:,:2,:2], ax=ax[0]);

    if score:
        print(model.score(demos))

    # plotting transition matrix
    ax[1].imshow(np.log(model.Trans+1e-10), interpolation='nearest', vmin=-5, cmap='viridis');
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_HMM('./data/10.npy', score=True, nb_states=7)

