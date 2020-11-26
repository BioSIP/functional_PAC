from tensorpac import Pac
import numpy as np
import os
import matplotlib.pyplot as plt 

f_init = 15
f_end = 120
f_res = 2

pac_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/tensorpac'
filename = '0_7_20_RD.npy'

global channel_n

channel_n = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
             'C3', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
             'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

p = Pac(f_pha='hres', f_amp=(f_init, f_end, f_res, f_res), dcomplex='hilbert')

data_subject = np.load(os.path.join(pac_path, filename))

def plot_connect_hipermatrix(hipermatriz, p):
    fig, ax = plt.subplots(31, 31, figsize=(40,40))
    n_chan, _ , _, _ = hipermatriz.shape
    for i in range(n_chan):
        for j in range(n_chan):
            plt.sca(ax[i,j])
            title = p.method.replace(' (', f' ({i})\n(')
            title += 'channels ({i}, {j})'
            p.comodulogram(hipermatriz[i,j], title=title, cmap='inferno')
            if i==0:
                ax[i,j].set_xlabel(channel_n[j])
            if j==0:
                ax[i,j].set_ylabel(channel_n[i])
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{filename}.pdf')


plot_connect_hipermatrix(data_subject, p)


def plot_avg_connectivity(hipermatriz):
    _, ax  = plt.subplots(figsize=(7,7))
    ax.matshow(hipermatriz.mean(axis=-1).mean(axis=-1))
    ax.set_yticks(range(31))
    ax.set_xticks(range(31))
    ax.set_yticklabels(channel_n)
    ax.set_xticklabels(channel_n,rotation=60)


plot_avg_connectivity(data_subject)