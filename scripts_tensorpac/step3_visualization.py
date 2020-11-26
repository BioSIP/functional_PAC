from tensorpac import Pac
import numpy as np
import os
import matplotlib.pyplot as plt 
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

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

def plot_connect_hipermatrix(hipermatriz, f_pha, f_amp, filename):
    n_chan, _ , _, _ = hipermatriz.shape
    fig, ax = plt.subplots(n_chan, n_chan, figsize=(n_chan*1.5,n_chan*1.5))
    vmin, vmax = np.nanmin(hipermatriz), np.nanmax(hipermatriz)
    colormap='inferno'
    if vmin<0:
        vmax = max(-vmin, vmax)
        vmin = -vmax
        colormap = nilearn_cmaps['cold_hot']
    for i in range(n_chan):
        for j in range(n_chan):
            matris = hipermatriz[i,j]
            matris[np.isnan(matris)] = 0.
            ax[i,j].imshow(matris, cmap=colormap, origin='lower',
                            vmin=vmin, vmax=vmax, aspect='auto',
                            extent=[f_pha[0], f_pha[1], f_amp[0], f_amp[1]])
            if i==(n_chan-1):
                ax[i,j].set_xlabel(channel_n[j])
            else:
                ax[i,j].set_xticklabels([])
            if j==0:
                ax[i,j].set_ylabel(channel_n[i])
            else:
                ax[i,j].set_yticklabels([])
    #plt.tight_layout()
    plt.show()
    fig.savefig(f'{filename}.pdf')


plot_connect_hipermatrix(data_subject, (2, 20), (f_init, f_end), filename)


def plot_avg_connectivity(hipermatriz):
    _, ax  = plt.subplots(figsize=(7,7))
    ax.matshow(hipermatriz.mean(axis=-1).mean(axis=-1))
    ax.set_yticks(range(31))
    ax.set_xticks(range(31))
    ax.set_yticklabels(channel_n)
    ax.set_xticklabels(channel_n,rotation=60)


plot_avg_connectivity(data_subject)