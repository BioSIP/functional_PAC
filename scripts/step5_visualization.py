import numpy as np
import visbrain as vs
import os

import matplotlib.pyplot as plt

global bands
bands = {'Delta': (0.5, 4),
         'Theta': (4, 8),
         'Alpha': (8, 12),
         'Beta': (12, 25),
         'Gamma': (25, 140)}

def plot_band_connectivity(PACband, bands, thr=None, ch=1):
    ch_names_g = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
                  'C3', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                  'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']
    nx, ny = PACband.shape[2:4]
    fig, ax = plt.subplots(nx, ny, figsize=(30, 25))
    if ch==1:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = PACband[...,ch].min(),PACband[...,ch].max()
    for bx in range(nx):
        for by in range(ny):
            if thr is None:
                im = ax[bx][by].imshow(PACband[:, :, bx, by, ch], vmin=vmin, vmax=vmax)
            else:
                im = ax[bx][by].imshow(PACband[:, :, bx, by, ch]>thr, vmin=vmin, vmax=vmax)
            ax[bx][by].set_xticks(np.arange(31))
            ax[bx][by].set_xticklabels(ch_names_g, rotation=90)
            ax[bx][by].set_yticks(np.arange(31))
            ax[bx][by].set_yticklabels(ch_names_g)
            if bx == 4:
                ax[bx][by].set_xlabel(f'Band {list(bands.keys())[by]}')
            if by == 0:
                ax[bx][by].set_ylabel(f'Band {list(bands.keys())[bx]}')
    fig.colorbar(im, ax=ax[:, :], shrink=0.6)

def plot_band_multicolor(PACband, bands, thr=0.95):
    ch_names_g = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
                  'C3', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                  'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']
    nx, ny = PACband.shape[2:4]
    ticklabels = [None]
    bandnames = list(bands.keys())
    fig, ax = plt.subplots(figsize=(12, 10))
    pacnan = np.ones(PACband.shape[:2])*np.nan
    val = 1
    for bx in range(nx):
        for by in range(bx, ny):
            select = PACband[:, :, bx, by, 1]>thr
            pacnan[select] = val
            val += 1
            ticklabels.append(f'({bandnames[bx]}, {bandnames[by]})')

    im = ax.imshow(pacnan, vmin=0, vmax=20, cmap='tab20c')
    ax.set_xticks(np.arange(31))
    ax.set_xticklabels(ch_names_g, rotation=90)
    ax.set_yticks(np.arange(31))
    ax.set_yticklabels(ch_names_g)
    ax.set_xlabel('Channel giving phase')
    ax.set_ylabel('Channel receiving amplitude')
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, ticks=np.arange(21))
    for _ in range(21-len(ticklabels)):
        ticklabels.append(None)
    cbar.ax.set_yticklabels(ticklabels)


descriptor_dir = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/descriptors'
filename = 'guardar_pac_band.npy'
PACband = np.load(os.path.join(descriptor_dir, filename))
    
plot_band_connectivity(PACband, bands, ch=1, thr=0.95)

plot_band_multicolor(PACband, bands, thr=0.95)


