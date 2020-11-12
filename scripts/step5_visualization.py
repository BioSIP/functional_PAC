import numpy as np
import visbrain as vs
import os

import matplotlib.pyplot as plt

global bands
global channel_n
bands = {'Delta': (0.5, 4),
         'Theta': (4, 8),
         'Alpha': (8, 12),
         'Beta': (12, 25),
         'Gamma': (25, 140)}

channel_n = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
                  'C3', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                  'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

def plot_band_connectivity(PACband, bands, thr=None, ch=1):
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
            ax[bx][by].set_xticklabels(channel_n, rotation=90)
            ax[bx][by].set_yticks(np.arange(31))
            ax[bx][by].set_yticklabels(channel_n)
            if bx == 4:
                ax[bx][by].set_xlabel(f'Band {list(bands.keys())[by]}')
            if by == 0:
                ax[bx][by].set_ylabel(f'Band {list(bands.keys())[bx]}')
    fig.colorbar(im, ax=ax[:, :], shrink=0.6)

def plot_band_multicolor(PACband, bands, thr=0.95):
    
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
    ax.set_xticklabels(channel_n, rotation=90)
    ax.set_yticks(np.arange(31))
    ax.set_yticklabels(channel_n)
    ax.set_xlabel('Channel giving phase')
    ax.set_ylabel('Channel receiving amplitude')
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, ticks=np.arange(21))
    for _ in range(21-len(ticklabels)):
        ticklabels.append(None)
    cbar.ax.set_yticklabels(ticklabels)

subject_id=0
descriptor_dir = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/descriptors'
filename = f'subject_{subject_id}_32_band.npy'
PACband = np.load(os.path.join(descriptor_dir, filename))
    
plot_band_connectivity(PACband, bands, ch=1, thr=0.95)

# plot_band_multicolor(PACband, bands, thr=0.95)

# from nilearn import plotting
# # correlation_matrix = PACband[...,0,1,0].copy()
# # np.fill_diagonal(correlation_matrix, 0)
# # plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=channel_n, reorder=True)

# def plot_nilearn_connectivity(PACband, bands, thr=None, ch=1, diag=False,
#                                 reorder=True):
#     nx, ny = PACband.shape[2:4]
#     _ , ax = plt.subplots(nx, ny, figsize=(30, 25))
#     for bx in range(nx):
#         for by in range(ny):
#             if ch==1:
#                 vmin, vmax = -1, 1
#             else:
#                 vmin, vmax = min(PACband[...,bx,by,ch].min(),0),PACband[...,bx,by,ch].max()
#             if thr is None:
#                 correlation_matrix = PACband[...,bx,by,ch].copy()
#             else:
#                 correlation_matrix = PACband[...,bx,by,ch].copy()>thr
#             if diag:
#                 np.fill_diagonal(correlation_matrix, 0)
#             plotting.plot_matrix(correlation_matrix, axes=ax[bx,by], 
#                                     labels=channel_n, reorder=reorder, 
#                                     vmin=vmin, vmax=vmax)
#             if bx == 4:
#                 ax[bx][by].set_xlabel(f'Band {list(bands.keys())[by]}')
#             if by == 0:
#                 ax[bx][by].set_ylabel(f'Band {list(bands.keys())[bx]}')

# plot_nilearn_connectivity(PACband, bands, ch=1, thr=0.95)


from visbrain.objects import TopoObj, ColorbarObj, SceneObj

def plot_visbrain_connectivity(PACband, subject_id):
    kw_top = dict(margin=15 / 100, chan_offset=(0., 0.1, 0.), chan_size=10)
    kw_cbar = dict(cbtxtsz=12, txtsz=10., width=.3, txtcolor='black', cbtxtsh=1.8,
                rect=(0., -2., 1., 4.), border=False)
    sc = SceneObj(bgcolor='white', size=(3000, 1300))
    nx, ny = PACband.shape[2:4]
    for bx in range(nx):
        for by in range(bx,ny):
            correlation_matrix = PACband[...,bx,by,0]
            data = np.diag(correlation_matrix)
            t_obj = TopoObj('topo', data, channels=channel_n, cmap='Spectral_r', **kw_top)
            cb_obj = ColorbarObj(t_obj, cblabel='MVL pac measure', **kw_cbar)
            connect = PACband[...,bx,by,1]
            np.fill_diagonal(connect, 0)
            select = connect > 0.95
            if np.any(select):
                t_obj.connect(connect, select = select, cmap='inferno', antialias=True,line_width=4.)
            sc.add_to_subplot(t_obj, row=by, col=bx*2, title=f'Phase:{list(bands.keys())[bx]} to Amp:{list(bands.keys())[by]}',
                            title_color='black', width_max=400)
            sc.add_to_subplot(cb_obj, row=by, col=bx*2+1, width_max=100)
    sc.link(-1)
    sc.screenshot(f'connect_subject_{subject_id}.eps', print_size=(30,15))

plot_visbrain_connectivity(PACband, subject_id)