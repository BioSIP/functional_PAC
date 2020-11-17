import numpy as np
import visbrain as vs
import os
import re
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

descriptor_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/descriptors'


filenames = [fi for fi in os.listdir(descriptor_path) if fi.endswith('npy')]


filename = filenames[0]

def load_subgroup(path, stimulus=2, group='C'):
    filenames = [fi for fi in os.listdir(descriptor_path) if fi.endswith('npy')]
    r = re.compile(f'\d*_7_{stimulus}_{group}_band.npy')
    filenames_r = list(filter(r.match, filenames))
    N = len(filenames_r)
    data = np.zeros((N, len(channel_n), len(channel_n), len(bands), len(bands), 2))
    for i,fil in enumerate(filenames_r):
        data[i] = np.load(os.path.join(path, fil))
    return data

datos_ctl = load_subgroup(descriptor_path, group='C')

datos_dlx = load_subgroup(descriptor_path, group='RD')

def plot_band_connectivity(PACband, bands, thr=None, ch=1, fusion=False):
    nx, ny = PACband.shape[2:4]
    fig, ax = plt.subplots(nx, ny, figsize=(30, 25))
    # if ch==1:
    #     vmin, vmax = 0, 1
    # else:
    for bx in range(nx):
        for by in range(bx,ny):
            if thr is None:
                vmin, vmax = PACband[...,ch].min(),PACband[...,ch].max()
                im = ax[bx][by].imshow(PACband[:, :, bx, by, ch], vmin=vmin, vmax=vmax)
            else:
                display = np.abs(PACband[:, :, bx, by, ch])>thr
                if fusion:
                    display = PACband[:,:,bx,by,0]*display
                vmin, vmax = display[...,ch].min(),display[...,ch].max()
                im = ax[bx][by].imshow(display, vmin=vmin, vmax=vmax)
            ax[bx][by].set_xticks(np.arange(31))
            ax[bx][by].set_xticklabels(channel_n, rotation=90)
            ax[bx][by].set_yticks(np.arange(31))
            ax[bx][by].set_yticklabels(channel_n)
            if bx == 0:
                ax[bx][by].set_title(f'Band {list(bands.keys())[by]}')
            if by == 0:
                ax[bx][by].set_ylabel(f'Band {list(bands.keys())[bx]}')
    fig.colorbar(im, ax=ax[:, :], shrink=0.6)
    return fig

plot_band_connectivity(np.nanmean(datos_ctl,axis=0), bands, ch=1, thr=0.6, fusion=True)
plot_band_connectivity(np.nanmean(datos_dlx,axis=0), bands, ch=1, thr=0.6, fusion=True)


datos_avg = np.nanmean(datos_ctl,axis=0)
datos_std = np.nanstd(datos_ctl, axis=0)
z_ctl = (datos_ctl-datos_avg[None,...])/datos_std[None,...]


datos_avg = np.nanmean(datos_dlx,axis=0)
datos_std = np.nanstd(datos_dlx, axis=0)
z_dlx = (datos_dlx-datos_avg[None,...])/datos_std[None,...]

plot_band_connectivity(np.nanmean(z_ctl,axis=0), bands, ch=0)#, thr=0.6, fusion=True)
plot_band_connectivity(np.nanmean(z_dlx,axis=0), bands, ch=0)#, thr=0.6, fusion=True)


from visbrain.objects import TopoObj, ColorbarObj, SceneObj

def plot_visbrain_connectivity(PACband, group, thr=0.94):
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
            select = connect > thr
            if np.any(select):
                t_obj.connect(correlation_matrix, select = select, cmap='inferno', antialias=True,line_width=4.)
            sc.add_to_subplot(t_obj, row=bx, col=by*2, title=f'Phase:{list(bands.keys())[bx]} to Amp:{list(bands.keys())[by]}',
                            title_color='black', width_max=400)
            sc.add_to_subplot(cb_obj, row=bx, col=by*2+1, width_max=100)
    sc.link(-1)
    sc.screenshot(f'connect_avg_{group}.eps', print_size=(30,15))

plot_visbrain_connectivity(datos_avg, 'control', thr=0.7)

from scipy.stats import ttest_ind, mannwhitneyu

def tstatistic(group_a,group_b, formula=ttest_ind):
    stats = np.zeros(group_a.shape[1:])
    stats_p = np.zeros(group_a.shape[1:])
    for cx in range(stats.shape[0]):
        for cy in range(stats.shape[1]):
            for bx in range(stats.shape[2]):
                for by in range(stats.shape[3]):
                    vala = group_a[:,cx,cy,bx,by]
                    valb = group_b[:,cx,cy,bx,by]
                    t, p = formula(vala[np.isnan(vala[:,0])==False,0],
                                                    valb[np.isnan(valb[:,0])==False,0])
                    stats[cx,cy,bx,by,0] = t
                    stats_p[cx,cy,bx,by,0] = p
                    t, p = formula(vala[np.isnan(vala[:,1])==False,1],
                                                    valb[np.isnan(valb[:,1])==False,1])
                    stats[cx,cy,bx,by,1] = t
                    stats_p[cx,cy,bx,by,1] = p
    return stats, stats_p

stats, stats_p = tstatistic(datos_ctl, datos_dlx, formula=mannwhitneyu)

M= np.prod(stats.shape[0:-1])
alfa = 0.05
plot_band_connectivity(stats_p<=(1-(1-alfa)**(1/M)), bands, ch=0)
plot_band_connectivity(stats_p, bands, ch=0, thr=2.96)

# def permutation_statistics(group_a, group_b):
plt.hist(stats.flatten())