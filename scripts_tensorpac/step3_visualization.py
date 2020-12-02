from tensorpac import Pac
import numpy as np
import os
import matplotlib.pyplot as plt 
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

# f_init = 15
# f_end = 120
# f_res = 2

# pac_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/tensorpac'
# filename = '0_7_20_RD.npy'

# global channel_n

# channel_n = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
#              'C3', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
#              'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

# p = Pac(f_pha='hres', f_amp=(f_init, f_end, f_res, f_res), dcomplex='hilbert')

# data_subject = np.load(os.path.join(pac_path, filename))



# plot_connect_hipermatrix(data_subject, (2, 20), (f_init, f_end), filename)




# plot_avg_connectivity(data_subject)



import matplotlib.pyplot as plt 
from matplotlib import cm 
from matplotlib.colors import LinearSegmentedColormap

def _pigtailed_cmap(cmap, swap_order=('green', 'red', 'blue')):
    """ Utility function to make a new colormap by concatenating a
        colormap with its reverse.
    """
    orig_cdict = cmap._segmentdata.copy()

    cdict = dict()
    cdict['green'] = [(0.5 * (1 - p), c1, c2)
                      for (p, c1, c2) in reversed(orig_cdict[swap_order[0]])]
    cdict['blue'] = [(0.5 * (1 - p), c1, c2)
                     for (p, c1, c2) in reversed(orig_cdict[swap_order[1]])]
    cdict['red'] = [(0.5 * (1 - p), c1, c2)
                    for (p, c1, c2) in reversed(orig_cdict[swap_order[2]])]

    for color in ('red', 'green', 'blue'):
        cdict[color].extend([(0.5 * (1 + p), c1, c2)
                             for (p, c1, c2) in orig_cdict[color]])

    return cdict

class BandConnectivityPlotter(object):
    def __init__(self, channels=None, bands=None, cmap='coldhot'):
        self.channels = channels
        if channels==None:
            self.channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9', 
                            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                            'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']
        self.n_chan = len(self.channels)
        self.bands = bands
        if channels==None:
            self.bands = {'Delta': (0.5, 4),
                            'Theta': (4, 8),
                            'Alpha': (8, 12),
                            'Beta': (12, 25),
                            'Gamma': (25, 100),
                            'Epsilon': (100,200)}
        if cmap=='coldhot':
            self.cmap = LinearSegmentedColormap('coldhot', _pigtailed_cmap(cm.hot))
        else:
            self.cmap = cm.get_cmap(cmap)

    def check_bands(self, f):
        for band in self.bands:
            if f>self.bands[band][0] and f<=self.bands[band][1]:
                bands = band
        return bands

    def inverse_band_check(self, f_range):
        bands = set()
        for f in range(f_range[0], f_range[1], 2):
            bands.add(self.check_bands(f))
        list_def = [band for band in self.bands if band in bands]
        return list_def

    def display_band_connectivity(self, hipermatrix, f_pha=(2,20), f_amp=(15,120)):
        # Check which bands there are: 
        bands_phase = self.inverse_band_check(f_pha)
        bands_amp = self.inverse_band_check(f_amp)
        fig, ax = plt.subplots(nrows=len(bands_amp), ncols=len(bands_phase),
                                figsize=(6*len(bands_phase),6*len(bands_amp)))
        fpha_ref = np.linspace(f_pha[0], f_pha[1], hipermatrix.shape[-1])
        famp_ref = np.linspace(f_amp[0], f_amp[1], hipermatrix.shape[-2])
        for i, fas in enumerate(bands_phase):
            for j, famp in enumerate(bands_amp):
                row = len(bands_amp)-1-j
                limits_phas = self.bands[fas]
                limits_amp = self.bands[famp]
                mask = np.outer((fpha_ref<=limits_phas[1])&(fpha_ref>limits_phas[0]),
                                (famp_ref<=limits_amp[1])&(famp_ref>limits_amp[0]))
                matrix = np.nanmean(hipermatrix[:,:,mask.T], axis=-1)
                ax[row, i].imshow(matrix, origin='lower')
                if row==(len(bands_amp)-1):
                    ax[row, i].set_xlabel(fas)
                if i==0:
                    ax[row, i].set_ylabel(famp)
                ax[row, i].set_yticks(range(self.n_chan))
                ax[row, i].set_xticks(range(self.n_chan))
                ax[row, i].set_yticklabels(self.channels)
                ax[row, i].set_xticklabels(self.channels,rotation=60)
        
    def display_avg_connectivity(self, hipermatrix):
        _, ax  = plt.subplots(figsize=(7,7))
        if np.nanmin(hipermatrix)<0:
            vmax = np.nanmax(abs(hipermatrix))
            vmin = -vmax
            colormap = 'coolwarm'
        else:
            vmin, vmax = 0, np.nanmax(hipermatrix)
            colormap = 'inferno'
        val = np.nanmean(np.nanmean(hipermatrix, axis=-1), axis=-1)
        f = ax.matshow(val, vmin=vmin, vmax=vmax, cmap= colormap)
        ax.set_yticks(range(self.n_chan))
        ax.set_xticks(range(self.n_chan))
        ax.set_yticklabels(self.channels)
        ax.set_xticklabels(self.channels,rotation=60)
        plt.colorbar(f)

    def plot_connect_hipermatrix(self, hipermatrix, f_pha, f_amp, filename):
        n_chan, _ , _, _ = hipermatrix.shape
        fig, ax = plt.subplots(n_chan, n_chan, figsize=(n_chan*1.5,n_chan*1.5))
        vmin, vmax = np.nanmin(hipermatrix), np.nanmax(hipermatrix)
        if vmin<0:
            vmax = max(-vmin, vmax)
            vmin = -vmax
        else:
            self.cmap = 'inferno'
        for giving_phase in range(n_chan):
            for receiving_amp in range(n_chan):
                matris = hipermatrix[giving_phase,receiving_amp]
                # matris[np.isnan(matris)] = 0.
                ax[receiving_amp, giving_phase].imshow(matris, cmap=self.cmap, origin='lower',
                                vmin=vmin, vmax=vmax, aspect='auto',
                                extent=[f_pha[0], f_pha[1], f_amp[0], f_amp[1]])
                if receiving_amp==(n_chan-1):
                    ax[receiving_amp, giving_phase].set_xlabel(self.channels[giving_phase])
                else:
                    ax[receiving_amp, giving_phase].set_xticklabels([])
                if giving_phase==0:
                    ax[receiving_amp, giving_phase].set_ylabel(self.channels[receiving_amp])
                else:
                    ax[receiving_amp, giving_phase].set_yticklabels([])
        #plt.tight_layout()
        plt.show()
        fig.savefig(f'{filename}.pdf')



#%% 
data = np.load('pvalue_clusterbased.npy')

th = 0.025
data_sig = data.copy()
data_sig[(data>th) & (data<1-th)] = np.nan

plotter = BandConnectivityPlotter()

plotter.display_band_connectivity(data_sig, f_pha=(2,20), f_amp=(15,120))
# %%
