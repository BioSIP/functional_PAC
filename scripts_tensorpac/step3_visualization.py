from tensorpac import Pac
import numpy as np
import os
import matplotlib.pyplot as plt 
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

point_loc = {'Fp1': (-82.5, -19.2, -46.0),
 'Fp2': (82.2, -19.2, -46.0) ,
 'F7': (53.4, 66.8, -10.3)  ,
 'F3': (-67.2, 39.9, -47.9) ,
 'Fz': (-61.8, 46.0, 16.2)  ,
 'F4': (-26.4, 54.6, 57.8)  ,
 'F8': (28.3, 55.2, 57.1)   ,
 'FC5': (69.1, 40.3, -48.4)  ,
 'FC1': (-77.4, 13.5, -10.7) ,
 'FC2': (-57.7, 21.8, 53.2)  ,
 'FC6': (0.4, 26.3, 85.0)    ,
 'T7': (78.4, 14.8, -10.9)  ,
 'C3': (-82.3, -15.2, -46.3),
 'C4': (-34.7, -9.6, 86.0)  ,
 'T8': (36.1, -9.2, 84.7)   ,
 'TP9': (80.0, -12.2, 28.0)  ,
 'CP5': (82.0, -15.7, -46.3) ,
 'CP1': (-81.3, -44.1, -6.8) ,
 'CP2': (-60.9, -45.1, 62.9) ,
 'CP6': (0.4, -45.4, 95.3)   ,
 'TP10': (63.8, -44.7, 62.9)  ,
 'P7': (82.0, -43.7, -6.8)  ,
 'P3': (-70.0, -70.7, -39.3),
 'Pz': (-64.5, -73.1, 27.2) ,
 'P4': (-27.4, -77.2, 72.3) ,
 'P8': (30.6, -77.1, 73.5)  ,
 'PO9': (65.1, -72.8, 26.9)  ,
 'PO10': (35.3, -96.7, 34.9)  ,
 'O1': (47.8, -95.3, 20.8)  ,
 'Oz': (53.4, -93.6, 2.6)   ,
 'O2': (52.7, -94.0, -34.1) }

#%%

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

    def plot_pac(self, hipermatrix, f_pha=(2,20), f_amp=(15,120), channels=[(0,0)]):
        n_plots = len(channels)
        ncols = max(1, np.ceil(np.sqrt(n_plots)).astype(int))
        nrows = max(1, np.floor(np.sqrt(n_plots)).astype(int))
        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
        ax = ax.flatten()
        vmin, vmax = np.nanmin(hipermatrix), np.nanmax(hipermatrix)
        if vmin<0:
            vmax = max(-vmin, vmax)
            vmin = -vmax
        else:
            self.cmap = 'inferno'
        for ix in range(n_plots):
            ax[ix].imshow(hipermatrix[channels[ix]],cmap=self.cmap, origin='lower',
                                vmin=vmin, vmax=vmax, aspect='auto',
                                extent=[f_pha[0], f_pha[1], f_amp[0], f_amp[1]])
            ax[ix].set_xlabel(f'Frequency of {self.channels[channels[ix][0]]} providing phase (Hz)')
            ax[ix].set_ylabel(f'Frequency of {self.channels[channels[ix][1]]} receiving amplitude (Hz)')
            ax[ix].set_title(f'PAC connectivity')

    def display_band_connectivity(self, hipermatrix, f_pha=(2,20), f_amp=(15,120)):
        # Check which bands there are: 
        bands_phase = self.inverse_band_check(f_pha)
        bands_amp = self.inverse_band_check(f_amp)
        fig, ax = plt.subplots(nrows=len(bands_amp), ncols=len(bands_phase),
                                figsize=(6*len(bands_phase),6*len(bands_amp)))
        fpha_ref = np.linspace(f_pha[0], f_pha[1], hipermatrix.shape[-1])
        famp_ref = np.linspace(f_amp[0], f_amp[1], hipermatrix.shape[-2])
        bc = []
        for i, fas in enumerate(bands_phase):
            cosa = []
            for j, famp in enumerate(bands_amp):
                row = len(bands_amp)-1-j
                limits_phas = self.bands[fas]
                limits_amp = self.bands[famp]
                mask = np.outer((fpha_ref<=limits_phas[1])&(fpha_ref>limits_phas[0]),
                                (famp_ref<=limits_amp[1])&(famp_ref>limits_amp[0]))
                matrix = np.nanmean(hipermatrix[:,:,mask.T], axis=-1)
                ax[row, i].imshow(matrix.T, origin='lower')
                # el T es necesario para que la banda de origen sea el eje X
                if row==(len(bands_amp)-1):
                    ax[row, i].set_xlabel(fas)
                if i==0:
                    ax[row, i].set_ylabel(famp)
                ax[row, i].set_yticks(range(self.n_chan))
                ax[row, i].set_xticks(range(self.n_chan))
                ax[row, i].set_yticklabels(self.channels)
                ax[row, i].set_xticklabels(self.channels,rotation=60)
                cosa.append(matrix.T) 
            bc.append(cosa)
        return bc 
        
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

    def plot_connect_hipermatrix(self, hipermatrix, f_pha=(2,20), f_amp=(15,120), filename='untitled'):
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
                    ax[recezs<iving_amp, giving_phase].set_yticklabels([])
        #plt.tight_layout()
        plt.show()
        fig.savefig(f'{filename}.pdf')

#%% 
class MNE_connectivity(object):
    def __init__(self):
        import mne 
        from mne.channels.montage import get_builtin_montages
        montage = mne.channels.make_standard_montage('standard_1020')


    


#%% 

if __name__ == "__main__":
    path = '/home/pakitochus/Dropbox/Investigacion/Experimentos/202010_Functional_PAC/'
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, default=path)
    args = parser.parse_args()
    descriptors_path = args.input_folder

    data = np.load(os.path.join(path, 'quantile.npy'))

    th = 0.025
    data_sig = data.copy()
    data_sig[(data>th) & (data<1-th)] = np.nan

    plotter = BandConnectivityPlotter()

    mm = plotter.display_band_connectivity(data_sig, f_pha=(2,20), f_amp=(15,120))
    # %%
    plotter = BandConnectivityPlotter()
    plotter.plot_pac(data_sig, channels=[(15,0), (1,10)])

    #%%
    plotter.plot_connect_hipermatrix(data_sig, filename='prueba')
    # %%
