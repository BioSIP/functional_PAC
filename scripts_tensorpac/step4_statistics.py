import numpy as np
import os
import matplotlib.pyplot as plt 
import re
from scipy.ndimage import label
from joblib import Parallel, delayed

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

descriptor_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/tensorpac'


filenames = [fi for fi in os.listdir(descriptor_path) if fi.endswith('npy')]


filename = filenames[0]

def load_subgroup(path, stimulus=2, group='C', age=7):
    filenames = [fi for fi in os.listdir(descriptor_path) if fi.endswith('npy')]
    r = re.compile(f'\d*_{age}_{stimulus}_{group}.npy')
    filenames_r = list(filter(r.match, filenames))
    N = len(filenames_r)
    data = np.zeros((N, len(channel_n), len(channel_n), 52, 50))
    for i,fil in enumerate(filenames_r):
        data[i] = np.load(os.path.join(path, fil))
    return data

datos_ctl = load_subgroup(descriptor_path, group='C', stimulus=2)

datos_dlx = load_subgroup(descriptor_path, group='RD', stimulus=2)


def plot_specific(hipermatriz, ch_x, ch_y):
    _, ax  = plt.subplots(figsize=(5,4))
    f = ax.imshow(hipermatriz[ch_x,ch_y], extent=[2, 20,15, 120], aspect='auto')
    ax.set_title(f'Channels {ch_x}, {ch_y}')
    ax.set_xlabel('Frequency for Phase (Hz)')
    ax.set_ylabel('Frequency for Amplitude (Hz)')
    plt.colorbar(f)


#%% 
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.ndimage import label

class PermutationTesting(object):

    def __init__(self, n_iter=200, alpha = 0.05, 
                test=ttest_ind, twosided=False, n_jobs=10):
        self.test = test
        self.n_iter = n_iter
        self.alpha = alpha
        self.twosided = twosided
        self.n_jobs = n_jobs

    def permutation_testing(self, X, y, ref, arr):
        y_surr = np.random.permutation(y)
        stat_surr, _ = self.test(X[y_surr==0], 
                                X[y_surr==1])
        # gets quantile 
        arr += ref>stat_surr

    def fit(self, X, y):
        # compute stats 
        stat, _ = self.test(X[y==0], X[y==1])
        # sets thresholds for umbralization
        if self.twosided:
            th = self.alpha/2
        #     th_l, th_u = np.quantile(stat, th), np.quantile(stat, 1-th)
        # else: 
        #     th_l, th_u = -np.inf, np.quantile(stat, 1-self.alpha)
        pval_arr = np.zeros(stat.shape)
        # go channel per channel: 
        for ch_x in range(stat.shape[0]):
            for ch_y in range(stat.shape[1]):
                with Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=20) as parallel:
                    parallel(
                        delayed(self.permutation_testing)(X[:,ch_x,ch_y], y, stat[ch_x,ch_y], pval_arr[ch_x,ch_y]) for i in range(self.n_iter))
        
        pval_arr = pval_arr/self.n_iter
        if self.twosided: 
            pval_arr[(pval_arr>=th)&(pval_arr<=1-th)] = np.nan
            pval_arr[pval_arr<th] = -pval_arr[pval_arr<th]
            pval_arr[pval_arr>1-th] = 1-pval_arr[pval_arr>1-th]
        else:
            pval_arr[pval_arr<1-self.alpha] = np.nan
            pval_arr[pval_arr>1-self.alpha] = 1-pval_arr[pval_arr>1-th]
        return pval_arr 


class ClusterBasedPermutationTesting(PermutationTesting):

    def __init__(self, n_iter=200, alpha = 0.05, cluster_threshold = 20, 
                test=ttest_ind, twosided=False, n_jobs=10, output='pval'):
        PermutationTesting.__init__(self, n_iter, alpha, test, twosided, n_jobs)
        self.cluster_threshold = cluster_threshold
        self.output = output

    def permutation_testing(self, X,y, cluster_stat, arr, select):
        y_surr = np.random.permutation(y)
        stat_surr, _ = self.test(X[y_surr==0], 
                                X[y_surr==1])
        cs_surr = stat_surr[select].sum()
        # Compute quantile. 
        if cluster_stat > cs_surr:
            arr[select] += 1

    def fit(self, X, y):
        """ Performs the Cluster-Based Permutation Testing for
        corrections. 
        It does: 
        1. Computes the hypothesis testing between groups. 
        2. Sets the test threshold. 
        3. Cluster selection. 
        4. Cluster_based statistic computation.
        5. Permutation testing and significance estimation. 

        Args:
            X (numpy.ndarray): [description]
            y (numpy.ndarray): [description]
        """
        # compute stats 
        stat, _ = self.test(X[y==0], X[y==1])
        # sets thresholds for umbralization
        if self.twosided:
            th = self.alpha/2
            th_l, th_u = np.nanquantile(stat, th), np.nanquantile(stat, 1-th)
        else: 
            th_l, th_u = -np.inf, np.nanquantile(stat, 1-self.alpha)
        # threshold the stat array 
        bin_stat = (stat < th_l) | (stat > th_u)
        # label connected clusters
        label_arr = np.zeros(stat.shape)
        pval_arr = np.empty(stat.shape)*np.nan
        for ch_x in range(stat.shape[0]):
            for ch_y in range(stat.shape[1]):
                arr, n_labs = label(bin_stat[ch_x, ch_y])
                # count elements per cluster
                count, idx = np.histogram(arr.flatten(), bins=n_labs+1)
                idx = idx.astype(int)[:-1] 
                # keep only supra-threshold clusters: 
                idx_sel = idx[count>self.cluster_threshold]+1
                if len(idx_sel>1):
                    for ix in idx_sel[1:]:
                        # save regions:
                        select = arr==ix
                        cluster_lab = np.zeros(arr.shape)
                        cluster_lab[select] = ix
                        pval_arr[ch_x, ch_y][select] = 0
                        label_arr[ch_x, ch_y] += cluster_lab
                        # compute cluster-based statistic: 
                        cluster_stat = stat[ch_x, ch_y][select].sum()
                        with Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=20) as parallel:
                            parallel(
                                delayed(self.permutation_testing)(X[:,ch_x,ch_y], y, cluster_stat, pval_arr[ch_x,ch_y], select) for i in range(self.n_iter))
                        # for _ in range(self.n_iter):
                        #     y_surr = np.random.permutation(y)
                        #     stat_surr, _ = self.test(X[y_surr==0,ch_x,ch_y], 
                        #                             X[y_surr==1,ch_x,ch_y])
                        #     cs_surr = stat_surr[select].sum()
                        #     # Compute quantile. 
                        #     if cluster_stat > cs_surr:
                        #         pval_arr[ch_x, ch_y][select] += 1
        pval_arr = pval_arr/self.n_iter
        if output=='pval':
            if self.twosided: 
                pval_arr[(pval_arr>=th)&(pval_arr<=1-th)] = np.nan
                pval_arr[pval_arr<th] = -pval_arr[pval_arr<th]
                pval_arr[pval_arr>1-th] = 1-pval_arr[pval_arr>1-th]
            else:
                pval_arr[pval_arr<1-self.alpha] = np.nan
                pval_arr[pval_arr>1-self.alpha] = 1-pval_arr[pval_arr>1-th]
        return pval_arr 

#%% 
X = np.r_[datos_ctl, datos_dlx]
y = np.r_[np.ones(datos_ctl.shape[0]), np.zeros(datos_dlx.shape[0])]

#%% 
cbpt = ClusterBasedPermutationTesting(n_iter=1000, twosided=True, 
                cluster_threshold = 10, test=ttest_ind, n_jobs=25, output='quantile')
pvalu = cbpt.fit(X,y)

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
                            'Gamma': (25, 140)}
        if cmap=='coldhot':
            self.cmap = LinearSegmentedColormap('coldhot', _pigtailed_cmap(cm.hot))
        else:
            self.cmap = cm.get_cmap(cmap)

    def check_bands(self, f):
        bands = []
        for band in self.bands:
            if f>self.bands[band][0] and f<self.bands[band][1]:
                bands.append(band)
        return bands

    def inverse_band_check(self, f_range):
        bands = set()
        for f in range(f_range[0], f_range[1], 2):
            bands.add(self.check_bands(f))
        return list(bands)

    def display_band_connectivity(self, hipermatrix, f_pha=(2,20), f_amp=(15,120)):
        # Check which bands there are: 
        bands_phase = self.inverse_band_check(f_pha)
        bands_amp = self.inverse_band_check(f_amp)
        fig, ax = plt.subplots(nrows=len(bands_amp), ncols=len(bands_phase))
        fpha_ref = np.linspace(f_pha[0], f_pha[1], hipermatrix.shape[-2])
        famp_ref = np.linspace(f_amp[0], f_amp[1], hipermatrix.shape[-1])
        for fas in range(len(bands_phase)):
            for famp in range(len(bands_amp)):
                #TODO
                famp 
                matrix = hipermatrix[:,:,]
                ax[famp, fas].imshow()
        
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
        for i in range(n_chan):
            for j in range(n_chan):
                matris = hipermatrix[i,j]
                matris[np.isnan(matris)] = 0.
                ax[i,j].imshow(matris, cmap=self.cmap, origin='lower',
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


#%% A PARTIR DE AQUÍ TESTING. 
test = ttest_ind
n_iter=1000
twosided=True
alpha =0.05
cluster_threshold=10
n_jobs = 10

def permutation_testing(test, X,y, cluster_stat, arr, select):
    y_surr = np.random.permutation(y)
    stat_surr, _ = test(X[y_surr==0], 
                            X[y_surr==1])
    cs_surr = stat_surr[select].sum()
    # Compute quantile. 
    if cluster_stat > cs_surr:
        arr[select] += 1

stat, _ = test(X[y==0], X[y==1])
# sets thresholds for umbralization
if twosided:
    th = alpha/2
    th_l, th_u = np.nanquantile(stat, th), np.nanquantile(stat, 1-th)
else: 
    th_l, th_u = -np.inf, np.nanquantile(stat, 1-alpha)
# threshold the stat array 
bin_stat = (stat < th_l) | (stat > th_u)
# label connected clusters
label_arr = np.zeros(stat.shape)
pval_arr = np.empty(stat.shape)*np.nan

ch_x, ch_y = 0,3

arr, n_labs = label(bin_stat[ch_x, ch_y])
# count elements per cluster
count, idx = np.histogram(arr.flatten(), bins=n_labs+1)
idx = idx.astype(int)[:-1] 
# keep only supra-threshold clusters: 
idx_sel = idx[count>cluster_threshold]+1
print(idx_sel)
if len(idx_sel>1):
    for ix in idx_sel[1:]:
        # save regions:
        select = arr==ix
        cluster_lab = np.zeros(arr.shape)
        cluster_lab[select] = ix
        pval_arr[ch_x, ch_y][select] = 0
        label_arr[ch_x, ch_y] += cluster_lab
        # compute cluster-based statistic: 
        cluster_stat = stat[ch_x, ch_y][select].sum()
        with Parallel(n_jobs=n_jobs, prefer="threads", verbose=20) as parallel:
            parallel(
                delayed(permutation_testing)(test, X[:,ch_x,ch_y], y, cluster_stat, pval_arr[ch_x,ch_y], select) for i in range(n_iter))

print(np.nanmax(pval_arr/n_iter))
#%%
pval_arr = pval_arr/n_iter
if twosided: 
    pval_arr[(pval_arr>=th)&(pval_arr<=1-th)] = np.nan
    pval_arr[pval_arr<th] = -pval_arr[pval_arr<th]
    pval_arr[pval_arr>1-th] = 1-pval_arr[pval_arr>1-th]
else:
    pval_arr[pval_arr<1-alpha] = np.nan
    pval_arr[pval_arr>1-alpha] = 1-pval_arr[pval_arr>1-th]


# plot_specific(bin_stat*np.sign(stat), ch_x, ch_y)
# cosa = pval_arr/n_iter
# plot_specific(cosa, ch_x, ch_y)


# # %%
# ch_x,ch_y = 30,23
# plt.imshow(pval[ch_x, ch_y])
# plt.colorbar()
# # %%
# def plot_avg_connectivity(hipermatriz):
#     _, ax  = plt.subplots(figsize=(7,7))
#     if np.nanmin(hipermatriz)<0:
#         vmax = np.nanmax(abs(hipermatriz))
#         vmin = -vmax
#         colormap = 'coolwarm'
#     else:
#         vmin, vmax = 0, np.nanmax(hipermatriz)
#         colormap = 'inferno'
#     val = np.nanmean(np.nanmean(hipermatriz, axis=-1), axis=-1)
#     f = ax.matshow(val, vmin=vmin, vmax=vmax, cmap= colormap)
#     ax.set_yticks(range(31))
#     ax.set_xticks(range(31))
#     ax.set_yticklabels(channel_n)
#     ax.set_xticklabels(channel_n,rotation=60)
#     plt.colorbar(f)

# plot_avg_connectivity((1-abs(pval))*np.sign(pval))

# #%%

# test = ttest_ind
# n_iter=1000
# n_jobs=10
# twosided=True
# alpha =0.05
# cluster_threshold=20

# def permutation_testing(test, X, y, ref, arr):
#     y_surr = np.random.permutation(y)
#     stat_surr, _ = test(X[y_surr==0], 
#                             X[y_surr==1])
#     # gets quantile 
#     arr += ref>stat_surr

# # compute stats 
# stat, _ = test(X[y==0], X[y==1])
# # sets thresholds for umbralization
# if twosided:
#     th = alpha/2
# #     th_l, th_u = np.quantile(stat, th), np.quantile(stat, 1-th)
# # else: 
# #     th_l, th_u = -np.inf, np.quantile(stat, 1-self.alpha)
# pval_arr = np.zeros(stat.shape)
# # go channel per channel: 
# # for ch_x in range(stat.shape[0]):
# #     for ch_y in range(stat.shape[1]):
# ch_x, ch_y = 0, 0
# with Parallel(n_jobs=n_jobs, prefer="threads", verbose=20) as parallel:
#     parallel(
#         delayed(permutation_testing)(test, X[:,ch_x,ch_y], y, stat[ch_x,ch_y], pval_arr[ch_x,ch_y]) for i in range(n_iter))

# pval_arr = pval_arr/n_iter
# if twosided: 
#     pval_arr[(pval_arr>=th)&(pval_arr<=1-th)] = np.nan
#     pval_arr[pval_arr<th] = -pval_arr[pval_arr<th]
#     pval_arr[pval_arr>1-th] = 1-pval_arr[pval_arr>1-th]
# else:
#     pval_arr[pval_arr<1-alpha] = np.nan
#     pval_arr[pval_arr>1-alpha] = 1-pval_arr[pval_arr>1-th]

# plot_avg_connectivity((1-abs(pval))*np.sign(pval))