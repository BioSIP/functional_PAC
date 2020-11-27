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
        pval_arr = np.empty(stat.shape)*np.nan
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

    def __init__(self, n_iter=200, significance = 0.05, cluster_threshold = 20, 
                test=ttest_ind, twosided=False, n_jobs=10):
        PermutationTesting.__init__(self, n_iter, alpha, test, twosided, n_jobs)
        self.cluster_threshold = cluster_threshold

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
            th_l, th_u = np.quantile(stat, th), np.quantile(stat, 1-th)
        else: 
            th_l, th_u = -np.inf, np.quantile(stat, 1-self.alpha)
        # threshold the stat array 
        bin_stat = (stat < th_l) | (stat > th_u)
        # label connected clusters
        label_arr = np.zeros(bin_stat.shape)
        pval_arr = np.empty(bin_stat.shape)*np.nan
        for ch_x in range(label_arr.shape[0]):
            for ch_y in range(label_arr.shape[1]):
                arr, n_labs = label(bin_stat[ch_x, ch_y])
                # count elements per cluster
                count, idx = np.histogram(arr.flatten(), bins=n_labs+1)
                idx = idx.astype(int)[:-1] 
                # keep only supra-threshold clusters: 
                idx_sel = idx[count>self.cluster_threshold]+1
                if len(idx_sel>1):
                    for ix in idx_sel[1:]:
                        # save regions:
                        cluster_lab = np.zeros(arr.shape)
                        cluster_lab[arr==ix] = ix
                        pval_arr[ch_x, ch_y][arr==ix] = 0
                        label_arr[ch_x, ch_y] += cluster_lab
                        # compute cluster-based statistic: 
                        cluster_stat = stat[ch_x, ch_y][arr==ix].sum()
                        with Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=20) as parallel:
                            parallel(
                                delayed(self.permutation_testing)(X[:,ch_x,ch_y], y, cluster_stat, pval_arr[ch_x,ch_y], arr==ix) for i in range(self.n_iter))
                        # for _ in range(self.n_iter):
                        #     y_surr = np.random.permutation(y)
                        #     stat_surr, _ = self.test(X[y_surr==0,ch_x,ch_y], 
                        #                             X[y_surr==1,ch_x,ch_y])
                        #     cs_surr = stat_surr[arr==ix].sum()
                        #     # Compute quantile. 
                        #     if cluster_stat > cs_surr:
                        #         pval_arr[ch_x, ch_y][arr==ix] += 1
        pval_arr = pval_arr/self.n_iter
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
                cluster_threshold = 10, test=ttest_ind, n_jobs=20)
pvalu = cbpt.fit(X,y)
#%%
test = ttest_ind
n_iter=5000
twosided=True
significance =0.05
cluster_threshold=20


def permutation_testing(X,y, cluster_stat, arr, select, test):
    y_surr = np.random.permutation(y)
    stat_surr, _ = test(X[y_surr==0], 
                            X[y_surr==1])
    cs_surr = stat_surr[select].sum()
    # Compute quantile. 
    if cluster_stat > cs_surr:
        arr[select] += 1

stat, _ = test(X[y==0], X[y==1])
if twosided:
    th = significance/2
    th_l, th_u = np.quantile(stat, th), np.quantile(stat, 1-th)
else: 
    th_l, th_u = -np.inf, np.quantile(stat, 1-significance)

bin_stat = (stat < th_l) | (stat > th_u)
label_arr = np.zeros(bin_stat.shape)
pval_arr = np.zeros(bin_stat.shape)*np.nan

ch_x, ch_y = 0,0
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
        cluster_lab = np.zeros(arr.shape)
        cluster_lab[arr==ix] = ix
        pval_arr[ch_x,ch_y][arr==ix] = 0
        label_arr[ch_x, ch_y] += cluster_lab
        # compute cluster-based statistic: 
        cluster_stat = stat[ch_x, ch_y][arr==ix].sum()
        print(cluster_stat)

        with Parallel(n_jobs=10, prefer="threads", verbose=20) as parallel:
            parallel(
                delayed(permutation_testing)(X[:,ch_x,ch_y], y, cluster_stat, pval_arr[ch_x,ch_y], arr==ix, test=test) for i in range(n_iter))

plot_specific(bin_stat*np.sign(stat), ch_x, ch_y)
cosa = pval_arr/n_iter
plot_specific(cosa, ch_x, ch_y)


# %%
ch_x,ch_y = 30,23
plt.imshow(pval[ch_x, ch_y])
plt.colorbar()
# %%
def plot_avg_connectivity(hipermatriz):
    _, ax  = plt.subplots(figsize=(7,7))
    if np.nanmin(hipermatriz)<0:
        vmax = np.nanmax(abs(hipermatriz))
        vmin = -vmax
        colormap = 'coolwarm'
    else:
        vmin, vmax = 0, np.nanmax(hipermatriz)
        colormap = 'inferno'
    val = np.nanmean(np.nanmean(hipermatriz, axis=-1), axis=-1)
    f = ax.matshow(val, vmin=vmin, vmax=vmax, cmap= colormap)
    ax.set_yticks(range(31))
    ax.set_xticks(range(31))
    ax.set_yticklabels(channel_n)
    ax.set_xticklabels(channel_n,rotation=60)
    plt.colorbar(f)

plot_avg_connectivity((1-abs(pval))*np.sign(pval))

#%%
