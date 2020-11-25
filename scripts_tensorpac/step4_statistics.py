import numpy as np
import os
import matplotlib.pyplot as plt 
import re
from scipy.ndimage import label


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

datos_dlx = load_subgroup(descriptor_path, group='RD', stimulus=20)



from scipy.stats import ttest_ind
from scipy.ndimage import label

class ClusterBasedPermutationTesting(object):

    def __init__(self, n_iter=200, significance = 0.05, cluster_threshold = 20, 
                test=ttest_ind, twosided=False):
        self.test = test
        self.n_iter = n_iter
        self.significance = significance
        self.cluster_threshold = cluster_threshold
        self.twosided = twosided


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
            th = self.significance/2
            th_l, th_u = np.quantile(stat, th), np.quantile(stat, 1-th)
        else: 
            th_l, th_u = -np.inf, np.quantile(stat, 1-self.significance)
        # threshold the stat array 
        bin_stat = (stat < th_l) | (stat > th_u)
        # label connected clusters
        label_arr = np.zeros(bin_stat.shape)
        pval_arr = np.zeros(bin_stat.shape)
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
                        label_arr[ch_x, ch_y] += cluster_lab
                        # compute cluster-based statistic: 
                        cluster_stat = stat[ch_x, ch_y][arr==ix].sum()
                        for _ in range(self.n_iter):
                            y_surr = np.random.permutation(y)
                            stat_surr, _ = self.test(X[y_surr==0,ch_x,ch_y], 
                                                    X[y_surr==1,ch_x,ch_y])
                            cs_surr = stat_surr[arr==ix].sum()
                            if cs_surr>cluster_stat:
                                pval_arr[ch_x, ch_y][arr==ix] += 1
                            
        pval_arr = pval_arr/self.n_iter
        return pval_arr 

#%% 
X = np.r_[datos_ctl, datos_dlx]
y = np.r_[np.ones(datos_ctl.shape[0]), np.zeros(datos_dlx.shape[0])]

#%% 
cbpt = ClusterBasedPermutationTesting(twosided=True)
pval = cbpt.fit(X,y)
#%%
label_arr = np.zeros(bin_stat.shape)
pval_arr = np.zeros(bin_stat.shape)
n_iter=1000


ch_x, ch_y = 0, 7
arr, n_labs = label(bin_stat[ch_x, ch_y])
# count elements per cluster
count, idx = np.histogram(arr.flatten(), bins=n_labs+1)
idx = idx.astype(int)[:-1] 
# keep only supra-threshold clusters: 
idx_sel = idx[count>20]+1
plt.imshow(arr)
# %%
if len(idx_sel>1):
    for ix in idx_sel[1:]:
        # save regions:
        cluster_lab = np.zeros(arr.shape)
        cluster_lab[arr==ix] = ix
        label_arr[ch_x, ch_y] += cluster_lab
        # compute cluster-based statistic: 
        cluster_stat = stat[ch_x, ch_y][arr==ix].sum()
        for _ in range(n_iter):
            y_surr = np.random.permutation(y)
            stat_surr, _ = ttest_ind(X[y_surr==0,ch_x,ch_y], 
                                    X[y_surr==1,ch_x,ch_y])
            cs_surr = stat_surr[arr==ix].sum()
            if cs_surr>cluster_stat:
                pval_arr[ch_x, ch_y][arr==ix] += 1
# %%
ch_x,ch_y = 30,23
plt.imshow(pval[ch_x, ch_y])
plt.colorbar()
# %%
