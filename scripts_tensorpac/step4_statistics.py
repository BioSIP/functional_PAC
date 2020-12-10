import numpy as np
import os
import matplotlib.pyplot as plt 
import re
from scipy.ndimage import label
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from scipy.ndimage import label
from scipy.stats import rankdata

def load_subgroup(path, stimulus=2, group='C', age=7, purge=False):
    filenames = [fi for fi in os.listdir(descriptor_path) if fi.endswith('npy')]
    r = re.compile(f'\d*_{age}_{stimulus}_{group}.npy')
    filenames_r = list(filter(r.match, filenames))
    N = len(filenames_r)
    data = np.zeros((N, len(channel_n), len(channel_n), 52, 50))
    for i,fil in enumerate(filenames_r):
        if purge: 
            pac = PAC_object(np.load(os.path.join(path, fil)), channel_n)
            pac.purge_decomposition()
            defin = pac.purged_mat
        else:
            defin = np.load(os.path.join(path, fil))
        data[i] = defin
    return data

def plot_specific(hipermatriz, ch_x, ch_y):
    _, ax  = plt.subplots(figsize=(5,4))
    f = ax.imshow(hipermatriz[ch_x,ch_y], extent=[2, 20,15, 120], aspect='auto')
    ax.set_title(f'Channels {ch_x}, {ch_y}')
    ax.set_xlabel('Frequency for Phase (Hz)')
    ax.set_ylabel('Frequency for Amplitude (Hz)')
    plt.colorbar(f)


from sklearn.linear_model import LinearRegression

class PAC_object(object):
    def __init__(self, pac_hiper, channels, fs=500): 
        self.pac_hiper = pac_hiper
        if channels == None:
            self.channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                             'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                             'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                             'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']
        else:
            self.channels = channels
        self.n_chan = len(self.channels)
        self.pac_shape = self.pac_hiper.shape[2:]
        self.fs = fs

    def purge_decomposition(self):
        self.purged_mat = np.zeros(self.pac_hiper.shape)
        for i in range(self.n_chan):
            for j in range(i, self.n_chan):
                if i!=j:
                    x = np.hstack((self.pac_hiper[i,i].reshape((-1,1)), self.pac_hiper[j,j].reshape(-1,1)))
                    for z in range(2):
                        vecs = [i,j]
                        model = LinearRegression()
                        y = self.pac_hiper[vecs[z],vecs[1-z]].reshape((-1,1))
                        model.fit(x,y)
                        coefs = model.coef_
                        fin = y-x.dot(coefs.T)
                        self.purged_mat[vecs[z],vecs[1-z]] = fin.reshape(self.pac_shape)
                else:
                    self.purged_mat[i,j] = self.pac_hiper[i,j]


# %%
# from sklearn.decomposition import PCA, SparsePCA
# pac_hiper = datos_ctl[0]
# n_chan = pac_hiper.shape[0]
# pca_reg = PCA(n_components=n_chan)#, svd_solver='randomized') 
# sc_reg = pca_reg.fit_transform(pac_hiper.reshape((n_chan**2, -1)))
# sc_reg = sc_reg.reshape([n_chan, n_chan, -1])
# sc_diag = sc_reg[np.diag_indices(31)]




#%% 

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
        assert output in ['both', 'pval', 'quantile']
        self.output = output

    def mwu_vectorized(self, x1,x2):
        """Computes the two-sided Mann-Whitney-Wilcoxon U-Test
        over whole matrices (unlike the scipy version).
        Use reshape to reduce any matrix to two-dimensional. 

        Args:
            x1 (np.ndarray): two-dimensional array with features of group 0
            x2 (np.ndarray): two-dimensional array with features of group 1

        Returns:
            np.ndarray: Array with the two-tailed values for U
        """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        n1 = len(x1)
        n2 = len(x2)
        
        ranked = rankdata(np.concatenate((x1, x2)), axis=0)
        rank1 = ranked[0:n1] # gets x-ranks
        u1 = n1*n2 + (n1*(n1+1))/2.0 - np.sum(rank1, axis=0) # calc U for 1
        u2 = n1*n2 - u1 # remainder is U for y
        bigu = np.max(np.c_[u1,u2],axis=1)
        return bigu

    def matrix_testing(self, X, y, axis=0):
        """Implement the statistical test in any given size matrix. 

        Args:
            X ([type]): [description]
            y ([type]): [description]
            axis (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """
        # axis: where the statistic is computed upon. 
        dims = X.shape[1:]
        X_test = X.reshape((X.shape[0], np.prod(dims)))
        if self.test=='mannwhitneyu':
            stats = self.mwu_vectorized(X_test[y==0], X_test[y==1])
        elif callable(self.test):
            stats,_ = self.test(X_test[y==0], X_test[y==1])
        stats = stats.reshape(dims)
        return stats

    def permutation_testing(self, X,y, cluster_stat, arr, select):
        y_surr = np.random.permutation(y)
        stat_surr = self.matrix_testing(X, y_surr)
        cs_surr = stat_surr[select].sum()
        # Compute quantile. 
        if cluster_stat > cs_surr:
            arr[select] += 1

    def supra_threshold(self, arr):
        # count elements per cluster
        n_labs = len(np.unique(arr))
        count, idx = np.histogram(arr.flatten(), bins=n_labs+1)
        idx = idx.astype(int)[:-1] 
        return idx[count>self.cluster_threshold]+1

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
        stat = self.matrix_testing(X, y)
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
        for ch_pha in range(stat.shape[0]):
            for ch_amp in range(stat.shape[1]):
                arr, n_labs = label(bin_stat[ch_pha, ch_amp])
                # keep only supra-threshold clusters: 
                idx_sel = self.supra_threshold(arr)
                if len(idx_sel>1):
                    for ix in idx_sel[1:]:
                        # save regions:
                        select = arr==ix
                        cluster_lab = np.zeros(arr.shape)
                        cluster_lab[select] = ix
                        pval_arr[ch_pha, ch_amp][select] = 0
                        label_arr[ch_pha, ch_amp] += cluster_lab
                        # compute cluster-based statistic: 
                        cluster_stat = stat[ch_pha, ch_amp][select].sum()
                        with Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=20) as parallel:
                            parallel(
                                delayed(self.permutation_testing)(X[:,ch_pha,ch_amp], y, cluster_stat, pval_arr[ch_pha,ch_amp], select) for i in range(self.n_iter))

        pval_arr = pval_arr/self.n_iter
        if self.output=='quantile' or self.output=='both':
            quantile_arr = pval_arr.copy()
        if self.output=='pval' or self.output=='both':
            if self.twosided: 
                pval_arr[(pval_arr>=th)&(pval_arr<=1-th)] = np.nan
                pval_arr[pval_arr<th] = -pval_arr[pval_arr<th]
                pval_arr[pval_arr>1-th] = 1-pval_arr[pval_arr>1-th]
            else:
                pval_arr[pval_arr<1-self.alpha] = np.nan
                pval_arr[pval_arr>1-self.alpha] = 1-pval_arr[pval_arr>1-th]
        if self.output=='both':
            pval_arr = (pval_arr, quantile_arr)
        return pval_arr 

#%% 

global bands
global channel_n
bands = {'Delta': (0.5, 4),
         'Theta': (4, 8),
         'Alpha': (8, 12),
         'Beta': (12, 25),
         'Gamma': (25, 140)}

ct=10
test = ttest_ind
n_iter = 1000
alpha = .05
njobs = 10


channel_n = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
                  'C3', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                  'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

descriptor_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/tensorpac'

datos_ctl = load_subgroup(descriptor_path, group='C', stimulus=2, purge=True)
datos_dlx = load_subgroup(descriptor_path, group='RD', stimulus=2, purge=True)

X = np.r_[datos_ctl, datos_dlx]
y = np.r_[np.ones(datos_ctl.shape[0]), np.zeros(datos_dlx.shape[0])]

cbpt = ClusterBasedPermutationTesting(n_iter=n_iter, twosided=True, 
                cluster_threshold = ct, test=test, n_jobs=njobs, output='both')
pvalu, quantile = cbpt.fit(X,y)

np.save('pvalue_purged.npy', pvalu)
np.save('quantile_purged.npy', quantile)


