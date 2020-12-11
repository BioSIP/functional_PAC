import numpy as np
from statistics import load_subgroup, ClusterBasedPermutationTesting
from scipy.stats import ttest_ind
#%% 

global bands
global channel_n
bands = {'Delta': (0.5, 4),
         'Theta': (4, 8),
         'Alpha': (8, 12),
         'Beta': (12, 25),
         'Gamma': (25, 140)}

ct=20
test = ttest_ind
n_iter = 100
alpha = .05
njobs = 10


channel_n = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
                  'C3', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                  'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']


ctl_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/tensorpac'
datos_ctl = load_subgroup(ctl_path, group='C', stimulus=2, purge=True, nan=0, channels=channel_n)
datos_dlx = load_subgroup(ctl_path, group='RD', stimulus=2, purge=True, nan=0,channels=channel_n)

X = np.r_[datos_ctl, datos_dlx]
y = np.r_[np.ones(datos_ctl.shape[0]), np.zeros(datos_dlx.shape[0])]

cbpt = ClusterBasedPermutationTesting(n_iter=n_iter, twosided=True, 
                cluster_threshold = ct, test=test, n_jobs=njobs, output='both',
                threshold_pac=0.9)
pvalu, quantile = cbpt.fit(X,y)

np.save('pvalue_purged.npy', pvalu)
np.save('quantile_purged.npy', quantile)


