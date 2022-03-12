#%% 
import numpy as np
from scipy.stats import ttest_ind

from functionalpac.utils import load_subgroup
from functionalpac.statistics import ClusterBasedPermutationTesting

global bands
global channel_n
bands = {'Delta': (0.5, 4),
         'Theta': (4, 8),
         'Alpha': (8, 12),
         'Beta': (12, 25),
         'Gamma': (25, 140)}

ct = 30
test = ttest_ind
n_iter = 10000
alpha = .05
njobs = 10
th_pac = 0.975
STIMULUS = 2

channel_n = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
             'C3', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
             'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

# ctl_path = '/media/NAS/home/marco/projects/PAC_Paper/functional_PAC/data/tensorpac'
# dlx_path = '/media/NAS/home/marco/projects/PAC_Paper/functional_PAC/data/tensorpac2'
ctl_path = '/home/pakitochus/Universidad/Investigación/Databases/dislexia/functional_PAC/data/tensorpac3'
dlx_path = '/home/pakitochus/Universidad/Investigación/Databases/dislexia/functional_PAC/data/tensorpac3'
datos_ctl, ids_ctl = load_subgroup(ctl_path, group='C', stimulus=STIMULUS,
                          purge=True, nan=0, channels=channel_n)
datos_dlx, ids_dlx = load_subgroup(dlx_path, group='RD',
                          stimulus=STIMULUS, purge=True, nan=0, channels=channel_n)

X = np.r_[datos_ctl, datos_dlx]
y = np.r_[np.ones(datos_ctl.shape[0]), np.zeros(datos_dlx.shape[0])]
cbpt = ClusterBasedPermutationTesting(n_iter=n_iter, twosided=True,
                                      cluster_threshold=ct, test=test, n_jobs=njobs, output='both',
                                      threshold_pac=th_pac)
pvalu, quantile = cbpt.fit(X, y)

np.save(f'pvalue_p1e4_ct{ct}_s{STIMULUS}.npy', pvalu)
np.save(f'quantile_p1e4_ct{ct}_s{STIMULUS}.npy', quantile)

# %%
