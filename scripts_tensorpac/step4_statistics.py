#%% 
import numpy as np
from scipy.stats import ttest_ind

from functionalpac.utils import load_subgroup, DEFAULT_BANDS
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

# %% REGRESSION 
from functionalpac.utils import PAC_object
import pandas as pd

bc_df = pd.DataFrame()
for ix, dats in enumerate(datos_ctl):
    pac0 = PAC_object(dats, channels=channel_n, fs=500, f_pha=(2,20), f_amp=(15,120))
    bc = pac0.extract_band_connectivity()
    for pha_band in bc.keys():
        for amp_band in bc[pha_band].keys():
            aux = pd.DataFrame(bc[pha_band][amp_band], index=channel_n, columns=channel_n)
            aux = aux.stack().reset_index()
            aux.columns = ['Channel_0', 'Channel_1', 'value']
            aux['pha_band'] = pha_band
            aux['amp_band'] = amp_band
            aux['ID'] = ids_ctl[ix]
            aux['class'] = 'CTL'
            bc_df = pd.concat((bc_df, aux))

for ix, dats in enumerate(datos_dlx):
    pac0 = PAC_object(dats, channels=channel_n, fs=500, f_pha=(2,20), f_amp=(15,120))
    bc = pac0.extract_band_connectivity()
    for pha_band in bc.keys():
        for amp_band in bc[pha_band].keys():
            aux = pd.DataFrame(bc[pha_band][amp_band], index=channel_n, columns=channel_n)
            aux = aux.stack().reset_index()
            aux.columns = ['Channel_0', 'Channel_1', 'value']
            aux['pha_band'] = pha_band
            aux['amp_band'] = amp_band
            aux['ID'] = ids_dlx[ix]
            aux['class'] = 'DLX'
            bc_df = pd.concat((bc_df, aux))
# bc_ctl = [PAC_object(el, channels=channel_n, fs=500, f_pha=(2,20), f_amp=(15,120)).extract_band_connectivity() for el in datos_ctl]
# bc_dlx = [PAC_object(el, channels=channel_n, fs=500, f_pha=(2,20), f_amp=(15,120)).extract_band_connectivity() for el in datos_dlx]

# %%
