#%% 
from sqlalchemy import column
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
STIMULUS = 8

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
#%% 
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

bc_df.to_csv(f'bandconn_pac_s{STIMULUS}.csv')

# %% 
import pandas as pd
STIMULUS = 2
bc_df = pd.read_csv(f'bandconn_pac_s{STIMULUS}.csv', index_col=0)
bc_df['ID'] = bc_df.ID.astype(int).apply(lambda x: f'{x:03d}')
uniqsidclass = bc_df[['ID', 'class']].groupby(['ID','class']).size().reset_index().rename(columns={0:'count'})
equiv = dict(zip(uniqsidclass['ID'].values, uniqsidclass['class'].values))
bc_df = bc_df.groupby(['ID', 'pha_band', 'amp_band', 'Channel_0', 'Channel_1']).mean()

data_serv = pd.read_csv('/home/pakitochus/Universidad/Investigación/Databases/dislexia/db_perc_data_EEG.csv',
                index_col=0)
data_eeg = data_serv.loc[data_serv.year==7].groupby('ID').mean()
data_eeg = data_eeg.dropna(how='all', axis=1)
data_eeg['EEGSTR'] = data_eeg.EEGCODE.astype(int).apply(lambda x: f'{x:03d}')
data_eeg['DIAG'] = data_eeg['EEGSTR'].map(equiv)
data_eeg = data_eeg.dropna(subset=['DIAG'])

#%% GENERATE LAST MATRIX: 
band_def = bc_df.unstack().unstack().unstack().unstack().copy()
data_eeg = data_eeg.set_index('EEGSTR')
final = band_def.join(data_eeg)
# for id, new_df in band_def.groupby(level=0):
#     for variable in targets+covar:
#         band_def.loc[id, variable] = data_eeg.loc[data_eeg.EEGSTR==id,variable].values[0]

#%%
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.formula.api as smf


def get_r2_smforumla(x, y, k=1, data=None, confound=None):
    formula = f'{y} ~ {x}'
    if confound:
        formula = f'{formula} + {confound}'
    # model = smf.glm(formula, data=mezcla, family=sm.families.Poisson()).fit()
    model = smf.ols(formula, data=data).fit()
    return model.params[x], model.pvalues[x]#, model.conf_int().T[x].values, model.pvalues[confound]


def get_r2_statsmodels(x, y, k=1, confound=None, ixret=1):
    xpoly = np.column_stack([x**i for i in range(k+1)])   
    if confound is not None:
        xpoly = np.column_stack((xpoly,confound))
    model = sm.OLS(y, xpoly).fit()
    return model.params[ixret], model.pvalues[ixret], model.rsquared

CONFOUND = 'ISE' #'ISE' # 'Intervencion' #'ISE' # 'ISE'

prefix = ('value',)

targets = ['HFON_CF',
            'HFON_VELAL',
            'HFON_MVER',
            'LECT_COMPRE',
            'LECT_FLUI',
            'LECT_RF',
            'LECT_RV']
covar = ['ISE']

corr = np.zeros((band_def.shape[1],len(targets)))
p = np.zeros_like(corr)
r2 = np.zeros_like(corr)

# iterar por columnas    
for i, combinacion in enumerate(band_def.columns):
    for j, variable in enumerate(targets):    
        # filtrado = final[[combinacion,variable,CONFOUND]]
        # filtrado.dropna(inplace=True)
        filtrado = np.c_[final[combinacion].values, final[variable].values, final[CONFOUND].values]
        filtrado = filtrado[np.isnan(filtrado).any(axis=1)==False]
        # # _, _, corr[i,j], p[i,j], _ = linregress(filtrado[:,0], filtrado[:,1])
        corr[i,j], p[i,j],r2[i,j] = get_r2_statsmodels(filtrado[:,0], filtrado[:,1],#k=2,ixret=2,#, 
                                               confound=filtrado[:,2])
        # corr[i,j], p[i,j] = get_r2_smforumla(combinacion, variable, 
        #                                     data=final, confound=CONFOUND)

        # corr[i,j], p[i,j] = pearsonr(filtrado[:,0], filtrado[:,1])


# %%
from statsmodels.stats.multitest import multipletests

CORR='fdr_bh'

for CORR in ['hs', 'bonferroni','sidak','holm-sidak','holm','simes-hochberg','hommel','fdr_bh','fdr_by','fdr_tsbh','fdr_tsbky']:
    nancols = np.isnan(p).all(axis=0)
    p_rej, p_corrected, _, _ = multipletests(p[:,6], alpha=.05, method=CORR)
    # p_rej = p_rej.reshape(p.shape)
    # p_corrected = p_corrected.reshape(p.shape)
    print(f'Total with {CORR}: {p_rej.sum()}')
# %%
pval_df = pd.DataFrame(p[:,:2], index=band_def.columns, columns=targets[:2])
pval_df.loc[(pval_df<0.05).any(axis=1)]
# %%
