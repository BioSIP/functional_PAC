# %%
import pickle
import os
import numpy as np
from functionalpac.visualization import BandConnectivityPlotter
from functionalpac.visualization import EEGarray, Head
from functionalpac.utils import PAC_connect, DEFAULT_BANDS
from functionalpac.visualization import plot_head_connectivity
import matplotlib.pyplot as plt

electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                            'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']
path = '/home/pakitochus/Investigacion/Experimentos/202010_Functional_PAC/results/defs/'

# for ct in [10, 20, 30]:
    # for stimulus in [2, 8]:
ct = 30
stimulus = 20
formato='pdf'
data = np.load(os.path.join(path, f's{stimulus}/p1e4_ct{ct}/quantile_p1e4_ct{ct}_s{stimulus}.npy'))

# sets the alpha threshold: 
th = 0.025 #twosided
data_sig = data.copy()
data_sig[(data > th) & (data < 1-th)] = np.nan
data_sig = data_sig-0.5

hey = PAC_connect(data_sig, electrodes)
fig_route = os.path.join(path, f's{stimulus}/p1e4_ct{ct}/head_connect_ct{ct}_s{stimulus}.{formato}')
plot_head_connectivity(hey, bands=DEFAULT_BANDS, show_topo=True, show_electrodes=True,
                        show_head=True, show_connect=True, projection='lambert',
                        savefig=fig_route, title=f'PAC Connectivity (CT:{ct}, Stim: {stimulus}Hz)')

#%% 

plotter = BandConnectivityPlotter()

mm = plotter.display_band_connectivity(
    data_sig, f_pha=(2, 20), f_amp=(15, 120))
# %%
plotter = BandConnectivityPlotter()
plotter.plot_pac(data_sig, channels=[(15, 0), (1, 10)])

# %%
plotter.plot_connect_hipermatrix(data_sig, filename='quantile_p1e4_ct30_s8')

# %%

electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                            'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

adjacenty = np.random.randn(len(electrodes), len(electrodes))
adjacenty[np.abs(adjacenty) < 1.5] = np.nan
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(adjacenty)
ax.set_yticks(range(len(electrodes)))
ax.set_xticks(range(len(electrodes)))
ax.set_yticklabels(electrodes)
ax.set_xticklabels(electrodes, rotation=60)
# %%

electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                            'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

path = '/home/pakitochus/Dropbox/Investigacion/Experimentos/202010_Functional_PAC/results/statistics'

data = np.load(os.path.join(path, 'quantile_p1e4_ct10.npy'))

th = 0.05/(2*31*31*2)
data_sig = data.copy()
data_sig[(data > th) & (data < 1-th)] = np.nan

plotter = BandConnectivityPlotter()

mm = plotter.display_band_connectivity(
    data_sig, f_pha=(2, 20), f_amp=(15, 120))

# %% COSAS DE HEADS.

from functionalpac.utils import PAC_connect, DEFAULT_BANDS
from functionalpac.visualization import plot_head_diagram

electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                            'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

filepath = f'/home/pakitochus/Investigacion/Experimentos/202010_Functional_PAC/results/ct10_p1e4/bonferroni_gamma_ct10_p1e4.pdf'
hey = PAC_connect(data_sig, electrodes)
plot_head_diagram(hey, DEFAULT_BANDS, show_electrodes=True, savefig=None)

# 