# %%
import pickle
import os
import numpy as np
from functionalpac.visualization import BandConnectivityPlotter
from functionalpac.visualization import EEGarray, Head
import matplotlib.pyplot as plt

point_loc = {'Fp1': (-82.5, -19.2, -46.0),
             'Fp2': (82.2, -19.2, -46.0),
             'F7': (53.4, 66.8, -10.3),
             'F3': (-67.2, 39.9, -47.9),
             'Fz': (-61.8, 46.0, 16.2),
             'F4': (-26.4, 54.6, 57.8),
             'F8': (28.3, 55.2, 57.1),
             'FC5': (69.1, 40.3, -48.4),
             'FC1': (-77.4, 13.5, -10.7),
             'FC2': (-57.7, 21.8, 53.2),
             'FC6': (0.4, 26.3, 85.0),
             'T7': (78.4, 14.8, -10.9),
             'C3': (-82.3, -15.2, -46.3),
             'C4': (-34.7, -9.6, 86.0),
             'T8': (36.1, -9.2, 84.7),
             'TP9': (80.0, -12.2, 28.0),
             'CP5': (82.0, -15.7, -46.3),
             'CP1': (-81.3, -44.1, -6.8),
             'CP2': (-60.9, -45.1, 62.9),
             'CP6': (0.4, -45.4, 95.3),
             'TP10': (63.8, -44.7, 62.9),
             'P7': (82.0, -43.7, -6.8),
             'P3': (-70.0, -70.7, -39.3),
             'Pz': (-64.5, -73.1, 27.2),
             'P4': (-27.4, -77.2, 72.3),
             'P8': (30.6, -77.1, 73.5),
             'PO9': (65.1, -72.8, 26.9),
             'PO10': (35.3, -96.7, 34.9),
             'O1': (47.8, -95.3, 20.8),
             'Oz': (53.4, -93.6, 2.6),
             'O2': (52.7, -94.0, -34.1)}


ct = 20
path = f'/home/pakitochus/Investigacion/Experimentos/202010_Functional_PAC/results/p1e4_ct{ct}'
data = np.load(os.path.join(path, f'quantile_p1e4_ct{ct}.npy'))

th = 0.05
data_sig = data.copy()
data_sig[(data > th) & (data < 1-th)] = np.nan

plotter = BandConnectivityPlotter()

mm = plotter.display_band_connectivity(
    data_sig, f_pha=(2, 20), f_amp=(15, 120))
# %%
plotter = BandConnectivityPlotter()
plotter.plot_pac(data_sig, channels=[(15, 0), (1, 10)])

# %%
plotter.plot_connect_hipermatrix(data_sig, filename='quantile_prueba_ct20')

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

show_electrodes = True
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta']):
    for j, bj in enumerate(['Gamma', 'Epsilon']):
        adjacenty = mm[i][j+1]

        array = EEGarray(electrodes, projection='lambert',
                         marker_size=0.03, ax=axes[j][i])
        array.topoplot(adjacenty)
        array.plot_topoplot(array.topodist['zz'], area='lambert')
        # if show_electrodes:
        axes[j][i].add_collection(Head())
        array.plot_connectivity(adjacenty, symmetric=False)
        axes[j][i].set_aspect(1)
        axes[j][i].set_ylim(-1.5, 1.3)
        axes[j][i].set_xlim(-1.5, 1.5)
        axes[j][i].axis('off')
        axes[j][i].set_title(f'{band}-{bj}')
plt.show()
plt.tight_layout()
fig.savefig(
    f'/home/pakitochus/Investigacion/Experimentos/202010_Functional_PAC/results/ct10_p1e4/bonferroni_{band}_gamma_ct10_p1e4.pdf')
