#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Create 
@Time    :   2021/02/01 09:46:12
@Author  :   F.J. Martinez-Murcia 
@Version :   1.0
@Contact :   pakitochus@gmail.com
@License :   (C)Copyright 2021, SiPBA-BioSIP
@Desc    :   None
'''

# here put the import lib

import pickle
import os
import numpy as np
from functionalpac.visualization import BandConnectivityPlotter
from functionalpac.visualization import EEGarray, Head
from functionalpac.utils import PAC_connect
from functionalpac.visualization import plot_head_connectivity
import matplotlib.pyplot as plt

electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                            'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']
path = '/home/pakitochus/Investigacion/Experimentos/202010_Functional_PAC/results/defs/'

# CONNECTIVITY DATA:
ct = 30
stimulus = 8
formato='pdf'
data = np.load(os.path.join(path, f's{stimulus}/p1e4_ct{ct}/quantile_p1e4_ct{ct}_s{stimulus}.npy'))

# sets the alpha threshold: 
th = 0.025 #twosided
data_sig = data.copy()
data_sig[(data > th) & (data < 1-th)] = np.nan
data_sig = data_sig-0.5

bandas = {'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 25),
        'Gamma': (25, 100),
        'Epsilon': (100, 200)}


# guardar coordenadas:
# coords_xy = pd.DataFrame( np.c_[arr.names, np.array([list(arr.convert_sph_xy(el)) for el in arr.pos])], columns=['name', 'x', 'y'])
# coords_xy.to_csv('coords_xy.csv')

hey = PAC_connect(data_sig, electrodes)
band_connectivity_dict = hey.extract_band_connectivity(bands=bandas)
#%% para plotear: 
plot_head_connectivity(hey, bands=DEFAULT_BANDS, show_topo=True, show_electrodes=True, theme='dark',
                        show_head=True, show_connect=True, projection='lambert', vrange=(-.5, .5),
                        show_brain=True, savefig=fig_route, title=f'PAC Connectivity (CT:{ct}, Stim: {stimulus}Hz)')
