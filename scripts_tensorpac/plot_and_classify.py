#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plot_and_classify.py
@Time    :   2021/03/23 16:58:50
@Author  :   F.J. Martinez-Murcia 
@Version :   1.0
@Contact :   pakitochus@gmail.com
@License :   (C)Copyright 2021, SiPBA-BioSIP
@Desc    :   None
'''

# here put the import lib

import matplotlib.pyplot as plt 
import numpy as np 
import os

data_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/tensorpac_stimuli/'
files = os.listdir(data_path)
electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                            'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']


clase='RD'
stimuli = '20'
plot = 'std'
fils = [el for el in files if el.endswith(f'{stimuli}_{clase}.npy')]

arrays = np.zeros((len(fils), 31, 58))

for ix, fil in enumerate(fils):
	aux = np.load(os.path.join(data_path, fil))
	arrays[ix] = aux

#%% CLASSIFICATION PART



#%% PLOTTING PART 
fig, ax  = plt.subplots()
if plot=='std':
	m = ax.imshow(arrays.mean(0)/arrays.std(0), extent=[2, 120,0, 31]) #/arrays.std(0)
elif plot=='avg':
	m = ax.imshow(arrays.mean(0), extent=[2, 120,0, 31]) #/arrays.std(0)
ax.set_ylabel('channel')
ax.set_xlabel('f_amp')
ax.set_title(f'MVL per channel for class {clase} ({stimuli}Hz)')
plt.colorbar(m)
ax.set_yticks(np.arange(31)+0.5)
ax.set_aspect('auto')
ax.set_yticklabels(electrodes)
fig.savefig(f'{clase}_{stimuli}_{plot}.png')
plt.show()
