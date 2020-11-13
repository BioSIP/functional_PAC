import numpy as np
import visbrain as vs
import os

import matplotlib.pyplot as plt

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

descriptor_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/descriptors'


filenames = [fi for fi in os.listdir(descriptor_path) if fi.endswith('npy')]


filename = filenames[0]



# plt.imshow(PACband[...,4,4,0])
plot_band_connectivity(PACband, bands, ch=0, thr=0.95)
for filename in filenames:
    PACband = np.load(os.path.join(descriptor_path, filename))
    print(f'for {filename}, total : {(PACband[...,1]>0).sum()}')