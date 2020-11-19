import os
import pickle
import numpy as np 

descriptor_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/descriptors_sum_sort'
filenames = [fi for fi in os.listdir(descriptor_path) if fi.endswith('pickle')]


global bands
bands = {'Delta': (0.5, 4),
         'Theta': (4, 8),
         'Alpha': (8, 12),
         'Beta': (12, 25),
         'Gamma': (25, 140)}

for fil in filenames:
    with open(os.path.join(descriptor_path, fil), 'rb') as f:
        PACs = pickle.load(f)
    # Nchan, Nbands = PACs.shape[0], len(bands)
    # PACband = np.zeros((Nchan, Nchan, Nbands, Nbands, 2))
    # for chx in range(Nchan):
    #     for chy in range(Nchan):
    #         pac = PACs[chx, chy]
    #         for bx, bandx in enumerate(bands.keys()):
    #             selx = (bands[bandx][0] <= pac[:, :, 2]) & (pac[:, :, 2] < bands[bandx][1])
    #             for by, bandy in enumerate(bands.keys()):
    #                 sely = (bands[bandy][0] <= pac[:, :, 3]) & (pac[:, :, 3] < bands[bandy][1])
    #                 sel = np.where(selx & sely)
    #                 PACband[chx, chy, bx, by] = pac[sel[0], sel[1]].mean(axis=0)[:2]
                    
    np.save(os.path.join(descriptor_path, f"{fil[:-7]}_band.npy"), PACs[...,:2])


#%% 
from tensorpac import Pac
from tensorpac.signals import pac_signals_wavelet

import matplotlib.pyplot as plt

f_pha = 10      # frequency phase for the coupling
f_amp = 100     # frequency amplitude for the coupling
n_epochs = 20   # number of trials
n_times = 4000  # number of time points
sf = 500.       # sampling frequency


data = np.load('datos.npy')

p = Pac(f_pha='hres', f_amp=(20, 120, 2, 2), dcomplex='wavelet')

phases = p.filter(sf, data[:3,0], ftype='phase', n_jobs=1)
amplitudes = p.filter(sf, data[:3,1], ftype='amplitude', n_jobs=1)


plt.figure(figsize=(14, 8))
for i, k in enumerate([1, 2, 3, 4, 5, 6]):
    # switch method of PAC
    p.idpac = (k, 0, 0)
    # compute only the pac without filtering
    xpac = p.fit(phases[:,0,np.newaxis,:], amplitudes[:,0,np.newaxis,:])
    # plot
    plt.subplot(2, 3, k)
    title = p.method.replace(' (', f' ({k})\n(')
    p.comodulogram(xpac.mean(-1), title=title, cmap='viridis')
    if k <= 3:
        plt.xlabel('')

plt.tight_layout()
plt.show()