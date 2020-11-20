from tensorpac import Pac
from tensorpac.signals import pac_signals_wavelet
import numpy as np
import os


import matplotlib.pyplot as plt

f_pha = 10      # frequency phase for the coupling
f_amp = 100     # frequency amplitude for the coupling
n_epochs = 20   # number of trials
n_times = 4000  # number of time points
sf = 500.       # sampling frequency


# %%
data = np.load('/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/eeg_data/EEG_7_2_C.npy')
# %%

p = Pac(f_pha='hres', f_amp=(15, 120, 2, 2), dcomplex='wavelet')

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