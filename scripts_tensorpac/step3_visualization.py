from tensorpac import Pac
import numpy as np
import os
import matplotlib.pyplot as plt 

f_init = 15
f_end = 120
f_res = 2

pac_path = 'un_path'
filename = '0_7_2_C.npy'

p = Pac(f_pha='hres', f_amp=(f_init, f_end, f_res, f_res), dcomplex='hilbert')

data_subject = np.load(os.path.join(pac_path, file))

def plot_connect_hipermatrix(hipermatriz, p):
    fig, ax = plt.subplots(31, 31, figsize=(40,40))
    n_chan, _ , size_amp, size_pha = hipermatriz.shape
    for i in range(n_chan):
        for j in range(n_chan):
            plt.sca(ax[i,j])
            title = p.method.replace(' (', f' ({i})\n(')
            title += 'channels ({i}, {j})'
            p.comodulogram(hipermatriz[i,j], title=title, cmap='inferno')
    plt.tight_layout()
    plt.show()

plot_connect_hipermatrix(data_subject, p)