import os
import pickle
import numpy as np 

descriptor_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/descriptors'
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
    Nchan, Nbands = PACs.shape[0], len(bands)
    PACband = np.zeros((Nchan, Nchan, Nbands, Nbands, 2))
    for chx in range(Nchan):
        for chy in range(Nchan):
            pac = PACs[chx, chy]
            for bx, bandx in enumerate(bands.keys()):
                selx = (bands[bandx][0] <= pac[:, :, 2]) & (pac[:, :, 2] < bands[bandx][1])
                for by, bandy in enumerate(bands.keys()):
                    sely = (bands[bandy][0] <= pac[:, :, 3]) & (pac[:, :, 3] < bands[bandy][1])
                    sel = np.where(selx & sely)
                    PACband[chx, chy, bx, by] = pac[sel[0], sel[1]].mean(axis=0)[:2]
                    
    np.save(os.path.join(descriptor_path, f"{fil[:-7]}_band.npy"), PACband)
