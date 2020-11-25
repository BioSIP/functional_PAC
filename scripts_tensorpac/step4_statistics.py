import numpy as np
import os
import matplotlib.pyplot as plt 
import re

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

descriptor_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/descriptors_sum_sort'


filenames = [fi for fi in os.listdir(descriptor_path) if fi.endswith('npy')]


filename = filenames[0]

def load_subgroup(path, stimulus=2, group='C'):
    filenames = [fi for fi in os.listdir(descriptor_path) if fi.endswith('npy')]
    r = re.compile(f'\d*_7_{stimulus}_{group}_band.npy')
    filenames_r = list(filter(r.match, filenames))
    N = len(filenames_r)
    data = np.zeros((N, len(channel_n), len(channel_n), len(bands), len(bands), 2))
    for i,fil in enumerate(filenames_r):
        data[i] = np.load(os.path.join(path, fil))
    return data

datos_ctl = load_subgroup(descriptor_path, group='C', stimulus=2)

datos_dlx = load_subgroup(descriptor_path, group='RD', stimulus=2)