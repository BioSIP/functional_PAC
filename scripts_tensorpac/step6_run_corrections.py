#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   step6_run_corrections.py
@Time    :   2021/04/05 19:52:44
@Author  :   F.J. Martinez-Murcia 
@Version :   1.0
@Contact :   pakitochus@gmail.com
@License :   (C)Copyright 2021, SiPBA-BioSIP
@Desc    :   None
'''

# here put the import lib
import numpy as np 
import glob
import os
import ast
import shutil

ORIG_PATH='/home/pakitochus/Investigacion/Databases/dislexia/nico'
DEST_PATH='/home/pakitochus/Investigacion/Databases/dislexia/REFURBISHED/segmentos_corregidos'
ORDER_EEG = '/home/pakitochus/Investigacion/Databases/dislexia/REFURBISHED/refs_order_eeg'

files = glob.glob(os.path.join(ORIG_PATH, '*.dat'))

stims_correspondence = {'2s': '2s',
                        '2b': '20b',
                        '8s': '8s',
                        '8b': '8b',
                        '20s': '20s',
                        '20b': '2b'}

for f in files:
    route, filename = os.path.split(f)
    name = filename.split('.')[0]
    sid, age, stim, diag = name.split('_')
    file_order = os.path.join(ORDER_EEG, f'subjects_order_{stim}_{diag}.txt')
    with open(file_order, 'r') as fil:
        sorder = ast.literal_eval(fil.read())
    dest_id, ag, stimud = sorder[int(sid)].split('.')[0].split('_')
    dest_filename = f'{dest_id}_{age}_{stims_correspondence[stimud]}_{diag}'
    dest_filename += '.npy'
    shutil.copyfile(os.path.join(route, filename),
                    os.path.join(DEST_PATH, dest_filename))

#%% CORRECCIÓN SEGMENTOS
import numpy as np 
import glob
import os
import pandas as pd 

ORIG_PATH='/home/pakitochus/Investigacion/Databases/dislexia/eeg/segmentos'
DEST_PATH='/home/pakitochus/Investigacion/Databases/dislexia/REFURBISHED/segmentos_corregidos'

files = glob.glob(os.path.join(ORIG_PATH, '*.dat'))

stims_correspondence = {'s2': '2s',
                        'b2': '20b',
                        's8': '8s',
                        'b8': '8b',
                        's20': '20s',
                        'b20': '2b'}

class_correspondence = {'c': 'C',
                        'd': 'RD'}

FS = 500 # sampling frequency
TS = 5 # timestep in seconds

for fil in files:
    filename = os.path.split(fil)[-1].split('.')[0]
    sid, cy, typ = filename.split('_')
    clas = class_correspondence[cy[0]]
    age = cy[-1]
    df = pd.read_csv(fil, sep=" ", decimal=',', header=None)
    array = df.values 
    n_seg = int(array.shape[1]/(FS*TS))
    array_segs = array.reshape((array.shape[0], n_seg, -1))
    dest_filename = f'{sid}_{age}_{stims_correspondence[typ]}_{clas}.npy'
    np.save(os.path.join(DEST_PATH, dest_filename), array_segs)

# %%
