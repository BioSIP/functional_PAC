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

ORIG_PATH='/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/pac'
DEST_PATH='/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/tensorpac3'
ORDER_EEG = '/home/pakitochus/Investigacion/Databases/dislexia/REFURBISHED/refs_order_eeg'

files = glob.glob(os.path.join(ORIG_PATH, '*.npy'))

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