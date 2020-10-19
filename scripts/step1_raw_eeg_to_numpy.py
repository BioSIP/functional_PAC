import os
import numpy as np
import pandas as pd
from tqdm import tqdm

#TODO argparser to obtain root path and save_path
#TODO parallel computation
#This script takes the original data and transform it to numpy array binary files
#subida/bajada are processed as if they were different subjects


""" EEG files format: XXX_{C|RD}Y_Z{s|b} where 
    XXX - ID
    C|RD - Control/Dislexia
    Y - age
    Z - Hz
    s|b - Subida/bajada Up/Down
    example: 063_C7_8s.dat """

"""Output: EEG_7_{HZ}_{C|RD}.npy binary numpy files """

root_path = "/media/NAS/Datos/dislexia/eeg/timeseries_entero/7/"
save_path = "../data/eeg_data/"
for target in ['C', 'RD']:
    for hz in [2, 8, 20]:
        path = os.path.join(root_path, target, str(hz))
        files = [file for file in os.listdir(path) if file.endswith('.dat')]
        datamatrix = np.zeros((len(files), 32, 75100))
        bar = tqdm(files)
        for index, f in enumerate(bar):
            bar.set_description("Processing {}".format(f))
            data = pd.read_csv(os.path.join(path,f), delimiter=' |;', header=None, decimal=",")
            data = data.values
            if data.shape[1] > 75100: data = data[:,:75100] #Certain files have s/b together
            datamatrix[index] = data
        save_file_name = "EEG_7_{}_{}".format(hz, target)
        print("Saving file: {}".format(save_file_name))
        np.save(save_path + save_file_name , datamatrix)
