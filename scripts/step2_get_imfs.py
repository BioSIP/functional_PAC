
import os
import numpy as np
from scipy import signal
from tqdm import tqdm
from PyEMD import EMD
from joblib import Parallel, delayed

"""This script obtains the IMF decomposition of the signals and saves it"""
"""It saves one file per subject containing #channels of array of shapes (imfs, samples)"""
#TODO argparser to obtain root path and save_path


"""
  @misc{pyemd,
  author = {Laszuk, Dawid},
  title = {Python implementation of Empirical Mode Decomposition algorithm},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/laszukdawid/PyEMD}},
}
"""


def preprocessing(data, ref_channel=31, fs=500 ):
    """Signal preprocessing

    Args:
        data (numpy.ndarray): data with shape (subjects, channels, samples)
        ref_channel (int, optional): Channel to be used as reference. Defaults to 31.
        fs (int, optional): Sampling frequency. Defaults to 500.

    Returns:
        [type]: data with shape (subjects, channels-1, samples)
    """    
    batch_data = data.copy()
    # Low filter
    sos_lp = signal.butter(10, 90, 'lowpass', fs=fs, output='sos')
    # High filter
    sos_hp = signal.butter(10, 0.5, 'hp', fs=fs, output='sos')
    
    for subject in tqdm(range(batch_data.shape[0]), desc="Preprocessing"):
        for channel in range(batch_data.shape[1]):
            #Reference channel substraction
            batch_data[subject, channel, :] = batch_data[subject, channel, :] - batch_data[subject, ref_channel, :]
            #Normalization
            batch_data[subject, channel, :] = batch_data[subject, channel, :] - (np.mean(batch_data[subject, channel, :]) / np.std(batch_data[subject, channel, :]))
            #Filtering
            batch_data[subject, channel, :] = signal.sosfilt(sos_lp, batch_data[subject, channel, :])
            batch_data[subject, channel, :] = signal.sosfilt(sos_hp, batch_data[subject, channel, :])
    return batch_data[:,:ref_channel,:]

def batch_imfs_decomposition(data, save_path, name):
    """input: data with shape (subjects, channels, samples)"""
    """output: one file per subject containing #channels of array of shapes (#imfs, samples) """
    def decomposition(channel):
        emd = EMD()
        imf = emd(channel)
        return imf
        
    for subject in tqdm(range(data.shape[0]), desc="IMFs for {}".format(name)):
        imfs = Parallel(n_jobs=10)(delayed(decomposition)(data[subject,k,:].copy()) for k in range(data.shape[1]))
        imfs = np.array(imfs)
        np.save(os.path.join(save_path, name + "_{}".format(subject)) , imfs)

if __name__ == "__main__":
    
    path = "../data/eeg_data"
    save_path = "../data/imfs"
    
    files = os.listdir(path)

    for filename in tqdm(files):
        splits = filename[:-4].split("_")
        name = "{}_{}_{}".format(splits[1], splits[2], splits[3])
        filepath = os.path.join(path, filename)
        data = np.load(filepath)
        data = preprocessing(data)
        batch_imfs_decomposition(data, save_path, name)
        
        
        
    