import os
import numpy as np
from scipy import signal
from tqdm import tqdm
from PyEMD import EMD
from joblib import Parallel, delayed

"""This script obtains the IMF decomposition of the signals and saves it"""
"""It saves one file per subject containing #channels of array of shapes (imfs, samples)"""
# TODO argparser to obtain root path and save_path


"""
  @misc{pyemd,
  author = {Laszuk, Dawid},
  title = {Python implementation of Empirical Mode Decomposition algorithm},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {'\'url{https://github.com/laszukdawid/PyEMD}},
}
"""


def preprocessing(data, ref_channel=31, fs=500):
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
            # Reference channel substraction
            batch_data[subject, channel, :] = batch_data[subject, channel, :] - batch_data[subject, ref_channel, :]
            # Normalization
            batch_data[subject, channel, :] = batch_data[subject, channel, :] - (
                    np.mean(batch_data[subject, channel, :]) / np.std(batch_data[subject, channel, :]))
            # Filtering
            batch_data[subject, channel, :] = signal.sosfilt(sos_lp, batch_data[subject, channel, :])
            batch_data[subject, channel, :] = signal.sosfilt(sos_hp, batch_data[subject, channel, :])
    return batch_data[:, :ref_channel, :]


def EMD_decomposition(data, name, max_imf=10):
    """Compute the EMD decomposition

    Args:
        data (numpy.ndarray): data with shape (subjects, channels, samples)
        name (string): name of the file to be shown in tqdm description
        max_imf (int, optional): max number of imfs. Defaults to 10

    Returns:
            : data with shape (subjects, channels, max_imf, samples)
            EMD decomposition with #imfs < max_imf will be filled with zeros

    """

    imfs_data = np.zeros((data.shape[0], data.shape[1], max_imf, data.shape[2]))

    def decomposition(channel, max_imf=max_imf):
        emd = EMD()
        emd(channel, max_imf=max_imf)
        decomposition = emd.imfs
        return np.vstack([decomposition, np.zeros(
            (10 - decomposition.shape[0], channel.shape[0]))])  # If we obtain imfs < max_imf we fill it with zeros

    for subject in tqdm(range(data.shape[0]), desc="IMFs for {}".format(name)):
        imfs = Parallel(n_jobs=10)(delayed(decomposition)(data[subject, k, :].copy()) for k in range(data.shape[1]))
        imfs_data[subject] = np.array(imfs)

    return imfs_data


if __name__ == "__main__":

    path = "../data/eeg_data"
    save_path = "../data/imfs_10/"

    files = os.listdir(path)

    for filename in tqdm(files):
        name = filename[:-4]
        filepath = os.path.join(path, filename)
        data = np.load(filepath)
        data = preprocessing(data)
        IMFs = EMD_decomposition(data, name)
        np.save(save_path + name, IMFs)
