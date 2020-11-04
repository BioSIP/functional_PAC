import numpy as np
import scipy
from scipy import signal
import os
from pqdm.processes import pqdm  # Parallel TQDM library


def normalize_IMF(IMF, thr=0.05, max_iter=20, current_iter=0):
    envelope = np.abs(signal.hilbert(IMF))
    normalized_imf = np.divide(IMF, envelope)
    peaks = signal.find_peaks(envelope)[0]

    if np.std(peaks) >= thr and current_iter <= max_iter:
        normalized_imf = normalize_IMF(normalized_imf, current_iter=current_iter + 1)
    else:
        normalized_imf = IMF
    return normalized_imf


def hilbert_t(filename, reduced_imfs_path="../data/reduced_imfs", save_path="../data/hilbert", normalize_imf=True):
    if os.path.isfile(os.path.join(save_path, filename)):  # Check if file already exists to avoid the processing
        return
    channels_imfs = np.load(os.path.join(reduced_imfs_path, filename), allow_pickle=True)
    ht_analytics = []

    for channel in channels_imfs:
        if normalize_imf:
            normalized_imfs = np.apply_along_axis(normalize_IMF, -1, channel)
        else:
            normalize_imfs = channel
        # Compute hilbert transforms of the IMFs of the current channel
        ht_analytics.append(signal.hilbert(normalized_imfs))
    np.save(os.path.join(save_path, filename), np.array(ht_analytics, dtype=object))


if __name__ == "__main__":
    reduced_imfs_path = "../data/reduced_imfs"
    save_path = "../data/hilbert"
    reduced_imfs_files = [file for file in os.listdir(reduced_imfs_path) if file.endswith('.npy')]
    #     hilbert_t(reduced_imfs_files[0]) #For a single file cointainig IMFs
    _ = pqdm(reduced_imfs_files, hilbert_t, n_jobs=10)
