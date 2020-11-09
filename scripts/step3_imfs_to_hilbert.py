import numpy as np
from scipy import signal
import os
from tqdm import tqdm
from joblib import Parallel, delayed


def normalize_IMF(IMF, thr=0.05, max_iter=20, current_iter=0):
    if (IMF == 0).all():
        return IMF

    envelope = np.abs(signal.hilbert(IMF))
    normalized_imf = np.divide(IMF, envelope)
    peaks = signal.find_peaks(envelope)[0]

    if np.std(peaks) >= thr and current_iter <= max_iter:
        normalized_imf = normalize_IMF(normalized_imf, current_iter=current_iter + 1)
    else:
        normalized_imf = IMF
    return normalized_imf


def normalize_and_hilbert(imfs):
    return signal.hilbert(np.apply_along_axis(normalize_IMF, -1, imfs))


if __name__ == "__main__":
    imfs_path = "../data/imfs_MEMD"
    save_path = "../data/hilbert_MEMD"
    files = [file for file in os.listdir(imfs_path) if file.endswith('.npy')]
    bar = tqdm(files)
    for filename in bar:
        bar.set_description(filename)
        if os.path.isfile(os.path.join(save_path, filename)):  # Check if file already exists to avoid the processing
            continue
        imfs = np.load(os.path.join(imfs_path, filename))
        print(imfs.shape)
        t = Parallel(n_jobs=5)(delayed(normalize_and_hilbert)(imfs[k]) for k in range(imfs.shape[0]))
        np.save(os.path.join(save_path, filename), np.array(t))
