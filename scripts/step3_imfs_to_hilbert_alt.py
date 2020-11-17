import numpy as np
from scipy import signal
import os
from tqdm import tqdm
from joblib import Parallel, delayed

global bands
bands = {'Delta': (0.5, 4),
         'Theta': (4, 8),
         'Alpha': (8, 12),
         'Beta': (12, 25),
         'Gamma': (25, 140)}


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
    return signal.hilbert(np.apply_along_axis(normalize_IMF, -1, imfs)).astype(np.complex64)


# def centroide_espectral(frec, amp):
#     num = frec * amp[:-1]
#     num = np.sum(num)
#     den = np.sum(amp)
#     centroide = num / den
#     return centroide

def centroide_espectral(hilbert, fs=500):
    insta_phase = np.unwrap(np.angle(hilbert))
    insta_freq = np.diff(insta_phase) * fs / (2 * np.pi)
    return np.mean(insta_freq, axis=-1)


def purge_IMF(subject_original_imfs, subject_htt, purge_freq=.5):
    """[summary]
    IMFs es matriz de (19, 31, 71000), devuelve:
    IMFs purgadas -> (10, 31, 71000)
        la purga es quitar todas aquellas IMFs que tienen un CE <0.5 Hz
    """
    subject_freqs = centroide_espectral(subject_htt)
    purged_imfs = np.zeros((len(bands),)+subject_original_imfs.shape[1:])

    for i, band in enumerate(bands):
        select = (subject_freqs >= bands[band][0]) & (subject_freqs < bands[band][1])
        purged_imfs[i] = subject_original_imfs[select.all(axis=1)].sum(axis=0)

    purged_norm_hilbert = normalize_and_hilbert(purged_imfs)
    # for channel_idx in range(subject_original_imfs.shape[1]):
    #     freqs = subject_freqs[:, channel_idx]

    #     channel_imfs = subject_original_imfs[:, channel_idx, :]
    #     purged_imfs.append(channel_imfs[freqs >= purge_freq])

    #     htt = subject_htt[:, channel_idx, :]
    #     purged_norm_hilbert.append(htt[freqs >= purge_freq])

    return purged_imfs, purged_norm_hilbert


# def fill_zeros(all_subjects):
#     max_imf = max([x.shape[0] for subject in all_subjects for x in subject])

#     for index, subject in enumerate(all_subjects):
#         for index2, channel_imfs in enumerate(subject):
#             all_subjects[index][index2] = np.vstack(
#                 [channel_imfs,
#                  np.zeros((max_imf - channel_imfs.shape[0], channel_imfs.shape[1])).astype(channel_imfs.dtype)])

#     return np.array(all_subjects)


# def pipelineTotal(imfs):
#     # 1. señal_hilbert_norm = normalize_and_hilbert()
#     # 2. usar CEs = estimar_CEs(señal_hilbert)-> np.mean(freq_instantanea)
#     # 3. htt = signal.hilbert(imfs_oriignales)
#     # 4. htt_def = htt[CEs>0.5]
#     # 5. Guardar htt_def, CEs porque nos viene bien pa luego.
#     # A(t) = np.abs(htt)
#     # phi(t) = np.angle(señal_hilbert_norm)
#     # htt (t) = A(t)*np.exp(1j*phi(t))
#     pass

pimfs, philb = purge_IMF(all_subject_imfs[0], hilbert_norm[0])

if __name__ == "__main__":
    imfs_path = "../data/imfs_MEMD"
    save_path = "../data/hilbert_MEMD"

    # Tests
    # imfs_path = "../data/reduced_data"
    # save_path = "../data/reduced_data"

    files = [file for file in os.listdir(imfs_path) if file.endswith('.npy')]

    bar = tqdm(files)
    for filename in bar:
        bar.set_description(filename)

        if os.path.isfile(
                os.path.join(save_path, filename)):  # Check if file already exists to avoid the processing
            continue
        all_subject_imfs = np.load(os.path.join(imfs_path, filename)).astype(np.float32)

        # # Test
        # all_subject_imfs = np.load(os.path.join(imfs_path, "test_imfs.npy")).astype(np.float32)

        with Parallel(n_jobs=10, prefer="threads", verbose=20) as parallel:
            # Normalize and hilbert
            hilbert_norm = parallel(
                delayed(normalize_and_hilbert)(all_subject_imfs[k]) for k in range(all_subject_imfs.shape[0]))

            # We use the normalized hilbert to compute CE and purge those IMFs with CE < threshold (0.5Hz)
            purged_IMFs = parallel(
                delayed(purge_IMF)(all_subject_imfs[k], hilbert_norm[k]) for k in range(all_subject_imfs.shape[0]))

        del all_subject_imfs, hilbert_norm

        purged_IMFs_zeros = np.array([x[0] for x in purged_IMFs])
        purged_hilbert_norm_zeros = np.array([x[1] for x in purged_IMFs])

        del purged_IMFs

        print("Saving {}".format(filename[:-4]))

        # # Test
        # np.savez(os.path.join(save_path, "test_hilbert_f32"),
        #          ces=centroide_espectral(purged_hilbert_norm_zeros),
        #          hilbert=np.abs(signal.hilbert(purged_IMFs_zeros)).astype(np.complex64) * np.exp(
        #              1j * np.angle(purged_hilbert_norm_zeros))
        #          )

        np.savez(os.path.join(save_path, filename[:-4]),
                 ces=centroide_espectral(purged_hilbert_norm_zeros),
                 hilbert=np.abs(signal.hilbert(purged_IMFs_zeros)).astype(np.complex64) * np.exp(
                     1j * np.angle(purged_hilbert_norm_zeros))
                 )

        del purged_IMFs_zeros, purged_hilbert_norm_zeros

        # Test
        # break
