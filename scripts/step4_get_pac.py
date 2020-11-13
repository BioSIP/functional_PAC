import numpy as np
import scipy
from scipy import signal
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import permutations
from scipy.stats import shapiro
import pickle

"""This script saves a dictionary with the MVL values in the form
key: tuple: (channel_x, channel_y)
values: matrix -> shape ((#hilbert_transforms of channel x, #hilbert_transforms of channel y, 6) -> last shape item is (MVL, MVLz, MVL_Surrogate_Mean, MVL_Surrogate_Std, Freq1, Freq2)

Bandas: 

"""

global bands
bands = {'Delta': (0.5, 4),
         'Theta': (4, 8),
         'Alpha': (8, 12),
         'Beta': (12, 25),
         'Gamma': (25, 140)}


def MVL_par(phase, amp):
    """Returns the value for Mean Vector Length 
    Args:
        phase (numpy.ndarray): [description]
        amp (numpy.ndarray): [description]

    Returns:
        numpy.ndarray: [description]
    """
    return np.abs(np.inner(amp, np.exp(1j * phase)) / amp.shape[1])


def get_pac_par(index_channel_x, index_channel_y, hilbert, ces,
                fs=500, numsurrogate=200, significance='quantile'):
    """
    Implements the routine for computing the PAC matrices in a vectorized
    way. It includes the computation of the primary MVL_raw matrix, the computation
    of surrogate matrices by applying block-permutation in the amplitude 
    envelopes, and the estimation of the Z-score and significance MVL matrices. 

    Refer to:
    R. T. Canolty et al. High Gamma Power Is Phase-Locked to Theta Oscillations 
    in Human Neocortex. Science 15/09/2006. 1626-1628. 

    Args:
        index_channel_x (int): Index of channel providing phase.
        index_channel_y (int): Index of channel providing amplitude.
        channel_x (numpy.ndarray or list of numpy.ndarray): [description]
        channel_y (numpy.ndarray or list of numpy.ndarray): [description]
        ce_x (numpy.ndarray or list of numpy.ndarray): spectral centroids for x.
        ce_y (numpy.ndarray or list of numpy.ndarray): spectral centroids for y.
        fs (int, optional): Sampling frequency of the data. Defaults to 500.
        numsurrogate (int, optional): Number of surrogate functions used to
                    estimate the significance matrix. Defaults to 200.
        significance ({'quantile', 'zscore'}): Chooses the type of significance
                    estimation of the coupling, via Z-score or quantile. 
                    Defaults to 'quantile'.

    Returns:
        index (tuple): x, y coordinates, equals to index_channel_x, and y
        pac matrix (numpy.ndarray): 3D array containing: 
                                    [:,:,0] - MVL raw matrix 
                                    [:,:,1] - Significance matrix from MVL
                                    [:,:,2] - Spectral Centroid for x
                                    [:,:,3] - Spectral Centroid for y
    """
    assert significance in {'quantile', 'zscore'}
    # First, we extract the data from channel x and y
    channel_x = hilbert[index_channel_x]
    channel_y = hilbert[index_channel_y]
    pac = np.zeros((channel_x.shape[0], channel_y.shape[0], 4))

    # Extracts the amplitude, phase and instantaneous frequency 
    # from the Hilbert signal of channels x and y
    amplitude_envelope_y = np.abs(channel_y)
    insta_phase_x = np.unwrap(np.angle(channel_x))

    # Compute of the spectral centroid 
    ce_x, ce_y = np.meshgrid(ces[index_channel_x], ces[index_channel_y])
    pac[:, :, 2] = ce_x.T  # as a convention, it stays at values 3 and 4
    pac[:, :, 3] = ce_y.T

    # Compute the raw MVL matrix from phase(x) and amplitude(y)
    mvl_raw = MVL_par(insta_phase_x, amplitude_envelope_y)

    # Generates the random amplitude block permutations of MVL
    numpoints = insta_phase_x.shape[1]
    skip = np.random.randint(fs, numpoints - fs, size=numsurrogate)

    if significance == 'zscore':
        # Gets the Z-score matrix from the distribution of surrogate MVLs
        # TODO: sequential estimation of mean and std to avoid memory problems.
        mvls = np.zeros(mvl_raw.shape + (numsurrogate,))
        for i, k in enumerate(skip):
            surrogate_amplitude = np.roll(amplitude_envelope_y, k, axis=1)
            mvls[..., i] = MVL_par(insta_phase_x, surrogate_amplitude)
        mvl_s = (mvl_raw - mvls.mean(axis=2)) / mvls.std(axis=2)
    elif significance == 'quantile':
        # And the quantile matrix from the distribution of surrogate MVLs
        mvls = np.zeros(mvl_raw.shape)
        for i, k in enumerate(skip):
            surrogate_amplitude = np.roll(amplitude_envelope_y, k, axis=1)
            MVLcandidate = MVL_par(insta_phase_x, surrogate_amplitude)
            mvls += mvl_raw> MVLcandidate
        mvl_s = mvls / numsurrogate
    pac[:, :, :2] = np.stack((mvl_raw, mvl_s)).transpose((2, 1, 0))

    return (index_channel_x, index_channel_y), pac


if __name__ == "__main__":

    hilbert_transform_path = "../data/hilbert_MEMD/"
    save_path = "/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/hilbert_MEMD/test"
    hilbert_files = [file for file in os.listdir(hilbert_transform_path) if file.endswith('.npz')]
    bar = tqdm(hilbert_files)
    for filename in bar:
        bar.set_description("Filename: {}".format(filename))

        data = np.load(os.path.join(hilbert_transform_path, filename), allow_pickle=True)
        hilbert_t = data['hilbert']
        ce = data['ces']
        del data
        channels_combinations = list(permutations(range(hilbert_t.shape[1]), 2))
        channels_combinations += [(i, i) for i in range(hilbert_t.shape[1])]

        for id_subject in tqdm(range(hilbert_t.shape[0]), desc="Subject #"):

            subject_filename = "{}_{}".format(id_subject, filename)
            check_filename = subject_filename[:-4] + ".pickle"
            if os.path.isfile(
                    os.path.join(save_path, check_filename)):  # Check if file already exists to avoid the processing
                continue
            lock_filename_path = os.path.join(save_path, check_filename + ".lock")
            if os.path.isfile(lock_filename_path):  # Check if file already exists to avoid the processing
                continue
            with open(lock_filename_path, 'w') as fp:
                pass

            # PACs = {}  # Dictionary: keys -> (channel_x, channel_y), values -> pac matrix

            parallel_pacs = Parallel(n_jobs=10, prefer="threads", verbose=5)(
                delayed(get_pac_par)(x, y, hilbert_t[id_subject], ce[id_subject], numsurrogate=10)
                for x, y in list(channels_combinations)
            )

            # for key, pac in parallel_pacs:
            #     PACs[key] = pac

            Nchan, Nimfs = hilbert_t.shape[1:3]
            Nbands = len(bands)
            PACs = np.zeros((Nchan, Nchan, Nimfs, Nimfs, 4))
            PACband = np.zeros((Nchan, Nchan, Nbands, Nbands, 2))
            for key, pac in parallel_pacs:
                PACs[key[0], key[1]] = pac
                for bx, bandx in enumerate(bands.keys()):
                    selx = (bands[bandx][0] <= pac[:, :, 2]) & (pac[:, :, 2] < bands[bandx][1])
                    for by, bandy in enumerate(bands.keys()):
                        sely = (bands[bandy][0] <= pac[:, :, 3]) & (pac[:, :, 3] < bands[bandy][1])
                        sel = np.where(selx & sely)
                        PACband[key[0], key[1], bx, by] = pac[sel[0], sel[1]].mean(axis=0)[:2]
                        
            with open(os.path.join(save_path, "{}.pickle".format(subject_filename[:-4])), 'wb') as handle:
                pickle.dump(PACs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(os.path.join(save_path, f"{subject_filename[:-4]}_band.npy"), PACband)

            os.remove(lock_filename_path)
            '''
            # Código para cuantificar mejoras en PAC. 
            repeats = 50
            starttime = timeit.default_timer()
            for _ in range(repeats):
                pac = get_pac_par(index_channel_x, index_channel_y, hilbert_data)
            t = timeit.default_timer() - starttime 
            print(f"Time for {repeats} parallel repeats is {t/repeats} :")
            
            starttime = timeit.default_timer()
            for _ in range(repeats):
                pac = get_pac(index_channel_x, index_channel_y, hilbert_data)
            t = timeit.default_timer() - starttime 
            print(f"Time for {repeats} sequential repeats is {t/repeats} :")
            '''
