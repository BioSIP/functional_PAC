from tensorpac import Pac
import numpy as np
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from itertools import permutations
from scipy import signal


def preprocessing(data, ref_channel=31, fs=500, cut=500):
    """Signal preprocessing

    Args:
        data (numpy.ndarray): data with shape (subjects, channels, samples)
        ref_channel (int, optional): Channel to be used as reference. Defaults to 31.
        fs (int, optional): Sampling frequency. Defaults to 500.

    Returns:
        [type]: data with shape (subjects, channels-1, samples)
    """
    batch_data = data.copy()
    batch_data = batch_data[:, :, cut:-cut]
    # Low filter
    sos_lp = signal.butter(10, 90, 'lowpass', fs=fs, output='sos')
    # High filter
    sos_hp = signal.butter(10, 0.5, 'hp', fs=fs, output='sos')

    for subject in range(batch_data.shape[0]):
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

    
f_pha = 10      # frequency phase for the coupling
f_amp = 100     # frequency amplitude for the coupling
n_epochs = 20   # number of trials
batches = False # whether to use batches (TODO)
n_perms = 200   # number of permutations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)
    parser.add_argument('f_init', type=int, default=15)
    parser.add_argument('f_end', type=int, default=120)
    parser.add_argument('f_res', type=int, default=2)
    parser.add_argument("-c", "--cut", type=int, default=500)
    parser.add_argument("-j", "--jobs_number", type=int, default=10)
    args = parser.parse_args()
    print(args.cut)
    path = args.input_folder
    save_path = args.output_folder

    f_init = args.f_init
    f_end = args.f_end
    f_res = args.f_res
    n_jobs = args.jobs_number
    fs = args.cut
    p = Pac(f_pha='hres', f_amp=(f_init, f_end, f_res, f_res), dcomplex='hilbert')

    files = [x for x in os.listdir(path) if x.endswith(".npy")]
    bar = tqdm(files)
    for filename in bar:
        name = filename[4:-4]
        filepath = os.path.join(path, filename)
        if os.path.isfile(os.path.join(save_path, name + ".npy")):
            continue
        data = np.load(filepath)
        bar.set_description(name + " Preprocessing data")
        # Quiza ni haga falta el preprocessing, si lugo vamos a filtrar. 
        data = preprocessing(data, cut=fs)

        n_subj, n_chan, n_times = data.shape

        channels_combinations = list(permutations(range(n_chan), 2))
        channels_combinations += [(i, i) for i in range(n_chan)]

        for subject_id in range(n_subj):
            bar.set_description(f'Subject {subject_id} PAC')
            # tamaños: 
            size_pha, size_amp = 50, len(np.arange(f_init, f_end, f_res))-1 

            if batches:
                pass #TODO
            else:
                phases = np.zeros((n_chan, size_pha, 1, n_times))
                amplitudes = np.zeros((n_chan, size_amp, 1, n_times))
                for chan in range(n_chan):
                    phases[chan] = p.filter(fs, data[subject_id, np.newaxis,chan], ftype='phase', edges=fs, n_jobs=n_jobs)
                    amplitudes[chan] = p.filter(fs, data[subject_id, np.newaxis,chan], ftype='amplitude', edges=fs, n_jobs=n_jobs)

                subject_pac = np.zeros((n_chan, n_chan, size_amp, size_pha))
                for comb in channels_combinations:
                    p.idpac = (1, 2, 4) # for MVL with surrogate. 
                    # compute only the pac without filtering
                    xpac = p.fit(phases[comb[0]], amplitudes[comb[1]], n_perm=n_perms)
                    subject_pac[comb[0], comb[1]] = xpac.mean(-1)

            np.save(save_path + f'{subject_id}_{name}.npy', subject_pac)


# %%
# amplitudes.shape
# subject_id = 0
# data = np.load('/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/eeg_data/EEG_7_2_C.npy')
