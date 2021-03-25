from tensorpac import Pac
import numpy as np
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from itertools import permutations
from scipy import signal
import socket

def preprocessing(data, ref_channel=31, fs=500, cut=500, normalization=True, filter=True):
    """Signal preprocessing

    Args:
        data (numpy.ndarray): data with shape (subjects, channels, samples)
        ref_channel (int, optional): Channel to be used as reference. Defaults to 31.
        fs (int, optional): Sampling frequency. Defaults to 500.
        cut(int, optional): Cut the EEG edges. Defaults to 500.
        normalization(bool, optional): Whether to normalize EEG. Defaults to True.
        filter(bool, optional): Whether to filter EEG. Defaults to True.

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
            if normalization:
                stdr = np.std(batch_data[subject, channel, :])
                stdr[stdr==0.]=1.
                batch_data[subject, channel, :] = (batch_data[subject, channel, :] - (
                        np.mean(batch_data[subject, channel, :]) / stdr)
            # Filtering
            if filter:
                batch_data[subject, channel, :] = signal.sosfilt(sos_lp, batch_data[subject, channel, :])
                batch_data[subject, channel, :] = signal.sosfilt(sos_hp, batch_data[subject, channel, :])
    batch_data[np.isnan(batch_data)] = np.random.normal(size=batch_data[np.isnan(batch_data)].shape)
    return batch_data[:, :ref_channel, :]


f_pha = 10  # frequency phase for the coupling
f_amp = 100  # frequency amplitude for the coupling
n_epochs = 20  # number of trials
batches = False  # whether to use batches (TODO)
n_perms = 200  # number of permutations
sf = 500
electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                            'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)
    parser.add_argument('--f_init', type=int, default=15)
    parser.add_argument('--f_end', type=int, default=120)
    parser.add_argument('--f_res', type=int, default=2)
    parser.add_argument("-c", "--cut", type=int, default=500)
    parser.add_argument("-j", "--jobs_number", type=int, default=10)
    parser.add_argument("-n", "--normalization", action="store_true", help="Normalize EEG")
    parser.add_argument("-f", "--filter", action="store_true", help="Filter EEG")
    parser.add_argument("-r", "--reverse", action="store_true", help="Whether to reverse files order")
    args = parser.parse_args()
    print(args)
    path = args.input_folder
    save_path = args.output_folder

    f_init = args.f_init
    f_end = args.f_end
    f_res = args.f_res
    n_jobs = args.jobs_number
    cut = args.cut
    files = [x for x in os.listdir(path) if x.endswith(".npy")]
    if args.reverse:
        files.sort(key=lambda i: i.split("_")[3], reverse=True)
    bar = tqdm(files)
    processed_files = []
    for filename in bar:
        filepath = os.path.join(path, filename)
        data = np.load(filepath)
        bar.set_description(filename + " Preprocessing data")
        # Quiza ni haga falta el preprocessing, si lugo vamos a filtrar. 
        data = preprocessing(data, cut=cut, normalization=args.normalization, filter=args.filter)

        n_subj, n_chan, n_times = data.shape

        for subject_id in range(n_subj):
            name = filename[4:-4]
            f_stimuli = int(name.split('_')[1])
            f_init = f_stimuli
            f_range = np.logspace(np.log10(f_init-.1), np.log10(f_end+10))
            p = Pac(f_pha=[f_stimuli-.5, f_stimuli+.5], f_amp=f_range, dcomplex='hilbert')

            if os.path.isfile(os.path.join(save_path, f'{subject_id}_{name}.npy')):
                continue
            lock_filename_path = os.path.join(save_path, f'{subject_id}_{name}.npy.lock')
            if os.path.isfile(lock_filename_path):
                continue
            with open(lock_filename_path, 'w') as fp:
                pass
            bar.set_description(f'Subject {subject_id}_{name} PAC')
            processed_files.append(f'{subject_id}_{name}')
            # tamaños:
            size_pha, size_amp = 1, len(f_range) - 1

            if batches:
                pass  # TODO
            else:
                stimuli_signal = np.cos(2*np.pi*f_stimuli*np.arange(n_times)/sf)
                amplitudes = np.zeros((n_chan, size_amp, 1, n_times))
                phases = p.filter(sf, stimuli_signal, ftype='phase', n_jobs=n_jobs)
                for chan in range(n_chan):
                    amplitudes[chan] = p.filter(sf, data[subject_id, np.newaxis, chan], ftype='amplitude',
                                                n_jobs=n_jobs)

                subject_pac = np.zeros((n_chan, size_amp, size_pha))
                for chan in range(n_chan):
                    # p.idpac = (1, 2, 4)  # for MVL with surrogate.
                    # p.idpac = (1, 2, 4) # for Modulation Index with surrogate
                    p.idpac = (6, 2, 4) # for Modulation Index with surrogate
                    # compute only the pac without filtering
                    xpac = p.fit(phases, amplitudes[chan], n_perm=n_perms, verbose="WARNING", n_jobs=n_jobs)
                    subject_pac[chan] = xpac.mean(-1)

            np.save(os.path.join(save_path, f'{subject_id}_{name}.npy'), subject_pac.mean(-1))
            os.remove(lock_filename_path)
            np.save(os.path.join(save_path, f'frange_{f_stimuli}.npy'), f_range)
    with open(os.path.join(save_path,"{}_processed.txt".format(socket.gethostname())), "w") as output:
        output.write(str(processed_files))
# %%
# amplitudes.shape
# subject_id = 0
# data = np.load('/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/eeg_data/EEG_7_2_C.npy')

#%% 
# fig, ax  = plt.subplots()
# m = ax.imshow(subject_pac.mean(-1)>2.5, extent=[f_init, f_end,0, n_chan])
# ax.set_ylabel('channel')
# ax.set_xlabel('f_amp')
# #ax.set_title(f'MVL for subject {subject_id}')
# plt.colorbar(m)
# ax.set_yticks(np.arange(n_chan)+0.5)
# ax.set_aspect('auto')
# ax.set_yticklabels(electrodes)
# %%
