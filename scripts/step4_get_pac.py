import numpy as np
import scipy
from scipy import signal
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import combinations_with_replacement
import pickle

"""This script saves a dictionary with the MVL values in the form
key: tuple: (channel_x, channel_y)
values: matrix -> shape ((#hilbert_transforms of channel x, #hilbert_transforms of channel y, 3) -> last shape item is (MVL, frequency1, frequency2)
"""


def centroide_espectral(frec, amp):
    num = frec * amp[:-1]
    num = np.sum(num)
    den = np.sum(amp)
    centroide = num / den
    return centroide

def MVL(phase, amp):
    return np.abs(np.mean(amp * np.exp(1j * phase)))

# Canolty RT (2006) High Gamma Power Is Phase-Locked to Theta. science
def MVL_Zscore(phase, amp):
    srate = 500  # sampling rate used in this study, in Hz

    numpoints = np.size(phase)  # number of sample points in raw signal
    numsurrogate = 200  # number of surrogate values to compare to actual value
    minskip = srate  # time lag must be at least this big
    maxskip = numpoints - srate  # time lag must be smaller than this

    skip = np.ceil(numpoints * np.random.rand(numsurrogate * 2));
    skip = np.delete(skip, np.argwhere(skip > maxskip))
    skip = np.delete(skip, np.argwhere(skip < minskip))
    skip = skip[0:numsurrogate]
    skip = np.array(skip, dtype='int')
    surrogate_m = np.zeros(numsurrogate)
    
    m_raw = MVL(phase, amp)

    surrogate_m = []
    for k in skip:
        surrogate_amplitude = np.concatenate((amp[k:], amp[:k]))
        surrogate_m.append(MVL(phase, surrogate_amplitude))        

    m_norm = (m_raw - np.mean(surrogate_m)) / np.std(surrogate_m)

    if m_norm < 2.326:  # 99
        m_raw = 0
    return m_raw

def get_pac(index_channel_x, index_channel_y, hilbert_data):
        """Input: channel index: int, hilber_data: shape ->(channel, imfs, samples)"""
        channel_x = hilbert_data[index_channel_x]
        channel_y = hilbert_data[index_channel_y]
        
        fs = 500
        #Store (MVL, Freq1, Freq2) for every hilbert transform in every channel
        pac = np.zeros((channel_x.shape[0], channel_y.shape[0], 3)) 
        
        #Iteration over Hilbert Transforms in each channel
        for ht_index_x, imf_ht_x in enumerate(channel_x):
            
            amplitude_envelope_x = np.abs(imf_ht_x)
            instantaneous_phase_x = np.unwrap(np.angle(imf_ht_x))
            instantaneous_frequency_x = (np.diff(instantaneous_phase_x) / (2.0*np.pi) * fs)

            ce_x = centroide_espectral(instantaneous_frequency_x, amplitude_envelope_x)

            for ht_index_y, imf_ht_y in enumerate(channel_y):            
                
                amplitude_envelope_y = np.abs(imf_ht_y)
                instantaneous_phase_y = np.unwrap(np.angle(imf_ht_y))
                instantaneous_frequency_y = (np.diff(instantaneous_phase_y) / (2.0*np.pi) * fs)
                    
                ce_y = centroide_espectral(instantaneous_frequency_y, amplitude_envelope_y)
                
                mvl = 0
                
                if ce_x < ce_y: #Low frequencies phase modulates the amplitude of high frequencies
                    mvl = MVL_Zscore(instantaneous_phase_x, amplitude_envelope_y)
                    
                pac[ht_index_x, ht_index_y] = np.array([mvl, ce_x, ce_y])
                
        return (index_channel_x, index_channel_y),pac
    
if __name__ == "__main__":
    
    hilbert_transform_path = "../data/hilbert"
    save_path = "../data/descriptors"
    hilbert_files = [file for file in os.listdir(hilbert_transform_path) if file.endswith('.npy')]
    
    bar = tqdm(hilbert_files)
    for filename in bar:
        bar.set_description("Filename: {}".format(filename))
        if os.path.isfile(os.path.join(save_path, filename)): #Check if file already exists to avoid the processing
            continue
        hilbert_t = np.load(os.path.join(hilbert_transform_path,filename), allow_pickle=True)
        channels_combinations = combinations_with_replacement(range(hilbert_t.shape[0]), r = 2)
        
        PACs = {} #Dictionary: keys -> (channel_x, channel_y), values -> pac matrix

        parallel_pacs = Parallel(n_jobs=10)(delayed(get_pac)(x, y, hilbert_t) for x,y in list(channels_combinations))

        for key, pac in parallel_pacs:
            PACs[key] = pac                   

        with open(os.path.join(save_path, "{}.pickle".format(filename[:-4])), 'wb') as handle:
            pickle.dump(PACs, handle, protocol=pickle.HIGHEST_PROTOCOL)


