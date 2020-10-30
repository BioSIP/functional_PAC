import numpy as np
import scipy
from scipy import signal
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import combinations_with_replacement
from scipy.stats import shapiro
import pickle

"""This script saves a dictionary with the MVL values in the form
key: tuple: (channel_x, channel_y)
values: matrix -> shape ((#hilbert_transforms of channel x, #hilbert_transforms of channel y, 6) -> last shape item is (MVL, MVLz, MVL_Surrogate_Mean, MVL_Surrogate_Std, Freq1, Freq2)
"""


def centroide_espectral(frec, amp):
    num = frec * amp[:-1]
    num = np.sum(num)
    den = np.sum(amp)
    centroide = num / den
    return centroide

def ce_par(freq,amp):
    num = freq*amp[:,:-1]
    centroids = num.sum(axis=1)/amp.sum(axis=1)
    return centroids

def MVL(phase, amp):
    return np.abs(np.mean(amp * np.exp(1j * phase)))

def MVL_par(phase, amp):
    return np.abs(np.inner(amp, np.exp(1j*phase))/amp.shape[1])

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
    
#     if shapiro(surrogate_m)[0] < .9: print("Not normal") #Check if surrogates distribution is normal
        
    mean_surrogate =  np.mean(surrogate_m)
    std_surrogate = np.std(surrogate_m)
    m_norm = (m_raw - mean_surrogate) / std_surrogate

    return np.array([m_raw, m_norm, mean_surrogate, std_surrogate])

def get_pac_par(index_channel_x, index_channel_y, hilbert_data, fs=500, numsurrogate=200):
        channel_x = hilbert_data[index_channel_x]
        channel_y = hilbert_data[index_channel_y]
        pac = np.zeros((channel_x.shape[0], channel_y.shape[0], 6)) 

        amplitude_envelope_x = np.abs(channel_x)
        amplitude_envelope_y = np.abs(channel_y)
        insta_phase_x = np.unwrap(np.angle(channel_x))
        insta_phase_y = np.unwrap(np.angle(channel_y))
        insta_freq_x = np.diff(insta_phase_x, axis=1)*fs/(2*np.pi)
        insta_freq_y = np.diff(insta_phase_y, axis=1)*fs/(2*np.pi)
        ce_x, ce_y = np.meshgrid(ce_par(insta_freq_x, amplitude_envelope_x),
                                ce_par(insta_freq_y, amplitude_envelope_y))
        pac[:,:,4] = ce_x.T # por convención
        pac[:,:,5] = ce_y.T
        mvl_raw = MVL_par(insta_phase_x, amplitude_envelope_y)

        numpoints = insta_phase_x.shape[1]
        skip = np.random.randint(fs, numpoints-fs, size=numsurrogate)
        mvls = np.zeros(mvl_raw.shape+(numsurrogate,))
        for i, k in enumerate(skip): 
            surrogate_amplitude = np.roll(amplitude_envelope_y, k, axis=1)
            mvls[...,i] = MVL_par(insta_phase_x, surrogate_amplitude)
        mvl_u = mvls.mean(axis=2)
        mvl_s = mvls.std(axis=2)
        mvl_z = (mvl_raw - mvl_u)/mvl_s
        pac[:,:,:4] = np.stack((mvl_raw, mvl_z, mvl_u, mvl_s)).transpose((2,1,0))
        return (index_channel_x, index_channel_y),pac

def get_pac(index_channel_x, index_channel_y, hilbert_data):
        """Input: channel index: int, hilber_data: shape ->(channel, imfs, samples)"""
        channel_x = hilbert_data[index_channel_x]
        channel_y = hilbert_data[index_channel_y]
        
        fs = 500
        #Store (MVL, MVLz, MVL_Surrogate_Mean, MVL_Surrogate_Std, Freq1, Freq2) for every hilbert transform
        pac = np.zeros((channel_x.shape[0], channel_y.shape[0], 6)) 
        
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
                
                mvl = np.zeros(4)
                
                if ce_x < ce_y: #Low frequencies phase modulates the amplitude of high frequencies
                    mvl = MVL_Zscore(instantaneous_phase_x, amplitude_envelope_y)
                    
                pac[ht_index_x, ht_index_y] = np.append(mvl,np.array([ce_x, ce_y]))
                
        return (index_channel_x, index_channel_y),pac
    
if __name__ == "__main__":
    
    # hilbert_transform_path = "../data/hilbert"
    # save_path = "../data/descriptors"
    hilbert_transform_path = "/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/hilbert"
    save_path = "/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/descriptors"
    hilbert_files = [file for file in os.listdir(hilbert_transform_path) if file.endswith('.npy')]
    
    bar = tqdm(hilbert_files)
    for filename in bar:
        bar.set_description("Filename: {}".format(filename))
        if os.path.isfile(os.path.join(save_path, filename)): #Check if file already exists to avoid the processing
            continue
        hilbert_t = np.load(os.path.join(hilbert_transform_path,filename), allow_pickle=True)
        channels_combinations = combinations_with_replacement(range(hilbert_t.shape[0]), r = 2)
        
        PACs = {} #Dictionary: keys -> (channel_x, channel_y), values -> pac matrix

        parallel_pacs = Parallel(n_jobs=10)(delayed(get_pac_par)(x, y, hilbert_t) for x,y in list(channels_combinations))

        for key, pac in parallel_pacs:
            PACs[key] = pac                   

        with open(os.path.join(save_path, "{}.pickle".format(filename[:-4])), 'wb') as handle:
            pickle.dump(PACs, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
