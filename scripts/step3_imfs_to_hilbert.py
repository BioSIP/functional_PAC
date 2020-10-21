import numpy as np
import scipy
import os
from pqdm.processes import pqdm #Parallel TQDM library


# def normalize_IMF2(IMFs,thr=0.05,max_iter=100):
#     peakvalue_std=[]
#     IMFsn=np.zeros((IMFs.shape[0]-1,IMFs.shape[1]))
#     for i in range(0,len(IMFs)-1):
#         #IMF=np.abs(IMFs[i,:])                        
#         IMFn=IMFs[i,:]
#         #IMFn[-1]=np.mean(IMFn)
#         #IMFn[0]=np.mean(IMFn)
#         IMFpeaks,_= find_peaks(IMFn, height=0)
#         pstd=np.std(IMFn[IMFpeaks])
#         ii=0
#         print("IMF= ",i+1,end='')
#         while (pstd>=thr and ii<max_iter):
#               print(". Normalizing... iter:",ii+1, "peakstd: ", pstd)
#               z=hilbert(IMFn)
#               ia,,=calcula_inst(z)            
#               #ia[0]=IMFn[0]; ia[-1]=IMFn[-1]
#               IMFn=IMFn/ia
#               IMFpeaks,_= find_peaks(IMFn, height=0)
#               pstd=np.std(IMFn[IMFpeaks])
#               ii=ii+1
#         print(". Final peakstd:",pstd)
#         peakvalue_std.append(pstd)
#         IMFsn[i,:]=IMFn
#     return IMFsn, peakvalue_std 

def normalize_IMF(IMF, thr=0.05, max_iter=100, current_iter=0):
    envelope = np.abs(signal.hilbert(IMF))
    normalized_imf = np.divide(IMF, envelope)
    peaks = signal.find_peaks(envelope)[0]
    
    if np.std(peaks) >= thr and current_iter <= max_iter:
        normalized_imf = normalize_IMF(normalized_imf, current_iter=current_iter+1)
    else:
        normalized_imf = IMF
    return normalized_imf  

def hilbert_t(filename, reduced_imfs_path = "../data/reduced_imfs", save_path = "../data/hilbert", normalize_imf=True ):
    if os.path.isfile(os.path.join(save_path, filename)): #Check if file already exists to avoid the processing
        return
    channels_imfs = np.load(os.path.join(reduced_imfs_path,filename), allow_pickle=True)
    ht_analytics = []
    
    for channel in channels_imfs:
        if normalize_imf:
            normalized_imfs = np.apply_along_axis(normalize_IMF, -1, channel)
        else:
            normalize_imfs = channel
        #Compute hilbert transforms in all IMFs of the current channel
        ht_analytics.append(signal.hilbert(normalized_imfs))
    np.save(os.path.join(save_path, filename) , np.array(ht_analytics))
 
if __name__ == "__main__":
    
    reduced_imfs_path = "../data/reduced_imfs"
    save_path = "../data/hilbert"
    reduced_imfs_files = [file for file in os.listdir(reduced_imfs_path) if file.endswith('.npy')]
    
    _ = pqdm(reduced_imfs_files, hilbert_t, n_jobs=10)