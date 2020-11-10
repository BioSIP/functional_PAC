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

def centroide_espectral(frec, amp):
    num = frec * amp[:-1]
    num = np.sum(num)
    den = np.sum(amp)
    centroide = num / den
    return centroide

def normalize_and_hilbert(imfs):
    return signal.hilbert(np.apply_along_axis(normalize_IMF, -1, imfs))

def pipelineTotal(imfs):
    # 1. señal_hilbert_norm = normalize_and_hilbert()
    # 2. usar CEs = estimar_CEs(señal_hilbert)-> np.mean(freq_instantanea)
    # 3. htt = signal.hilbert(imfs_oriignales)
    # 4. htt_def = htt[CEs>0.5]
    # 5. Guardar htt_def, CEs porque nos viene bien pa luego. 
    # A(t) = np.abs(htt)
    # phi(t) = np.angle(señal_hilbert_norm)
    # htt (t) = A(t)*np.exp(1j*phi(t))
    pass

def purge_IMF(imfs):
    """[summary]
    IMFs es matriz de (19, 31, 71000), devuelve: 
    IMFs purgadas -> (10, 31, 71000)
        la purga es quitar todas aquellas IMFs que tienen un CE <0.5 Hz
    CEs -> (10, 31)
    """

    for index, imfs:
            amplitude_envelope_x = np.abs(imf_ht_x)
        instantaneous_phase_x = np.unwrap(np.angle(imf_ht_x))
        instantaneous_frequency_x = (np.diff(instantaneous_phase_x) / (2.0 * np.pi) * fs)

        ce_x = centroide_espectral(instantaneous_frequency_x, amplitude_envelope_x)

    return imfs_sin_mierdas, ces

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
        t = purge_IMFs(t)
        np.save(os.path.join(save_path, filename), np.array(t))
