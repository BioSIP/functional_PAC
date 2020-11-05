import numpy as np
import scipy
from PyEMD import EMD
from scipy import signal

"""
Vamos a hacerlo de forma que, dados:
- Los segmentos temporales de EEG de un sujeto
En general el algoritmo consta de las siguientes partes: 
1. Normalización y preprocesado del EEG.
2. Estimación de IMFs a partir de los datos.
3. Aplicación de la transformada de hilbert.
4. Creación de las matrices PAC inter- e intra-canal. 
"""

class EEGNormalizer(object):
    """
    Helper to normalize EEG data in numpy.ndarray format. 
    """
    pass


class PACMatrix(object):
    """
    Construye una matriz PAC
    """
    pass


class HilbertHuang(object):
    """
    TODO
    This class is intended to apply the Hibert-Huang transform (HHT) [Huang 2005] on 
    the input data. The HHT uses the Empirical Mode Decomposition (EMD) method to 
    decompose a signal into so-called intrinsic mode functions (IMF) with a trend, 
    and applies the Hilbert Spectral Analysis method to the IMFs to obtain 
    instantaneous frequency data.
    This procedure consists of three major steps: 
        1. EMD of the input signals and modelling of the IMFs. 
        2. Normalization of the IMFs to constant amplitude [Huang 2005]. 
        2. Application of Hilbert Spectral Analysis over the normalized IMFs.
    Since the signal is decomposed in time domain and the length of the IMFs is the 
    same as the original signal, HHT preserves the characteristics of the varying 
    frequency.

    Huang 2005. HUANG, Norden Eh. Hilbert-Huang transform and its applications. 
        World Scientific, 2005.
    """
    pass
    
    def __init__(self, n_imf=10):
        self.emd = EMD()
        self.imfs = None
        self.normed = False # To be True when normalize_imfs has been applied
        if isinstance(n_imf, int):
            self.n_imf = n_imf
        else:
            self.n_imf = -1
        
        pass
    
    def normalize_IMF(self, thr=0.05, max_iter=20, current_iter=0):
        """Recursive implementation of IMF normalization procedure. 
        TODO: comprobar que es más rapido que iterativo. 

        Args:
            thr (float, optional): [description]. Defaults to 0.05.
            max_iter (int, optional): [description]. Defaults to 20.
            current_iter (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """
        envelope = np.abs(signal.hilbert(self.imfs))
        normed_imf = np.divide(self.imfs, envelope)
        peaks = signal.find_peaks(envelope)[0]

        if np.std(peaks) >= thr and current_iter <= max_iter:
            normed_imf = self.normalize_IMF(normed_imf, current_iter=current_iter + 1)
        else:
            normed_imf = self.imfs

        self.normed = True
        return normed_imf

    def apply_hsa(self):
        """Applies Hilbert Spectral Analysis (HSA) to current IMFs

        Returns:
            [type]: [description]
        """
        assert self.normed
        return signal.hilbert(self.imfs)

    def funcname(self, parameter_list):
        pass

    def fit(self, x):
        # estimate imfs. normalize imfs. apply hsa. 
        pass

    def predict(self, x):
        pass



class FunctionalPAC(object):
    """
    TODO
    This class implements the Cross-Frequency Coupling (CFC) estimation for Phase-
    Amplitude Coupling (PAC), using the Mean-Vector Length (MVL) method proposed 
    in [Canolty 2010]. 

    MVL is suitable for noisy data, exploratory analyses (analyzing a broad frequency 
    spectrum) and when the power of the amplitude providing frequency band is low.
    [Onslow 2011]. 

    Canolty 2010. CANOLTY, Ryan T.; KNIGHT, Robert T. The functional role of 
        cross-frequency coupling. Trends in cognitive sciences, 2010, vol. 14, no 11, 
        p. 506-515.
    """
    pass
    
    def __init__(self, numsurrogate=200, significance='quantile', pair='band'):  

        assert significance in {'quantile', 'zscore'}
        self.significance = significance
        assert pair in {'imf', 'band'}
        self.pair = pair
        assert isinstance(numsurrogate, int)
        self.numsurrogate = numsurrogate
        pass

    def funcname(self, parameter_list):
        """
        docstring
        """
        pass

    def fit(self, x):
        pass

    def predict(self, x):
        pass

    def fit_predict(self, x):
        pass





