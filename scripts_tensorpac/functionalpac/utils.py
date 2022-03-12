"""PAC Managing Utilities

This module has the PAC object, along with several utilities for dealing with
PAC connectivity, data loading, etc. 

Typical usage example
----------------------
::

    pac = Pac_object(pac_matrix)
    pac_purg = pac.purge_decomposition()

..  _Google Python Style Guide:
    http://google.github.io/styleguide/pyguide.html

Util classes:
-------------------
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import re
import os

global DEFAULT_BANDS

DEFAULT_BANDS = {'Delta': (0.5, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 25),
                 'Gamma': (25, 100),
                 'Epsilon': (100, 200)}


class PAC_object(object):
    """This is the basis object for PAC matrices, including the representation of 
    matrix, the channels recorded, the sampling frequency, frequency ranges for amplitude
    and phase, and some utility methods. 

    Args:
        pac_mat ([numpy.ndarray]): PAC hipermatrix. 
        channels (list, numpy.ndarray]): List of channels recorded
        fs (int, optional): Sampling frequency of the EEG. Defaults to 500.
        f_pha (tuple, optional): Frequency range for phase signals. Defaults to (2, 20).
        f_amp (tuple, optional): Frequency range for amplitude signals. Defaults to (15, 120).
    """

    def __init__(self, pac_mat: np.ndarray, channels, fs=500, f_pha=(2, 20), f_amp=(15, 120)):
        """Construction method
        """
        self.pac_mat = pac_mat
        if channels == None:
            self.channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                             'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                             'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                             'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']
        else:
            self.channels = channels
        self.n_chan = len(self.channels)
        self.pac_shape = self.pac_mat.shape[2:]
        self.fs = fs
        self.f_pha = f_pha
        self.f_amp = f_amp

    def purge_decomposition(self):
        """Purges the inter-channel PAC matrices (PAC[i,j]) by estimating the contributions 
        of the intra-channel PAC as: 

        .. math::

            PAC[i,j] = a PAC[i,i] + b PAC[i,j] + e

        and sets PAC[i,j] = e

        TODO: support for PCA/ICA decomposition and volume conductivity. 
        """
        self.purged_mat = np.zeros(self.pac_mat.shape)
        for i in range(self.n_chan):
            for j in range(i, self.n_chan):
                if i != j:
                    x = np.hstack((self.pac_mat[i, i].reshape(
                        (-1, 1)), self.pac_mat[j, j].reshape(-1, 1)))
                    for z in range(2):
                        vecs = [i, j]
                        model = LinearRegression()
                        y = self.pac_mat[vecs[z], vecs[1-z]].reshape((-1, 1))
                        model.fit(x, y)
                        coefs = model.coef_
                        fin = y-x.dot(coefs.T)
                        self.purged_mat[vecs[z], vecs[1-z]
                                        ] = fin.reshape(self.pac_shape)
                else:
                    self.purged_mat[i, j] = self.pac_mat[i, j]

    def check_bands(self, f, bands=DEFAULT_BANDS):
        for band in bands:
            if f > bands[band][0] and f <= bands[band][1]:
                return band

    def inverse_band_check(self, f_range, bands=DEFAULT_BANDS):
        band_set = set()
        for f in range(f_range[0], f_range[1], 2):
            band_set.add(self.check_bands(f, bands))
        list_def = [band for band in bands if band in band_set]
        return list_def

    def extract_band_connectivity(self, bands=DEFAULT_BANDS) -> dict:
        """Extracts the band connectivity of the matrix. 

        Args:
            bands (dictionary): dictionary with band information.

        Returns:
            [dict]: dictionary of dictionaries for conexions. 
        """
        bands_phase = self.inverse_band_check(self.f_pha, bands)
        bands_amp = self.inverse_band_check(self.f_amp, bands)
        fpha_ref = np.linspace(
            self.f_pha[0], self.f_pha[1], self.pac_mat.shape[-1])
        famp_ref = np.linspace(
            self.f_amp[0], self.f_amp[1], self.pac_mat.shape[-2])
        bc = dict()
        for fas in bands_phase:
            cosa = dict()
            for famp in bands_amp:
                limits_phas = bands[fas]
                limits_amp = bands[famp]
                mask = np.outer((fpha_ref <= limits_phas[1]) & (fpha_ref > limits_phas[0]),
                                (famp_ref <= limits_amp[1]) & (famp_ref > limits_amp[0]))
                matrix = np.nanmean(self.pac_mat[:, :, mask.T], axis=-1)
                cosa[famp] = matrix.T
            bc[fas] = cosa
        return bc

# from sklearn.decomposition import PCA, SparsePCA
# pac_hiper = datos_ctl[0]
# n_chan = pac_hiper.shape[0]
# pca_reg = PCA(n_components=n_chan)#, svd_solver='randomized')
# sc_reg = pca_reg.fit_transform(pac_hiper.reshape((n_chan**2, -1)))
# sc_reg = sc_reg.reshape([n_chan, n_chan, -1])
# sc_diag = sc_reg[np.diag_indices(31)]


class PAC_connect(PAC_object):
    """An implementation of PAC_object that focuses on utilities for managing
    connectivity matrices. 

    Args:
        pac_connect ([numpy.ndarray]): PAC connectivity hipermatrix, of size
            n_chan x n_chan x freqs_pha x freqs_amp 
        channels (list, numpy.ndarray]): List of channels recorded
        fs (int, optional): Sampling frequency of the EEG. Defaults to 500.
        f_pha (tuple, optional): Frequency range for phase signals. Defaults to (2, 20).
        f_amp (tuple, optional): Frequency range for amplitude signals. Defaults to (15, 120).
    """

    def __init__(self, pac_connect, channels, fs=500, f_pha=(2, 20), f_amp=(15, 120)):
        """Construction method
        """
        PAC_object.__init__(self, pac_mat=pac_connect, channels=channels, fs=fs,
                            f_pha=f_pha, f_amp=f_amp)


def load_subgroup(path, stimulus=2, group='C', age=7, purge=False, nan=None, channels=None) -> np.array:
    """Utility function for loading a subgroup of the dislexia dataset.

    Args:
        path ([type]): path where the data is.
        stimulus (int, optional): [description]. Defaults to 2.
        group (str, optional): [description]. Defaults to 'C'.
        age (int, optional): [description]. Defaults to 7.
        purge (bool, optional): Whether to purge for inter-channel interferences. Defaults to False.
        nan ([type], optional): Set nans to this. Defaults to None.
        channels ([type], optional): Channel list. Defaults to None.

    Returns:
        [type]: [description]

    TODO: safe check of arguments.
    """
    filenames = [fi for fi in os.listdir(path) if fi.endswith('npy')]
    r = re.compile(f'(\d*)_{age}_{stimulus}[sb]_{group}.npy')
    filenames_r = list(filter(r.match, filenames))
    ids = [r.match(el).groups()[0] for el in filenames_r]
    N = len(filenames_r)
    data = np.zeros((N, len(channels), len(channels), 52, 50))
    for i, fil in enumerate(filenames_r):
        defin = np.load(os.path.join(path, fil))
        if nan is not None:
            defin[np.isnan(defin)] = nan
        if purge:
            pac = PAC_object(defin, channels)
            pac.purge_decomposition()
            defin = pac.purged_mat
        data[i] = defin
    return data
