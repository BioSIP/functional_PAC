"""Visualization Module

This module implements the different visualization options for the 
functional_PAC package. It has several options such as displaying an hiper-matrix
of 

Typical usage example
----------------------
.. code-block:: python

    plotter = BandConnectivityPlotter(electrode_list)
    plotter.plot_connect_hipermatrix(pac_hipermatrix)

-------------------
"""

import pickle
import os
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from scipy.stats import multivariate_normal

from .utils import DEFAULT_BANDS


def plot_specific(hipermatriz, ch_x, ch_y, f_pha=(2, 20), f_amp=(15, 120)):
    """Simply plots the PAC connectivity at channels x and y

    Args:
        hipermatriz (numpy.ndarray): Hipermatrix of PAC connectivity
        ch_x (int): Index of channel giving phase.
        ch_y (int): Index of channel receiving amplitude.
        f_pha (tuple, optional): Range of frequencies for phase. Defaults to (2,20).
        f_amp (tuple, optional): Range of frequencies for amplitude. Defaults to (15,120).
    """
    _, ax = plt.subplots(figsize=(5, 4))
    f = ax.imshow(hipermatriz[ch_x, ch_y], extent=[
                  f_pha[0], f_pha[1], f_amp[0], f_amp[1]], aspect='auto')
    ax.set_title(f'Channels {ch_x}, {ch_y}')
    ax.set_xlabel('Frequency for Phase (Hz)')
    ax.set_ylabel('Frequency for Amplitude (Hz)')
    plt.colorbar(f)


def _pigtailed_cmap(cmap, swap_order=('green', 'red', 'blue')) -> dict:
    """ Utility function to make a new colormap by concatenating a
        colormap with its reverse.
    """
    orig_cdict = cmap._segmentdata.copy()

    cdict = dict()
    cdict['green'] = [(0.5 * (1 - p), c1, c2)
                      for (p, c1, c2) in reversed(orig_cdict[swap_order[0]])]
    cdict['blue'] = [(0.5 * (1 - p), c1, c2)
                     for (p, c1, c2) in reversed(orig_cdict[swap_order[1]])]
    cdict['red'] = [(0.5 * (1 - p), c1, c2)
                    for (p, c1, c2) in reversed(orig_cdict[swap_order[2]])]

    for color in ('red', 'green', 'blue'):
        cdict[color].extend([(0.5 * (1 + p), c1, c2)
                             for (p, c1, c2) in orig_cdict[color]])

    return cdict


class BandConnectivityPlotter(object):
    """Meta object for plotting different sorts of connectivity.
    TODO: reorganize.

    Args:
        channels (list, optional): List of tested channels. Defaults to None.
        bands (dict, optional): Dictionary of bands. Defaults to None.
        cmap (str, optional): Colormap to be used. Defaults to 'coldhot'.
    """

    def __init__(self, channels=None, bands=None, cmap='coldhot'):
        """Constructor method.
        """
        self.channels = channels
        if channels == None:
            self.channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                             'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                             'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                             'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']
        self.n_chan = len(self.channels)
        self.bands = bands
        if bands == None:
            self.bands = {'Delta': (0.5, 4),
                          'Theta': (4, 8),
                          'Alpha': (8, 12),
                          'Beta': (12, 25),
                          'Gamma': (25, 100),
                          'Epsilon': (100, 200)}
        if cmap == 'coldhot':
            self.cmap = LinearSegmentedColormap(
                'coldhot', _pigtailed_cmap(cm.get_cmap('hot')))
        else:
            self.cmap = cm.get_cmap(cmap)

    def check_bands(self, f):
        # TODO: remove
        for band in self.bands:
            if f > self.bands[band][0] and f <= self.bands[band][1]:
                bands = band
        return bands

    def inverse_band_check(self, f_range):
        # TODO: remove
        bands = set()
        for f in range(f_range[0], f_range[1], 2):
            bands.add(self.check_bands(f))
        list_def = [band for band in self.bands if band in bands]
        return list_def

    def plot_pac(self, hipermatrix, f_pha=(2, 20), f_amp=(15, 120), channels=[(0, 0)]):
        n_plots = len(channels)
        ncols = max(1, np.ceil(np.sqrt(n_plots)).astype(int))
        nrows = max(1, np.floor(np.sqrt(n_plots)).astype(int))
        _, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
        ax = ax.flatten()
        vmin, vmax = np.nanmin(hipermatrix), np.nanmax(hipermatrix)
        if vmin < 0:
            vmax = max(-vmin, vmax)
            vmin = -vmax
        else:
            self.cmap = 'inferno'
        for ix in range(n_plots):
            ax[ix].imshow(hipermatrix[channels[ix]], cmap=self.cmap, origin='lower',
                          vmin=vmin, vmax=vmax, aspect='auto',
                          extent=[f_pha[0], f_pha[1], f_amp[0], f_amp[1]])
            ax[ix].set_xlabel(
                f'Frequency of {self.channels[channels[ix][0]]} providing phase (Hz)')
            ax[ix].set_ylabel(
                f'Frequency of {self.channels[channels[ix][1]]} receiving amplitude (Hz)')
            ax[ix].set_title(f'PAC connectivity')

    def plot_multiax_matrix(self, hipermatrix, ma_x, ma_y, mtype='image', 
                            extent='auto', axlabel=None, aylabel=None,
                            figsize='auto'):
        assert mtype in ['matrix', 'image']
        n_x, n_y  = len(ma_x), len(ma_y)
        nn_x, nn_y, n_ph, n_am = hipermatrix.shape
        assert n_x == nn_x
        assert n_y == nn_y
        vmin, vmax = np.nanmin(hipermatrix), np.nanmax(hipermatrix)
        if vmin == vmax:
            vmin, vmax = 0, 1
            print('normalizado')
        if vmin < 0:
            vmax = max(-vmin, vmax)
            vmin = -vmax
        else:
            self.cmap = 'inferno'
        if mtype=='image' and extent=='auto':
            extent = (0, n_ph, 0, n_am)
        if mtype=='matrix' and (axlabel is None or aylabel is None):
            axlabel = np.arange(n_ph)
            aylabel = np.arange(n_am)
            
        if figsize=='auto':
            figsize=(n_x*1.5, n_y*1.5)
        fig, ax = plt.subplots(n_y, n_x, figsize=figsize)
        for ft_x in range(n_x):
            for ft_y in range(n_y):
                matris = hipermatrix[ft_x, ft_y]
                # matris[np.isnan(matris)] = 0. # not correcting.
                if mtype=='matrix':
                    ax[ft_y, ft_x].imshow(matris, cmap=self.cmap, origin='lower',
                                        vmin=vmin, vmax=vmax, aspect='auto')
                    ax[ft_y, ft_x].set_yticks(range(n_ph))
                    ax[ft_y, ft_x].set_xticks(range(n_am))
                    ax[ft_y, ft_x].set_yticklabels(axlabel)
                    ax[ft_y, ft_x].set_xticklabels(aylabel, rotation=60)
                elif mtype=='image':
                    ax[ft_y, ft_x].imshow(matris, cmap=self.cmap, origin='lower',
                                        vmin=vmin, vmax=vmax, aspect='auto',
                                        extent=extent)
                if ft_y == (n_y-1):
                    ax[ft_y, ft_x].set_xlabel(
                        ma_x[ft_x])
                else:
                    ax[ft_y, ft_x].set_xticklabels([])
                if ft_x == 0:
                    ax[ft_y, ft_x].set_ylabel(
                        ma_y[ft_y])
                else:
                    ax[ft_y, ft_x].set_yticklabels([])
        return fig

    def compute_band_connectivity(self, hipermatrix, f_pha=(2, 20), f_amp=(15, 120)):
        # Check which bands there are:
        # TODO: add dependency with PAC_connect and remove this.
        bands_phase = self.inverse_band_check(f_pha)
        bands_amp = self.inverse_band_check(f_amp)
        fpha_ref = np.linspace(f_pha[0], f_pha[1], hipermatrix.shape[-1])
        famp_ref = np.linspace(f_amp[0], f_amp[1], hipermatrix.shape[-2])
        bc = []
        for fas in bands_phase:
            cosa = []
            for famp in bands_amp:
                limits_phas = self.bands[fas]
                limits_amp = self.bands[famp]
                mask = np.outer((fpha_ref <= limits_phas[1]) & (fpha_ref > limits_phas[0]),
                                (famp_ref <= limits_amp[1]) & (famp_ref > limits_amp[0]))
                matrix = np.nanmean(hipermatrix[:, :, mask.T], axis=-1)
                cosa.append(matrix.T)
            bc.append(np.array(cosa))
        return bc

    def display_band_connectivity(self, hipermatrix, f_pha=(2, 20), f_amp=(15, 120)):
        # Check which bands there are:
        # TODO: add dependency with PAC_connect and remove this.
        bands_phase = self.inverse_band_check(f_pha)
        bands_amp = self.inverse_band_check(f_amp)
        figsize=(6*len(bands_phase), 6*len(bands_amp))
        bc = np.array(self.compute_band_connectivity(hipermatrix, f_pha=f_pha, f_amp=f_amp))
        fig = self.plot_multiax_matrix(bc, ma_x=bands_phase, ma_y=bands_amp, mtype='matrix', 
              axlabel=self.channels, aylabel=self.channels, figsize=figsize)
        plt.show()
        return fig

    def display_avg_connectivity(self, hipermatrix):
        _, ax = plt.subplots(figsize=(7, 7))
        if np.nanmin(hipermatrix) < 0:
            vmax = np.nanmax(abs(hipermatrix))
            vmin = -vmax
            colormap = 'coolwarm'
        else:
            vmin, vmax = 0, np.nanmax(hipermatrix)
            colormap = 'inferno'
        val = np.nanmean(np.nanmean(hipermatrix, axis=-1), axis=-1)
        f = ax.matshow(val, vmin=vmin, vmax=vmax, cmap=colormap)
        ax.set_yticks(range(self.n_chan))
        ax.set_xticks(range(self.n_chan))
        ax.set_yticklabels(self.channels)
        ax.set_xticklabels(self.channels, rotation=60)
        plt.colorbar(f)


    def plot_connect_hipermatrix(self, hipermatrix, f_pha=(2, 20), f_amp=(15, 120), filename='untitled'):
        """Plots the hipermatrix connectivity.

        Sets a n_chan by n_chan array of subplots in order, where the x axis is the
        phase frequency and the y axis is the amplitude frequency for displaying the
        PAC matrices between electrodes one by one.

        TODO: check "extent" in imshow.

        Args:
            hipermatrix (numpy.ndarray): PAC connectivity matrix.
            f_pha (tuple, optional): Range of frequencies for phase. Defaults to (2,20).
            f_amp (tuple, optional): Range of frequencies for amplitude. Defaults to (15,120).
            filename (str, optional): Filename to save as pdf. Defaults to 'untitled'. Choose None for not saving.
        """
        n_chan, n_chan_1 , _ , _ = hipermatrix.shape
        assert n_chan == n_chan_1 
        fig = self.plot_multiax_matrix(hipermatrix, ma_x = self.channels[:n_chan], 
                                    ma_y = self.channels[:n_chan],
                                    extent=(f_pha[0], f_pha[1], f_amp[0], f_amp[1]),
                                    mtype='image')
        # plt.tight_layout()
        plt.show()
        if filename is not None:
            fig.savefig(f'{filename}.pdf')


class Head(PatchCollection):
    """Creates a Patch Collection with the form of a head, to be used
    by the EEG array connectivity assistant.

    Args:
        pos (tuple, optional): Center of the head. Defaults to (0, 0).
    """

    def __init__(self, pos=(0, 0)):
        """Constructor method
        """
        self.lineproperties = {'ec': 'black', 'fc': 'none'}
        self.patches = []
        self.head = mpatches.Circle(pos, 1, **self.lineproperties)
        self.nose = self.generate_nose()
        self.ear_L = self.generate_ear(side='left')
        self.ear_R = self.generate_ear(side='right')
        self.patches.append(self.ear_L)
        self.patches.append(self.ear_R)
        self.patches.append(self.nose)
        self.patches.append(self.head)
        PatchCollection.__init__(self, self.patches, linewidths=1,
                                 match_original=True, antialiaseds=True)

    def generate_ear(self, side='left'):
        """Draws the left or right ear.

        Args:
            side (str, optional): Side of the ear. Can be ['left', 'right']. Defaults to 'left'

        Returns:
            PathPatch: Patch with the form of an ear.
        """
        assert side in ['left', 'right']
        sign = -1 if side == 'left' else 1
        path_data = [
            (mpath.Path.MOVETO, [sign*np.cos(np.arcsin(0.2)), -0.2]),
            (mpath.Path.CURVE4, [sign*1.1, -0.3]),
            (mpath.Path.CURVE4, [sign*1.1, 0.3]),
            (mpath.Path.CURVE4, [sign*np.cos(np.arcsin(0.2)), 0.2])]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        return mpatches.PathPatch(path, **self.lineproperties)

    def generate_nose(self):
        """Draws a triangular nose.

        Returns:
            PathPatch: Patch with the form of a nose.
        """
        path_data = [
            (mpath.Path.MOVETO, [-0.15, np.sin(np.arccos(0.15))]),
            (mpath.Path.LINETO, [0, 1.2]),
            (mpath.Path.LINETO, [0.15, np.cos(np.arcsin(0.15))])]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        return mpatches.PathPatch(path, **self.lineproperties)


class EEGarray(PatchCollection):
    """Creates a Patch Array of electrodes, including the background
    of the head.

    Args:
        electrodes ([type]): [description]
        layout (str, optional): [description]. Defaults to 'ten_twenty'.
        marker_size (float, optional): [description]. Defaults to 0.03.
        projection (str, optional): [description]. Defaults to 'simple'.
        ax ([type], optional): [description]. Defaults to None.
    """

    def __init__(self, electrodes, layout='ten_twenty', marker_size=0.03,
                 projection='simple', ax=None):
        """[summary]
        """
        self.layout = layout
        self.marker_size = marker_size
        if ax == None:
            _, self.ax = plt.subplots()
        else:
            self.ax = ax
        if type(electrodes) == list:
            self.names, self.pos = self.get_positions(electrodes)
        elif type(electrodes) == dict:
            self.names = electrodes.keys()
            self.pos = np.asarray(list(electrodes.values()))
        else:
            pass
        assert projection in ['simple', 'lambert']
        self.projection = projection
        self.patches = self.generate_patch_list()
        xx, yy = np.meshgrid(np.arange(-1.5, 1.5, 0.05),
                             np.arange(-1.5, 1.5, 0.05))
        zz = np.zeros(xx.shape)
        self.topodist = dict(xx=xx, yy=yy, zz=zz)
        self.cmap = cm.get_cmap('coolwarm')
        PatchCollection.__init__(
            self, self.patches, match_original=True, antialiaseds=True)
        self.ax.add_collection(self)

    def __len__(self):
        return len(self.names)

    def get_positions(self, electrodes):
        path_data = 'functionalpac/data/'
        eeg_layout = self.load_dict(path_data+self.layout)
        names = [el for i in range(len(electrodes)) for el in list(eeg_layout.keys())
                 if el.upper() == electrodes[i].upper()]
        positions = [eeg_layout[el] for el in names]
        return names, positions

    def generate_patch_list(self):
        patches = []
        for ix, pair in enumerate(self.pos):
            coords_xy = self.convert_sph_xy(pair)
            patches.append(mpatches.Circle(coords_xy, self.marker_size,
                                           ec=None, fc='black'))
            self.label(pair, self.names[ix])
        return patches

    def label(self, pair, text):
        radius = 1.15  # +0.05*np.sin(np.abs(np.deg2rad(pair[0])))
        xy = self.convert_sph_xy(pair, r=radius)
        fs = int(self.marker_size*400)
        self.ax.text(xy[0], xy[1], text, ha="center", size=fs)

    def convert_sph_xy(self, pair, r=1):
        pair = np.deg2rad(pair)
        xyz_coords = (r*np.sin(pair[0])*np.cos(pair[1]),
                      r*np.sin(pair[0])*np.sin(pair[1]),
                      r*np.cos(pair[0]))
        if self.projection == 'simple':
            coords = xyz_coords[:2]
        elif self.projection == 'lambert':
            x, y, z = xyz_coords
            coords = (x*np.sqrt(1/(1+z)), y*np.sqrt(1/(1+z)))
        return coords

    def save_dict(self, obj, name):
        with open(name + '.lay', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_dict(self, name):
        with open(name + '.lay', 'rb') as f:
            return pickle.load(f)

    def get_ch_pos(self, ch):
        return np.asarray(self.convert_sph_xy(self.pos[ch], 1))

    def get_chname_pos(self, chname):
        # TODO: implement
        return np.asarray(self.convert_sph_xy(self.pos[chname], 1))


class Topoplot(object):

    def __init__(self, adjacency, eegarray, cov=None, ax=None, vrange=None):
        self.adjacency = adjacency
        self.eegarray = eegarray
        xx, yy = np.meshgrid(np.arange(-1.6, 1.6, 0.05),
                             np.arange(-1.6, 1.6, 0.05))
        zz = np.zeros(xx.shape)
        self.topodist = dict(xx=xx, yy=yy, zz=zz)
        self.compute_topoplot(cov=cov)
        if ax is not None:
            self.ax = ax
        else:
            _, self.ax = plt.subplots()
        self.vrange = vrange

    def reset_topo(self):
        self.topodist['zz'] = np.zeros(self.topodist['xx'].shape)

    def compute_topoplot(self, cov=None):
        """Generates a topoplot array via the Gaussian Mixture Models method.

        Args:
            cov (np.ndarray, optional): Covariance matrix for the Gaussian models
                                        to be used in the mixture. Defaults to None, 
                                        which means identity covariance with values
                                        1/len(electrodes) at the diagonal.
        """
        nchan = self.adjacency.shape[0]
        assert nchan == self.adjacency.shape[1]
        if cov is None:
            cov = np.eye(2)/len(self.eegarray)
        self.reset_topo()
        xxx = np.stack((self.topodist['xx'], self.topodist['yy']), axis=-1)
        for ch in range(nchan):
            if np.isnan(self.adjacency[ch, ch]) == False:
                mu = self.eegarray.get_ch_pos(ch)
                topo_act = multivariate_normal.pdf(xxx, mean=mu, cov=cov)
                # renormalize for maximum.
                topo_act = (topo_act-topo_act.min()) / \
                    (topo_act.max()-topo_act.min())
                # 0.7 is a normalization coefficient just for visualization.
                self.topodist['zz'] += 0.7*self.adjacency[ch, ch]*topo_act

    def lambert_patch(self):
        """Creates a custom Lambert patch for visualization of the lambert
        head projection

        Returns:
            [matplotlib.mpatches]: Patch for shape of lambert cut
        """
        path_data = [
            (mpath.Path.MOVETO, [-1.5, 0.8]),
            (mpath.Path.CURVE4, [0, 1.77]),
            (mpath.Path.CURVE4, [0, 1.77]),
            (mpath.Path.CURVE4, [1.5, 0.8]),
            (mpath.Path.LINETO, [1.5, -0.1]),
            (mpath.Path.CURVE4, [1.5, -2]),
            (mpath.Path.CURVE4, [-1.5, -2]),
            (mpath.Path.CURVE4, [-1.5, -0.1]),
            (mpath.Path.CLOSEPOLY,  [-1.5, -0.5])]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, transform=self.ax.transData)
        return patch

    def get_min_max(self):
        if self.vrange == None:
            vmax = np.nanmax(np.abs(self.topodist['zz']))
            vmin = -vmax
            if vmin == vmax:
                vmin, vmax = -1, 1
            self.vrange = (vmin, vmax)
        else:
            vmin, vmax = self.vrange
        return vmin, vmax

    def plot_topoplot(self, area=None):
        """Displays the topoplot over the given axis, and cut. 

        Args:
            area ([type], optional): [description]. Defaults to None.
        """
        vmin, vmax = self.get_min_max()
        im = self.ax.pcolormesh(self.topodist['xx'], self.topodist['yy'],
                                self.topodist['zz'], vmin=vmin, vmax=vmax,
                                cmap='coolwarm', antialiaseds=True)  # , shading='gouraud')
        if area is not None:
            if area == 'lambert' and self.eegarray.projection == 'lambert':
                patch = self.lambert_patch()
            else:
                patch = mpatches.Circle(
                    (0, 0), radius=1, transform=self.ax.transData)
        im.set_clip_path(patch)


class SVGPatch(object):
    def __init__(self):
        pass

    def load_patch(self, name):
        with open(name + '.patch', 'rb') as f:
            return pickle.load(f)

    def save_patch(self, obj, name):
        with open(name + '.patch', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class BrainPatch(SVGPatch):

    def __init__(self, projection='lambert', alpha=.5, clip=None, theme='light', color=None):
        self.projection = projection
        assert theme in ['light', 'dark']
        self.route = os.path.dirname(__file__)
        if projection == 'lambert':
            brainpath = os.path.join(
                self.route, f'data/brain_{self.projection}_{theme}')
            self.collection = self.load_patch(brainpath)
            self.collection.set_alpha(alpha)
        else:
            self.collection = None
        if clip:
            self.collection.set_clip_path(clip)
            self.collection.set_clip_on(True)
        if color:
            self.collection.set_color(color)


class HeadPlot(object):

    def __init__(self, adjacency, eegarray, layout='ten_twenty', marker_size=0.03,
                 projection='simple', theme='light', ax=None, vrange=None):
        self.adjacency = adjacency
        self.projection = projection
        self.marker_size = marker_size
        self.layout = layout
        assert theme in ['light', 'dark']
        self.theme = theme
        if ax is not None:
            self.ax = ax
        else:
            _, self.ax = plt.subplots()
        self.vrange = vrange
        if type(eegarray) == list:
            self.eegarray = EEGarray(eegarray, layout=self.layout, projection=self.projection,
                                     marker_size=self.marker_size, ax=self.ax)
        else:
            self.eegarray = eegarray

    def plot_topo(self):
        self.topo = Topoplot(self.adjacency, self.eegarray,
                             cov=None, ax=self.ax, vrange=self.vrange)
        self.topo.plot_topoplot(area='lambert')

    def show_head(self):
        self.ax.add_collection(Head())

    def show_array(self):
        self.ax.add_collection(self.eegarray)

    def show_brainpatch(self, alpha=.5):
        brain_template = BrainPatch(alpha=alpha, theme=self.theme,
                                    clip=self.topo.lambert_patch())
        if brain_template.collection:
            brain_template.collection.set_transform(self.ax.transData)
            if self.theme=='dark':
                kwargs = {'edgecolor': 'white', 'facecolor':'grey'}
            else:       
                kwargs = {'edgecolor': 'grey', 'facecolor':'red'}
            brain_template.collection.set(**kwargs)
            self.ax.add_collection(brain_template.collection)

    def arrow_color(self, value, vmin, vmax, cmap=cm.get_cmap('jet')):
        val = (value-vmin)/(vmax-vmin)
        return cmap(val)

    def connect(self, symmetric=False):
        # check for square matrix
        adjacency = self.adjacency.copy()  # safeguard for pointer passing
        nchan = adjacency.shape[0]
        assert nchan == adjacency.shape[1]
        # arrows = []
        #total = np.sum(np.isnan(adjacency) == False)
        # ix = 0
        cmap = cm.get_cmap('coolwarm')
        vmin, vmax = self.topo.get_min_max()
        for ch_x in range(nchan):
            rang = range(ch_x, nchan) if symmetric else range(nchan)
            for ch_y in rang:
                if ch_x == ch_y:
                    continue
                if np.isnan(adjacency[ch_x, ch_y]) == False:
                    arws = '-|>'
                    if np.isnan(adjacency[ch_y, ch_x]) == False:
                        # TODO: check for bilateral
                        symmetric = True
                        arws = '<|-|>'
                    posA = np.asarray(self.eegarray.get_ch_pos(ch_x))
                    posB = np.asarray(self.eegarray.get_ch_pos(ch_y))
                    color = self.arrow_color(
                        adjacency[ch_x, ch_y], vmin, vmax, cmap)
                    # print(adjacency[ch_x, ch_y], color, vmin, vmax) # old-fashion debugging
                    self.ax.annotate("",
                                     xy=posA, xycoords='data',
                                     xytext=posB, textcoords='data',
                                     arrowprops=dict(arrowstyle=arws, color=color,
                                                     fc=color, ec=color))
                    # arrows.append(mpatches.ConnectionPatch(
                    #     xyA=posA, xyB=posB, coordsA='data', axesA=plt.gca(),
                    #     axesB=plt.gca(), arrowstyle='-|>', fc=cmap(ix), ec=cmap(ix)
                    # ))
                    # arrows.append(mpatches.FancyArrowPatch(
                    #                 posA=posA, posB=posB, fc=cmap(ix), ec=cmap(ix),
                    # arrowstyle=mpatches.ArrowStyle.Simple(head_length=1,
                    # head_width=.707, tail_width=.1), mutation_scale=.1))#,
                    # connectionstyle='angle3'
                    # arrowstyle="-|>", mutation_scale=.2))
                    # ix = ix + 1/total
        # self.patches = self.patches+arrows
        # PatchCollection.__init__(self, self.patches, match_original=True,
        #                         antialiaseds=True)


def plot_head_diagram(adjacency, electrodes, show_topo=True, show_electrodes=True,
                      show_head=True, show_connect=True, projection='simple', theme='light',
                      show_brain=True, title=None, savefig=None):
    """Shows a simple head diagram from the adjacency matrix and electrode list

    Args:
        adjacency (array-like): Adjacency matrix
        electrodes (array-like): List of channels to be included.
        show_topo (bool, optional): Whether to show a topo-plot with intra-channel
                                    connectivity. Defaults to True.
        show_electrodes (bool, optional): Whether to show electrode position. Defaults to True.
        show_head (bool, optional): Whether to show the head schema. Defaults to True.
        show_connect (bool, optional): Whether to show a inter-channel connectivity 
                                with arrows. Defaults to True.
        projection (str, optional): Type of projection. Defaults to 'simple' (simple 
        polar projection). Can be 'lambert' for azimuthal.
        title (str, optional): Title of the graph. Defaults to None.
        savefig ([type], optional): Path to save the figure. Defaults to None.
    """
    fig, ax = plt.subplots()
    headplot = HeadPlot(adjacency, electrodes, theme=theme,
                        projection=projection, ax=ax)
    if show_topo:
        headplot.plot_topo()
    if show_brain:
        headplot.show_brainpatch(alpha=.3)
    if show_head:
        headplot.show_head()
    if show_electrodes:
        headplot.show_array()
    if show_connect:
        headplot.connect()
    ax.set_aspect(1)
    ax.set_ylim(-1.6, 1.5)
    ax.set_xlim(-1.5, 1.5)
    ax.axis('off')
    ax.set_title(title)
    plt.show()
    plt.tight_layout()
    if savefig != None:
        fig.savefig(savefig)


def plot_head_connectivity(pac_object, bands=DEFAULT_BANDS, show_topo=True, show_electrodes=True,
                           show_head=True, show_connect=True, projection='simple', theme='light',
                           show_brain=True, title=None, vrange=None, savefig=None):
    """Shows the connectivity diagram over an array of head plots of shape 

    Args:
        pac_object (PAC_object): PAC_object or PAC_connect object describing the inter-
                                    and intra-channel connectivity values. 
        bands (dict, optional): Band definition dictionary. Defaults to DEFAULT_BANDS.
        show_topo (bool, optional): Whether to show a topo-plot with intra-channel
                                    connectivity. Defaults to True.
        show_electrodes (bool, optional): Whether to show electrode position. Defaults to True.
        show_head (bool, optional): Whether to show the head schema. Defaults to True.
        show_connect (bool, optional): Whether to show a inter-channel connectivity 
                                with arrows. Defaults to True.
        projection (str, optional): Type of projection. Defaults to 'simple' (simple 
        polar projection). Can be 'lambert' for azimuthal.
        title (str, optional): Title of the graph. Defaults to None.
        savefig ([type], optional): Path to save the figure. Defaults to None.
    """
    mm = pac_object.extract_band_connectivity(bands)
    bands_phase = mm.keys()
    bands_amp = mm[list(bands_phase)[0]].keys()
    fig, axes = plt.subplots(len(bands_amp), len(bands_phase),
                             figsize=(4*len(bands_phase), 4*len(bands_amp)+2))
    for i, b_pha in enumerate(bands_phase):
        for j, b_amp in enumerate(bands_amp):
            headplot = HeadPlot(mm[b_pha][b_amp], pac_object.channels, theme=theme,
                                projection=projection, ax=axes[j][i], vrange=vrange)
            if show_topo:
                headplot.plot_topo()
            if show_brain:
                headplot.show_brainpatch(alpha=.3)
            if show_head:
                headplot.show_head()
            if show_electrodes:
                headplot.show_array()
            if show_connect:
                headplot.connect()
            axes[j][i].set_aspect(1)
            axes[j][i].set_ylim(-1.5, 1.3)
            axes[j][i].set_xlim(-1.5, 1.5)
            axes[j][i].axis('off')
            axes[j][i].set_title(f'{b_pha}-{b_amp}')
    if title != None:
        fig.suptitle(title, fontsize=16)
    plt.show()
    plt.tight_layout()
    if savefig != None:
        fig.savefig(savefig)
