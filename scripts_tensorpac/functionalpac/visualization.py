import pickle

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from scipy.stats import multivariate_normal
from tensorpac import Pac

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
        if channels == None:
            self.bands = {'Delta': (0.5, 4),
                          'Theta': (4, 8),
                          'Alpha': (8, 12),
                          'Beta': (12, 25),
                          'Gamma': (25, 100),
                          'Epsilon': (100, 200)}
        if cmap == 'coldhot':
            self.cmap = LinearSegmentedColormap(
                'coldhot', _pigtailed_cmap(cm.hot))
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

    def display_band_connectivity(self, hipermatrix, f_pha=(2, 20), f_amp=(15, 120)):
        # Check which bands there are:
        # TODO: add dependency with PAC_connect and remove this.
        bands_phase = self.inverse_band_check(f_pha)
        bands_amp = self.inverse_band_check(f_amp)
        _, ax = plt.subplots(nrows=len(bands_amp), ncols=len(bands_phase),
                             figsize=(6*len(bands_phase), 6*len(bands_amp)))
        fpha_ref = np.linspace(f_pha[0], f_pha[1], hipermatrix.shape[-1])
        famp_ref = np.linspace(f_amp[0], f_amp[1], hipermatrix.shape[-2])
        bc = []
        for i, fas in enumerate(bands_phase):
            cosa = []
            for j, famp in enumerate(bands_amp):
                row = len(bands_amp)-1-j
                limits_phas = self.bands[fas]
                limits_amp = self.bands[famp]
                mask = np.outer((fpha_ref <= limits_phas[1]) & (fpha_ref > limits_phas[0]),
                                (famp_ref <= limits_amp[1]) & (famp_ref > limits_amp[0]))
                matrix = np.nanmean(hipermatrix[:, :, mask.T], axis=-1)
                ax[row, i].imshow(matrix.T, origin='lower')
                # el T es necesario para que la banda de origen sea el eje X
                if row == (len(bands_amp)-1):
                    ax[row, i].set_xlabel(fas)
                if i == 0:
                    ax[row, i].set_ylabel(famp)
                ax[row, i].set_yticks(range(self.n_chan))
                ax[row, i].set_xticks(range(self.n_chan))
                ax[row, i].set_yticklabels(self.channels)
                ax[row, i].set_xticklabels(self.channels, rotation=60)
                cosa.append(matrix.T)
            bc.append(cosa)
        return bc

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
            filename (str, optional): Filename to save as pdf. Defaults to 'untitled'.
        """
        n_chan, _, _, _ = hipermatrix.shape
        fig, ax = plt.subplots(
            n_chan, n_chan, figsize=(n_chan*1.5, n_chan*1.5))
        vmin, vmax = np.nanmin(hipermatrix), np.nanmax(hipermatrix)
        if vmin == vmax:
            vmin, vmax = 0, 1
            print('normalizado')
        if vmin < 0:
            vmax = max(-vmin, vmax)
            vmin = -vmax
        else:
            self.cmap = 'inferno'
        for giving_phase in range(n_chan):
            for receiving_amp in range(n_chan):
                matris = hipermatrix[giving_phase, receiving_amp]
                # matris[np.isnan(matris)] = 0. # not correcting.
                ax[receiving_amp, giving_phase].imshow(matris, cmap=self.cmap, origin='lower',
                                                       vmin=vmin, vmax=vmax, aspect='auto',
                                                       extent=(f_pha[0], f_pha[1], f_amp[0], f_amp[1]))
                if receiving_amp == (n_chan-1):
                    ax[receiving_amp, giving_phase].set_xlabel(
                        self.channels[giving_phase])
                else:
                    ax[receiving_amp, giving_phase].set_xticklabels([])
                if giving_phase == 0:
                    ax[receiving_amp, giving_phase].set_ylabel(
                        self.channels[receiving_amp])
                else:
                    ax[receiving_amp, giving_phase].set_yticklabels([])
        # plt.tight_layout()
        plt.show()
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


# def plot_head_diagram(pac_object, bands=DEFAULT_BANDS, show_electrodes=True,
#                       savefig=None):
#     """Plots a head diagram - topoplot.

#     Args:
#         pac_object ([type]): [description]
#         bands ([type], optional): [description]. Defaults to DEFAULT_BANDS.
#         show_electrodes (bool, optional): [description]. Defaults to True.
#         savefig ([type], optional): [description]. Defaults to None.
#     """
#     mm = pac_object.extract_band_connectivity(bands)
#     bands_phase = mm.keys()
#     bands_amp = mm[list(bands_phase)[0]].keys()
#     fig, axes = plt.subplots(len(bands_amp), len(bands_phase),
#                              figsize=(4*len(bands_phase), 4*len(bands_amp)))
#     for i, band in enumerate(bands_phase):
#         for j, bj in enumerate(bands_amp):
#             adjacenty = mm[band][bj]
#             array = EEGarray(pac_object.channels, projection='lambert',
#                              marker_size=0.03, ax=axes[j][i])
#             array.topoplot(adjacenty)
#             array.plot_topoplot(array.topodist['zz'], area='lambert')
#             # if show_electrodes:
#             axes[j][i].add_collection(Head())
#             array.plot_connectivity(adjacenty, symmetric=False)
#             axes[j][i].set_aspect(1)
#             axes[j][i].set_ylim(-1.5, 1.3)
#             axes[j][i].set_xlim(-1.5, 1.5)
#             axes[j][i].axis('off')
#             axes[j][i].set_title(f'{band}-{bj}')
#     plt.show()
#     plt.tight_layout()
#     if savefig is not None:
#         fig.savefig(savefig)


class Topoplot(object):

    def __init__(self, adjacency, eegarray, cov=None, ax=None):
        self.adjacency = adjacency
        self.eegarray = eegarray
        xx, yy = np.meshgrid(np.arange(-1.5, 1.5, 0.05),
                             np.arange(-1.5, 1.5, 0.05))
        zz = np.zeros(xx.shape)
        self.topodist = dict(xx=xx, yy=yy, zz=zz)
        self.compute_topoplot(cov=cov)
        if ax is not None:
            self.ax = ax
        else:
            _, self.ax = plt.subplots()

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
                self.topodist['zz'] += self.adjacency[ch, ch]*topo_act

    def lambert_patch(self):
        """Creates a custom Lambert patch for visualization of the lambert
        head projection

        Returns:
            [matplotlib.mpatches]: Patch for shape of lambert cut
        """
        path_data = [
            (mpath.Path.MOVETO, [-1.5, 0.6]),
            (mpath.Path.CURVE4, [0, 1.5]),
            (mpath.Path.CURVE4, [0, 1.5]),
            (mpath.Path.CURVE4, [1.5, 0.6]),
            (mpath.Path.LINETO, [1.5, -0.1]),
            (mpath.Path.CURVE4, [1.5, -2]),
            (mpath.Path.CURVE4, [-1.5, -2]),
            (mpath.Path.CURVE4, [-1.5, -0.1]),
            (mpath.Path.CLOSEPOLY,  [-1.5, -0.5])]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, transform=self.ax.transData)
        return patch

    def plot_topoplot(self, area=None):
        """Displays the topoplot over the given axis, and cut. 

        Args:
            area ([type], optional): [description]. Defaults to None.
        """
        vmax = np.nanmax(np.abs(self.topodist['zz']))
        vmin = -vmax
        if vmin-vmax != 0:
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


class HeadPlot(object):

    def __init__(self, adjacency, eegarray, layout='ten_twenty', marker_size=0.03,
                 projection='simple', ax=None):
        self.adjacency = adjacency
        self.projection = projection
        self.marker_size = marker_size
        self.layout = layout
        if ax is not None:
            self.ax = ax
        else:
            _, self.ax = plt.subplots()
        if type(eegarray) == list:
            self.eegarray = EEGarray(eegarray, layout=self.layout, projection=self.projection,
                                     marker_size=self.marker_size, ax=self.ax)
        else:
            self.eegarray = eegarray

    def plot_topo(self):
        self.topo = Topoplot(self.adjacency, self.eegarray, cov=None, ax=self.ax)
        self.topo.plot_topoplot(area='lambert')

    def show_head(self):
        self.ax.add_collection(Head())

    def show_array(self):
        self.ax.add_collection(self.eegarray)

    def connect(self, symmetric=False):
        # check for square matrix
        adjacency = self.adjacency.copy()  # safeguard for pointer passing
        nchan = adjacency.shape[0]
        assert nchan == adjacency.shape[1]
        # arrows = []
        total = np.sum(np.isnan(adjacency) == False)
        ix = 0
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
                    cmap = cm.get_cmap('jet')  # correct for colors in arrows
                    self.ax.annotate("",
                                     xy=posA, xycoords='data',
                                     xytext=posB, textcoords='data',
                                     arrowprops=dict(arrowstyle=arws, color=cmap(ix),
                                                     fc=cmap(ix), ec=cmap(ix)))
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
                    ix = ix + 1/total
        # self.patches = self.patches+arrows
        # PatchCollection.__init__(self, self.patches, match_original=True,
        #                         antialiaseds=True)


def plot_head_connectivity(pac_object, bands=DEFAULT_BANDS, show_topo=True, show_electrodes=True,
                           show_head=True, show_connect=True, projection='simple',
                           savefig=None):
    """Shows the connectivity diagram over head plots 

    Args:
        pac_connect (PAC_object): Can be PAC_object or PAC_connect.
        band_select ([type]): [description]
        show_topo (bool, optional): [description]. Defaults to True.
        show_electrodes (bool, optional): [description]. Defaults to True.
        show_head (bool, optional): [description]. Defaults to True.
        show_connect (bool, optional): [description]. Defaults to True.
    """
    mm = pac_object.extract_band_connectivity(bands)
    bands_phase = mm.keys()
    bands_amp = mm[list(bands_phase)[0]].keys()
    fig, axes = plt.subplots(len(bands_amp), len(bands_phase),
                             figsize=(4*len(bands_phase), 4*len(bands_amp)))
    for i, b_pha in enumerate(bands_phase):
        for j, b_amp in enumerate(bands_amp):
            headplot = HeadPlot(mm[b_pha][b_amp], pac_object.channels,
                                projection=projection, ax=axes[j][i])
            if show_topo:
                headplot.plot_topo()
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
    plt.show()
    plt.tight_layout()
    if savefig is not None:
        fig.savefig(savefig)
