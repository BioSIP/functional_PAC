from scipy.stats import multivariate_normal
from matplotlib import cm
from matplotlib.collections import PatchCollection
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from tensorpac import Pac
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps


def _pigtailed_cmap(cmap, swap_order=('green', 'red', 'blue')):
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
    def __init__(self, channels=None, bands=None, cmap='coldhot'):
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
        for band in self.bands:
            if f > self.bands[band][0] and f <= self.bands[band][1]:
                bands = band
        return bands

    def inverse_band_check(self, f_range):
        bands = set()
        for f in range(f_range[0], f_range[1], 2):
            bands.add(self.check_bands(f))
        list_def = [band for band in self.bands if band in bands]
        return list_def

    def plot_pac(self, hipermatrix, f_pha=(2, 20), f_amp=(15, 120), channels=[(0, 0)]):
        n_plots = len(channels)
        ncols = max(1, np.ceil(np.sqrt(n_plots)).astype(int))
        nrows = max(1, np.floor(np.sqrt(n_plots)).astype(int))
        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
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
        bands_phase = self.inverse_band_check(f_pha)
        bands_amp = self.inverse_band_check(f_amp)
        fig, ax = plt.subplots(nrows=len(bands_amp), ncols=len(bands_phase),
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
        n_chan, _, _, _ = hipermatrix.shape
        fig, ax = plt.subplots(
            n_chan, n_chan, figsize=(n_chan*1.5, n_chan*1.5))
        vmin, vmax = np.nanmin(hipermatrix), np.nanmax(hipermatrix)
        if vmin < 0:
            vmax = max(-vmin, vmax)
            vmin = -vmax
        else:
            self.cmap = 'inferno'
        for giving_phase in range(n_chan):
            for receiving_amp in range(n_chan):
                matris = hipermatrix[giving_phase, receiving_amp]
                # matris[np.isnan(matris)] = 0.
                print(matris.shape)
                ax[receiving_amp, giving_phase].imshow(matris, cmap=self.cmap, origin='lower',
                                                       vmin=vmin, vmax=vmax, aspect='auto',
                                                       extent=[f_pha[0], f_pha[1], f_amp[0], f_amp[1]])
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
    def __init__(self, pos=(0, 0)):
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
        sign = -1 if side == 'left' else 1
        path_data = [
            (mpath.Path.MOVETO, [sign*np.cos(np.arcsin(0.2)), -0.2]),
            (mpath.Path.CURVE4, [sign*1.1, -0.3]),
            (mpath.Path.CURVE4, [sign*1.1, 0.3]),
            (mpath.Path.CURVE4, [sign*np.cos(np.arcsin(0.2)), 0.2])]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        return mpatches.PathPatch(path, **self.lineproperties)

    def generate_nose(self, side='left'):
        path_data = [
            (mpath.Path.MOVETO, [-0.15, np.sin(np.arccos(0.15))]),
            (mpath.Path.LINETO, [0, 1.2]),
            (mpath.Path.LINETO, [0.15, np.cos(np.arcsin(0.15))])]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        return mpatches.PathPatch(path, **self.lineproperties)


class EEGarray(PatchCollection):
    def __init__(self, electrodes, layout='ten_twenty', marker_size=0.03,
                 projection='simple', ax=None):
        self.layout = layout
        self.marker_size = marker_size
        if ax == None:
            fig, self.ax = plt.subplots()
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

    def get_positions(self, electrodes):
        path_data = '../data/'
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

    def reset_topo(self):
        self.topodist['zz'] = np.zeros(self.topodist['xx'].shape)

    def topoplot(self, adjacency, cov=None):
        nchan = adjacency.shape[0]
        assert nchan == adjacency.shape[1]
        if cov is None:
            cov = 0.05*np.eye(2)
        self.reset_topo()
        xxx = np.stack((self.topodist['xx'], self.topodist['yy']), axis=-1)
        for ch in range(nchan):
            if np.isnan(adjacency[ch, ch]) == False:
                mu = np.asarray(self.convert_sph_xy(self.pos[ch], 1))
                topo_act = multivariate_normal.pdf(xxx, mean=mu, cov=cov)
                self.topodist['zz'] += adjacency[ch, ch]*topo_act

    def plot_topoplot(self, adjacency, area=None):
        vmax = np.nanmax(np.abs(self.topodist['zz']))
        vmin = -vmax
        if vmin-vmax != 0:
            im = self.ax.pcolormesh(self.topodist['xx'], self.topodist['yy'],
                                    self.topodist['zz'], vmin=vmin, vmax=vmax,
                                    cmap='coolwarm', antialiaseds=True)  # , shading='gouraud')
            if area == 'lambert' and self.projection == 'lambert':
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
            else:
                patch = mpatches.Circle(
                    (0, 0), radius=1, transform=self.ax.transData)
            im.set_clip_path(patch)

    def plot_connectivity(self, adjacency, symmetric=False):
        # check for square matrix
        adjacency = adjacency.copy()  # safeguard for pointer passing
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
                    posA = np.asarray(self.convert_sph_xy(self.pos[ch_x], 1))
                    posB = np.asarray(self.convert_sph_xy(self.pos[ch_y], 1))
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
