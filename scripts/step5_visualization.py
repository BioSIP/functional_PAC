import numpy as np
import visbrain as vs
import os
import shutil
import argparse
import matplotlib.pyplot as plt
from visbrain.objects import TopoObj, ColorbarObj, SceneObj, ConnectObj, VisbrainObject
from vispy.scene import visuals
from visbrain.utils import (array2colormap, color2vb, vector_to_opacity)
from collections import Counter
import logging

global bands
global channel_n
bands = {'Delta': (0.5, 4),
         'Theta': (4, 8),
         'Alpha': (8, 12),
         'Beta': (12, 25),
         'Gamma': (25, 140)}

channel_n = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
             'C3', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
             'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']


def plot_band_connectivity(PACband, bands, thr=None, ch=1):
    nx, ny = PACband.shape[2:4]
    fig, ax = plt.subplots(nx, ny, figsize=(30, 25))
    if ch == 1:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = PACband[..., ch].min(), PACband[..., ch].max()
    for bx in range(nx):
        for by in range(ny):
            if thr is None:
                im = ax[bx][by].imshow(PACband[:, :, bx, by, ch], vmin=vmin, vmax=vmax)
            else:
                im = ax[bx][by].imshow(PACband[:, :, bx, by, ch] > thr, vmin=vmin, vmax=vmax)
            ax[bx][by].set_xticks(np.arange(31))
            ax[bx][by].set_xticklabels(channel_n, rotation=90)
            ax[bx][by].set_yticks(np.arange(31))
            ax[bx][by].set_yticklabels(channel_n)
            if bx == 4:
                ax[bx][by].set_xlabel(f'Band {list(bands.keys())[by]}')
            if by == 0:
                ax[bx][by].set_ylabel(f'Band {list(bands.keys())[bx]}')
    fig.colorbar(im, ax=ax[:, :], shrink=0.6)


def plot_band_multicolor(PACband, bands, thr=0.95):
    nx, ny = PACband.shape[2:4]
    ticklabels = [None]
    bandnames = list(bands.keys())
    fig, ax = plt.subplots(figsize=(12, 10))
    pacnan = np.ones(PACband.shape[:2]) * np.nan
    val = 1
    for bx in range(nx):
        for by in range(bx, ny):
            select = PACband[:, :, bx, by, 1] > thr
            pacnan[select] = val
            val += 1
            ticklabels.append(f'({bandnames[bx]}, {bandnames[by]})')

    im = ax.imshow(pacnan, vmin=0, vmax=20, cmap='tab20c')
    ax.set_xticks(np.arange(31))
    ax.set_xticklabels(channel_n, rotation=90)
    ax.set_yticks(np.arange(31))
    ax.set_yticklabels(channel_n)
    ax.set_xlabel('Channel giving phase')
    ax.set_ylabel('Channel receiving amplitude')
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, ticks=np.arange(21))
    for _ in range(21 - len(ticklabels)):
        ticklabels.append(None)
    cbar.ax.set_yticklabels(ticklabels)


# plot_band_multicolor(PACband, bands, thr=0.95)

# from nilearn import plotting
# # correlation_matrix = PACband[...,0,1,0].copy()
# # np.fill_diagonal(correlation_matrix, 0)
# # plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=channel_n, reorder=True)

# def plot_nilearn_connectivity(PACband, bands, thr=None, ch=1, diag=False,
#                                 reorder=True):
#     nx, ny = PACband.shape[2:4]
#     _ , ax = plt.subplots(nx, ny, figsize=(30, 25))
#     for bx in range(nx):
#         for by in range(ny):
#             if ch==1:
#                 vmin, vmax = -1, 1
#             else:
#                 vmin, vmax = min(PACband[...,bx,by,ch].min(),0),PACband[...,bx,by,ch].max()
#             if thr is None:
#                 correlation_matrix = PACband[...,bx,by,ch].copy()
#             else:
#                 correlation_matrix = PACband[...,bx,by,ch].copy()>thr
#             if diag:
#                 np.fill_diagonal(correlation_matrix, 0)
#             plotting.plot_matrix(correlation_matrix, axes=ax[bx,by], 
#                                     labels=channel_n, reorder=reorder, 
#                                     vmin=vmin, vmax=vmax)
#             if bx == 4:
#                 ax[bx][by].set_xlabel(f'Band {list(bands.keys())[by]}')
#             if by == 0:
#                 ax[bx][by].set_ylabel(f'Band {list(bands.keys())[bx]}')

# plot_nilearn_connectivity(PACband, bands, ch=1, thr=0.95)

logger = logging.getLogger('visbrain')


class customConnectObject(ConnectObj):
    def __init__(self, name, nodes, edges, select=None, line_width=2.,
                 color_by='strength', custom_colors=None, alpha=1.,
                 antialias=False, dynamic=None, dynamic_order=1,
                 dynamic_orientation='ascending', cmap='viridis', clim=None,
                 vmin=None, vmax=None, under='gray', over='red',
                 transform=None, parent=None, verbose=None, _z=-10., **kw):
        """Init."""
        VisbrainObject.__init__(self, name, parent, transform, verbose, **kw)
        self._update_cbar_args(cmap, clim, vmin, vmax, under, over)
        # _______________________ CHECKING _______________________
        # Nodes :
        assert isinstance(nodes, np.ndarray) and nodes.ndim == 2
        sh = nodes.shape
        self._n_nodes = sh[0]
        assert sh[1] >= 2
        pos = nodes if sh[1] == 3 else np.c_[nodes, np.full((len(self),), _z)]
        self._pos = pos.astype(np.float32)
        logger.info("    %i nodes detected" % self._pos.shape[0])
        # Edges :
        assert edges.shape == (len(self), len(self))
        if not np.ma.isMA(edges):
            mask = np.zeros(edges.shape, dtype=bool)
            edges = np.ma.masked_array(edges, mask=mask)
        # Select :
        if isinstance(select, np.ndarray):
            assert select.shape == edges.shape and select.dtype == bool
            edges.mask = np.invert(select)
        if color_by is not 'causal':
            edges.mask[np.tril_indices(len(self), 0)] = True
        edges.mask[np.diag_indices(len(self))] = True
        self._edges = edges
        # Colorby :
        assert color_by in ['strength', 'count', 'causal']
        self._color_by = color_by
        # Dynamic :
        if dynamic is not None:
            assert len(dynamic) == 2
        self._dynamic = dynamic
        assert isinstance(dynamic_order, int) and dynamic_order > 0
        self._dyn_order = dynamic_order
        self._dyn_orient = dynamic_orientation
        # Custom color :
        if custom_colors is not None:
            assert isinstance(custom_colors, dict)
        self._custom_colors = custom_colors
        # Alpha :
        assert 0. <= alpha <= 1.
        self._alpha = alpha

        # _______________________ LINE _______________________
        # self._connect = visuals.Line(name='ConnectObjLine', width=line_width,
        #                              antialias=antialias, parent=self._node,
        #                              connect='segments')
        self._connect = visuals.Arrow(name='ConnectObjLine', width=line_width,
                                      antialias=antialias, parent=self._node,
                                      connect='segments', arrow_type="triangle_30", arrow_size=3.)
        self._connect.set_gl_state('translucent', depth_test=False,
                                   cull_face=False)
        self._build_line()

    def _build_line(self):
        """Build the connectivity line."""
        pos, edges = self._pos, self._edges
        # Color either edges or nodes :
        logger.info("    %s coloring method for connectivity" % self._color_by)
        # Switch between coloring method :
        if self._color_by in ['strength', 'count']:
            # Build line position
            nnz_x, nnz_y = np.where(~edges.mask)
            indices = np.c_[nnz_x, nnz_y].flatten()
            line_pos = pos[indices, :]
            if self._color_by == 'strength':
                nnz_values = edges.compressed()
                values = np.c_[nnz_values, nnz_values].flatten()
            elif self._color_by == 'count':
                node_count = Counter(np.ravel([nnz_x, nnz_y]))
                values = np.array([node_count[k] for k in indices])
        elif self._color_by == 'causal':
            idx = np.array(np.where(~edges.mask)).T
            # If the array is not symetric, the line needs to be drawn between
            # points. If it's symetric, line should stop a the middle point.
            # Here, we get the maske value of the symetric and use it to
            # ponderate middle point calculation :
            pond = (~np.array(edges.mask))[idx[:, 1], idx[:, 0]]
            pond = pond.astype(float).reshape(-1, 1)
            div = pond + 1.
            # Build line pos :
            line_pos = np.zeros((2 * idx.shape[0], 3), dtype=float)
            line_pos[0::2, :] = pos[idx[:, 0], :]
            line_pos[1::2, :] = (pos[idx[:, 1]] + pond * pos[idx[:, 0]]) / div
            # Build values :
            values = np.full((line_pos.shape[0],), edges.min(), dtype=float)
            values[1::2] = edges.compressed()
        logger.info("    %i connectivity links displayed" % line_pos.shape[0])
        self._minmax = (values.min(), values.max())
        if self._clim is None:
            self._clim = self._minmax

        # Get the color according to values :
        if isinstance(self._custom_colors, dict):  # custom color
            if None in list(self._custom_colors.keys()):  # {None : 'color'}
                color = color2vb(self._custom_colors[None], length=len(values))
            else:  # black by default
                color = np.zeros((len(values), 4), dtype=np.float32)
            for val, col in self._custom_colors.items():
                color[values == val, :] = color2vb(col)
        else:
            color = array2colormap(values, **self.to_kwargs())
        color[:, -1] = self._alpha

        # Dynamic color :
        if self._dynamic is not None:
            color[:, 3] = vector_to_opacity(values, clim=self._clim,
                                            dyn=self._dynamic,
                                            order=self._dyn_order,
                                            orientation=self._dyn_orient)

        # Send data to the connectivity object :
        arrow_pos = line_pos.reshape(-1, line_pos.shape[-1] * 2)
        print(arrow_pos.shape, color.shape, line_pos.shape)
        self._connect.set_data(pos=line_pos, color=color, arrows=arrow_pos)
        self._connect.arrow_color = color[::2]

def plot_visbrain_connectivity(PACband, path):
    kw_top = dict(margin=15 / 100, chan_offset=(0., 0.1, 0.), chan_size=10)
    kw_cbar = dict(cbtxtsz=12, txtsz=10., width=.3, txtcolor='black', cbtxtsh=1.8,
                   rect=(0., -2., 1., 4.), border=False)
    sc = SceneObj(bgcolor='white', size=(3000, 1300))
    nx, ny = PACband.shape[2:4]
    for bx in range(nx):
        for by in range(bx, ny):
            correlation_matrix = PACband[..., bx, by, 0]
            data = np.diag(correlation_matrix)
            t_obj = TopoObj('topo', data, channels=channel_n, cmap='Spectral_r', **kw_top)
            cb_obj = ColorbarObj(t_obj, cblabel='MVL pac measure', **kw_cbar)
            connect = PACband[..., bx, by, 1]
            np.fill_diagonal(connect, 0)
            select = connect > 0.94
            if np.any(select):
                t_obj._connect = customConnectObject('ChanConnect', t_obj._xyz, correlation_matrix, select=select,
                                                     cmap='inferno', antialias=True, line_width=4.,
                                                     parent=t_obj.node_chan)
                # t_obj.connect(correlation_matrix, select=select, cmap='inferno', antialias=True, line_width=4.)
            sc.add_to_subplot(t_obj, row=by, col=bx * 2,
                              title=f'Phase:{list(bands.keys())[bx]} to Amp:{list(bands.keys())[by]}',
                              title_color='black', width_max=400)
            sc.add_to_subplot(cb_obj, row=by, col=bx * 2 + 1, width_max=100)
    sc.link(-1)
    sc.screenshot(path, print_size=(30, 15))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)
    args = parser.parse_args()

    descriptors_path = args.input_folder
    save_path = args.output_folder

    files = [file for file in os.listdir(descriptors_path) if file.endswith('.npy')]
    for filename in files:
        pac = np.load(os.path.join(descriptors_path, filename))
        new_path = os.path.join(save_path, filename[:-4])
        if os.path.isdir(new_path):
            shutil.rmtree(new_path)
        os.mkdir(new_path)
        plot_band_connectivity(pac, bands)
        plt.savefig(os.path.join(new_path, filename[:-4] + "_connectivity.eps"), format="eps")
        plot_band_multicolor(pac, bands)
        plt.savefig(os.path.join(new_path, filename[:-4] + "_multicolor.eps"), format="eps")
        plot_visbrain_connectivity(pac, os.path.join(new_path, filename[:-4] + "_visbrain.jpg"))
        break
