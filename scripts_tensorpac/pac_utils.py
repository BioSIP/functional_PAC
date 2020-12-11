from sklearn.linear_model import LinearRegression
import numpy as np 
import re

class PAC_object(object):
    def __init__(self, pac_hiper, channels, fs=500): 
        self.pac_hiper = pac_hiper
        if channels == None:
            self.channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                             'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                             'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                             'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']
        else:
            self.channels = channels
        self.n_chan = len(self.channels)
        self.pac_shape = self.pac_hiper.shape[2:]
        self.fs = fs

    def purge_decomposition(self):
        self.purged_mat = np.zeros(self.pac_hiper.shape)
        for i in range(self.n_chan):
            for j in range(i, self.n_chan):
                if i!=j:
                    x = np.hstack((self.pac_hiper[i,i].reshape((-1,1)), self.pac_hiper[j,j].reshape(-1,1)))
                    for z in range(2):
                        vecs = [i,j]
                        model = LinearRegression()
                        y = self.pac_hiper[vecs[z],vecs[1-z]].reshape((-1,1))
                        model.fit(x,y)
                        coefs = model.coef_
                        fin = y-x.dot(coefs.T)
                        self.purged_mat[vecs[z],vecs[1-z]] = fin.reshape(self.pac_shape)
                else:
                    self.purged_mat[i,j] = self.pac_hiper[i,j]


# %%
# from sklearn.decomposition import PCA, SparsePCA
# pac_hiper = datos_ctl[0]
# n_chan = pac_hiper.shape[0]
# pca_reg = PCA(n_components=n_chan)#, svd_solver='randomized') 
# sc_reg = pca_reg.fit_transform(pac_hiper.reshape((n_chan**2, -1)))
# sc_reg = sc_reg.reshape([n_chan, n_chan, -1])
# sc_diag = sc_reg[np.diag_indices(31)]


def load_subgroup(path, stimulus=2, group='C', age=7, purge=False, nan=None, channels=None):
    #TODO: safe check of arguments. 
    filenames = [fi for fi in os.listdir(path) if fi.endswith('npy')]
    r = re.compile(f'\d*_{age}_{stimulus}_{group}.npy')
    filenames_r = list(filter(r.match, filenames))
    N = len(filenames_r)
    data = np.zeros((N, len(channels), len(channels), 52, 50))
    for i,fil in enumerate(filenames_r):
        defin = np.load(os.path.join(path, fil))
        if nan is not None:
            defin[np.isnan(defin)] = nan
        if purge: 
            pac = PAC_object(defin, channels)
            pac.purge_decomposition()
            defin = pac.purged_mat
        data[i] = defin
    return data