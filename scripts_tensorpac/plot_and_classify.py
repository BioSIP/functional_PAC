#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plot_and_classify.py
@Time    :   2021/03/23 16:58:50
@Author  :   F.J. Martinez-Murcia 
@Version :   1.0
@Contact :   pakitochus@gmail.com
@License :   (C)Copyright 2021, SiPBA-BioSIP
@Desc    :   None
'''
#%%
# here put the import lib

import matplotlib.pyplot as plt 
import numpy as np 
import os

data_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/stimuli_pac_mi/'
files = os.listdir(data_path)
electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                            'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

bands = {'Delta': (0.5, 4),
         'Theta': (4, 8),
         'Alpha': (8, 12),
         'Beta': (12, 25),
         'Gamma': (25, 140)}

clases=['C', 'RD']
stimuli = '2'
f_init = int(stimuli)
f_end = 200
data=dict()
perband = False
for clase in clases:
    fils = [el for el in files if el.endswith(f'{stimuli}_{clase}.npy')]
    space = np.load(os.path.join(data_path, f'frange_{stimuli}.npy'))
    scenter = space[:-1] + np.diff(space)/2
    arrays = np.zeros((len(fils), 31, len(scenter)))

    for ix, fil in enumerate(fils):
        aux = np.load(os.path.join(data_path, fil))
        arrays[ix] = aux

    if perband: 
        # space = np.logspace(np.log10(f_init-.1), np.log10(f_end+10))
        # space = np.linspace(f_init, f_end, aux.shape[-1])
        auxiliary = np.zeros(arrays.shape[:-1]+(len(bands),))
        for i, cosa in enumerate(bands):
            f_in, f_out = bands[cosa]
            select = (scenter>f_in)&(scenter<f_out)
            auxiliary[:,:,i] = (arrays[...,select]>2).sum(-1)>0
            # auxiliary[:,:,i] = arrays[...,select].mean(-1)
        arrays = auxiliary.copy()
    data[clase] = arrays

nc = len(data['C'])
nd = len(data['RD'])
channel=11
# X = np.vstack((data['C'].reshape((nc, -1)), data['RD'].reshape((nd, -1))))
X = np.vstack((data['C'][:,channel], data['RD'][:,channel]))
# X[X<2.5] = 0
# X = X > 2.5
y = np.r_[np.zeros(nc), np.ones(nd)]

#%% CLASSIFICATION PART
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import confusion_matrix

prep = StandardScaler()
skf = StratifiedKFold(n_splits=5)

acc=[]
cf = []
for train, test in skf.split(X,y):
    Xtrain = X[train]
    Xtest = X[test]
    # Xtrain = prep.fit_transform(X[train])
    # Xtest = prep.transform(X[test])
    # Xtest[Xtest<0]=0
    

    # pca = PCA(n_components=12)
    # Ztrain = pca.fit_transform(Xtrain)
    # Ztest = pca.transform(Xtest)
    t, p = ttest_ind(Xtrain[y[train]==0], Xtrain[y[train]==1])
    select = p<0.05
    Ztrain = Xtrain[:,select]
    Ztest = Xtest[:,select]
    # Ztrain = Ztrain[:,select]
    # Ztest = Ztest[:,select]
    # Ztrain = Xtrain
    # Ztest = Xtest
    model = LogisticRegression(class_weight='balanced')
    # model = svm.LinearSVC(class_weight='balanced',max_iter=1000000)
    # model = RandomForestClassifier(class_weight='balanced')
    # model = AdaBoostClassifier()
    # model = GaussianNB()
    model.fit(Ztrain, y[train])
    # model = svm.OneClassSVM()
    # model = IsolationForest()
    # model.fit(Ztrain[y[train]==0])
    yhat = model.predict(Ztest)
    cmat = confusion_matrix(yhat, y[test])
    acc.append(np.diag(cmat).sum()/cmat.sum())
    # acc.append(model.score(Ztest, y[test]))
    cf.append(cmat)
    print(f'predicted: {yhat}')

confmat = np.array(cf).sum(axis=0)
print(f'Accuracy: {np.mean(acc):.3f}, per fold: {acc}')
print(f'Sensitivity: {confmat[-1,-1]/confmat[:,-1].sum()}')
print(f'Specificity: {confmat[0,0]/confmat[:,0].sum()}')

#%%
from sklearn.manifold import SpectralEmbedding

model = SpectralEmbedding(n_components=2)
Z = model.fit_transform(X)
fig, ax = plt.subplots()
ax.scatter(Z[y==0,0], Z[y==0,1])
ax.scatter(Z[y==1,0], Z[y==1,1])

# %%
c1 = 2
fig, ax = plt.subplots()
ax.scatter(Ztrain[y[train]==0,c1], Ztrain[y[train]==0,1])
ax.scatter(Ztrain[y[train]==1,c1], Ztrain[y[train]==1,1])
ax.scatter(Ztest[y[test]==0,c1], Ztest[y[test]==0,1])
ax.scatter(Ztest[y[test]==1,c1], Ztest[y[test]==1,1])
#%% PLOTTING PART 
# arrays = model.coef_.reshape((1, 31,-1))
# arrays = pca.components_[-2].reshape((1, 31, -1))
# arrays = p.reshape((1,31,-1))<.1
clase = 'C'
arrays = data[clase]>2.9
plot = 'avg'
fig, ax  = plt.subplots()
if plot=='std':
	m = ax.imshow(arrays.mean(0)/arrays.std(0), extent=[2, 120,0, 31]) #/arrays.std(0)
elif plot=='avg':
	m = ax.imshow(arrays.mean(0), extent=[2, 120,0, 31]) #/arrays.std(0)
ax.set_ylabel('channel')
ax.set_xlabel('f_amp')
ax.set_title(f'MVL per channel for class {clase} ({stimuli}Hz)')
plt.colorbar(m)
ax.set_yticks(np.arange(31)+0.5)
ax.set_aspect('auto')
ax.set_yticklabels(electrodes)
ax.set_xticks(np.linspace(2,120,10))
ax.set_xticklabels([f'{el:.2f}' for el in space[::5]])
#fig.savefig(f'{clase}_{stimuli}_{plot}.png')
plt.show()

# %%
