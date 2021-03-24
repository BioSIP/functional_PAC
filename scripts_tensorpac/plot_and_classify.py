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

data_path = '/home/pakitochus/Investigacion/Databases/dislexia/functional_PAC/data/tensorpac_stimuli/'
files = os.listdir(data_path)
electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9',
                            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ',
                            'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10']

clases=['C', 'RD']
stimuli = '20'

data=dict()
for clase in clases:
    fils = [el for el in files if el.endswith(f'{stimuli}_{clase}.npy')]

    arrays = np.zeros((len(fils), 31, 58))

    for ix, fil in enumerate(fils):
        aux = np.load(os.path.join(data_path, fil))
        arrays[ix] = aux
    data[clase] = arrays

nc = len(data['C'])
nd = len(data['RD'])
channel=1
X = np.vstack((data['C'].reshape((nc, -1)), data['RD'].reshape((nd, -1))))
# X = np.vstack((data['C'][:,channel], data['RD'][:,channel]))
# X[X<0] = 0
X = X > 2.5
y = np.r_[np.zeros(nc), np.ones(nd)]
#%% CLASSIFICATION PART
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

prep = StandardScaler()
skf = StratifiedKFold(n_splits=5)

acc=[]
for train, test in skf.split(X,y):
    Xtrain = X[train]
    Xtest = X[test]
    # Xtrain = prep.fit_transform(X[train])
    # Xtest = prep.transform(X[test])
    # Xtest[Xtest<0]=0
    
    pca = PCA(n_components=12)
    Ztrain = pca.fit_transform(Xtrain)
    Ztest = pca.transform(Xtest)
    # Ztrain = Xtrain
    # Ztest = Xtest
    # model = LogisticRegression(class_weight='balanced')
    model = svm.LinearSVC(class_weight='balanced')
    model.fit(Ztrain, y[train])
    yhat = model.predict(Ztest)
    acc.append(model.score(Ztest, y[test]))
    print(yhat)

print(acc)
print(np.mean(acc))
#%%
from sklearn.manifold import SpectralEmbedding

model = SpectralEmbedding(n_components=2)
Z = model.fit_transform(X)
fig, ax = plt.subplots()
ax.scatter(Z[y==0,0], Z[y==0,1])
ax.scatter(Z[y==1,0], Z[y==1,1])

# %%
fig, ax = plt.subplots()
ax.scatter(Ztrain[y[train]==0,2], Ztrain[y[train]==0,1])
ax.scatter(Ztrain[y[train]==1,2], Ztrain[y[train]==1,1])
ax.scatter(Ztest[y[test]==0,2], Ztest[y[test]==0,1])
ax.scatter(Ztest[y[test]==1,2], Ztest[y[test]==1,1])
# %%

#%% PLOTTING PART 
# arrays = model.coef_.reshape((1, 31,-1))
arrays = pca.components_[-2].reshape((1, 31, -1))
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
#fig.savefig(f'{clase}_{stimuli}_{plot}.png')
plt.show()

# %%
