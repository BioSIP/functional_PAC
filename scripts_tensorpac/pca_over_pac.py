# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:52:27 2020

@author: gmnico
"""

#Scrip para aplicar PCA sobre los resultados de PAC obtenidos con tensorpac
#Se construye cada matriz con todos los sujetos y para todos los resultados PAC de
#cada combinación de bandas Delta-Theta,...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from sklearn.decomposition import PCA

#Se cargan todos los sujetos para cada estímulo 2Hz, 8Hz y 20Hz
#Para construir la matriz de PCA para cada combinacion de bandas
#Solo para Delta-Gamma
delta_gamma=[]

for filename_s in filename_array:

    npzfile = np.load('pac_tensor_MI_p_'+filename_s[0:10]+'.npz',allow_pickle=True)
    pac_delta_gamma=npzfile['arr_0']
    delta_gamma.append(pac_delta_gamma)
   delta_gamma=np.array(delta_gamma).reshape((np.size(filename_array,0)*10,31))

#En caso de estandarizar
#from sklearn.preprocessing import StandardScaler    
#delta_gamma = StandardScaler().fit_transform(delta_gamma)

#Se aplica PCA
pca=PCA(n_components=0.90,svd_solver = 'full')
#Con todos los instantes temporales
pca_delta_theta=pca.fit_transform(delta_gamma)
#Con solo un instante temporal
#pca_delta_theta=pca.fit_transform(delta_gamma[:,0,:])
print(pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())  
#Se obtiene eigenvectores para tener y representar topoplot de eigenpac    
eigenvalues = pca.explained_variance_
eigenvector = pca.components_

#Se representan los vectores propios
#Poner la misma escala para comparar
#Primera componente
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
im,cn=mne.viz.plot_topomap(eigenvector[0,:],coords_cart[ind_arr,:2],cmap='Spectral_r',
                           extrapolate='head',vmin=np.min(eigenvector[0,:]),vmax=np.max(eigenvector[0,:]),
                           sphere=np.array([0,0,0,110]),
                           sensors=True,names=electrodos,show_names=True,
                           outlines='head')

#ax.set_title('Delta-Gamma 20Hz') 

#Cinco primeras componentes
fig, axs = plt.subplots(1, 5)  
for n in range(0,5):
    im,cn=mne.viz.plot_topomap(eigenvector[n,:],coords_cart[ind_arr,:2],cmap='Spectral_r',
                               extrapolate='head',vmin=np.min(eigenvector[0:5]),vmax=np.max(eigenvector[0:5]),
                               sphere=np.array([0,0,0,110]),axes=axs[n],
                               sensors=True,names=electrodos,show_names=True,
                               outlines='head')
    
fig.suptitle('Delta-Gamma 20Hz')



