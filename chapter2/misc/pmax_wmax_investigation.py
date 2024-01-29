import os
import numpy as np
import pandas as pd
from skimage.util import random_noise
from tqdm import tqdm
working_directory = "<enter working directory here>"
os.chdir(working_directory)
from utils import calculate_fom, LevelsetMedianSmoother, AdaptiveMedianSmoother

'''
This scirpt investigates the effect of pmax for the proposed smoother and wmax
for the adaptive median filter when smoothing a noisey image. A simple image
with a clear edge is generated, three types of noise (salt and pepper, gumbel 
and gaussian) is the added to the image. The image is then smoothed and the
FOM calculated and documented. This experiment is repeated n times which can
be changed in line 28 the mean FOM over then n experiments are calculated and
saved to a file called hyper_parameter_investigation.csv
'''

AM10 = [[] for a in np.arange(2,11,1)]
AM20 = [[] for a in np.arange(2,11,1)]
AM30 = [[] for a in np.arange(2,11,1)]
LS10 = [[] for a in np.arange(2,11,1)]
LS20 = [[] for a in np.arange(2,11,1)]
LS30 = [[] for a in np.arange(2,11,1)]

#Experiment is repeated n times
n = 1
for nn in tqdm(range(n)):
    #Create image
    img = np.ones((100,100))*255
    for i in range(img.shape[0]):
        for j in np.arange(i, img.shape[1]):
            img[i,j] = 0
    #Add noise
    saltpepper_10 = random_noise(img, mode='s&p', amount=0.1)*255
    saltpepper_20 = random_noise(img, mode='s&p', amount=0.2)*255
    saltpepper_30 = random_noise(img, mode='s&p', amount=0.3)*255
    #Smooth image and calculate FOM
    for p in np.arange(2,11,1):
        AM10[p-2].append(calculate_fom(img, AdaptiveMedianSmoother(saltpepper_10, sMax=p)))
        AM20[p-2].append(calculate_fom(img, AdaptiveMedianSmoother(saltpepper_20, sMax=p)))
        AM30[p-2].append(calculate_fom(img, AdaptiveMedianSmoother(saltpepper_30, sMax=p)))
        LS10[p-2].append(calculate_fom(img, LevelsetMedianSmoother(saltpepper_10, pmax=p)))
        LS20[p-2].append(calculate_fom(img, LevelsetMedianSmoother(saltpepper_20, pmax=p)))
        LS30[p-2].append(calculate_fom(img, LevelsetMedianSmoother(saltpepper_30, pmax=p)))
        
#Consolidate results
results = pd.DataFrame([
    [np.array(a).mean() for a in AM10],
    [np.array(a).mean() for a in LS10],
    [np.array(a).mean() for a in AM20],
    [np.array(a).mean() for a in LS20],
    [np.array(a).mean() for a in AM30],
    [np.array(a).mean() for a in LS30]]
    ).T

results.columns = ['AM10','LS10','AM20','LS20','AM30','LS30']
results.to_csv('hyper_parameter_investigation.csv')
