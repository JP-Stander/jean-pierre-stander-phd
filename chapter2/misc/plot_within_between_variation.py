import os
import random
import numpy as np
from matplotlib import pyplot as plt

working_directory = "<enter working directory here>"
os.chdir(working_directory)
from utils import LevelsetMedianSmoother, calculate_fom

'''
This script is a simple example of the strength of an images based on how noisey
the image is.
A simple black and white image with a diagonal edge is generated. On a copy of
this image noise is added systemactically and the FOM with the true image
calculated. The areas on either side of the edge is treated as clusters and
the within and between cluster variation calculated. The FOM is the plotted
against the within, between cluster variation as well as the ratio thereof.
'''

        
#Determine range of white and black areas in image
withins = []
betweens = []
ratios = []
FOMs = []
whites = [(180,255),(150, 210),(120,180),(0,255)]
blacks = [(0,50), (20, 100), (50, 150), (0, 255)]

#Create images of various size (optional to put more than 1 size in list)
for img_size in [32]: #, 64, 128, 256
    img = np.zeros((img_size, img_size))
    for i in range(img.shape[0]):
        for j in np.arange(i, img.shape[1]):
            img[i, j] = 255
            
    img_gold = img.copy()
    
    cluster1_idx = []
    cluster2_idx = []
    
    #Detemine true clusters
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i <= j:
                img[i,j] = 255
                cluster1_idx.append((i,j))
            else:
                cluster2_idx.append((i,j))
                
    noise_perc = 0.05
    noise_per_step = int(noise_perc*img.shape[0]*img.shape[1]/2)
                
    for white, black in zip(whites, blacks):
        all_idx = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                all_idx += [(i,j)]
        print('Number of idxs')
        print(len(all_idx))
        for _ in range(img_size):   
            if len(all_idx) == 0:
                break
            noise_idx = random.choices(all_idx,k=noise_per_step)
            for idx in noise_idx:
                if idx[0] <= idx[1]:
                    # img[idx[0], idx[1]]=np.random.randint(80,235,1)
                    img[idx[0], idx[1]]=np.random.randint(white[0],white[1],1)
                if idx[0] > idx[1]:
                    # img[idx[0], idx[1]]=np.random.randint(15,130,1)        
                    img[idx[0], idx[1]]=np.random.randint(black[0],black[1],1)        
     
            all_idx = list(set(all_idx)-set(noise_idx))#[a for a in all_idx if a not in noise_idx]
            cluster1 = [img[idx] for idx in cluster1_idx]
            cluster2 = [img[idx] for idx in cluster2_idx]
            
            cluster1_mean = np.mean(cluster1)
            cluster2_mean = np.mean(cluster2)
            overall_mean = (cluster1_mean + cluster2_mean)/2
            
            
            within = np.var(cluster1) + np.var(cluster2)
            between = len(cluster1)*(cluster1_mean-overall_mean)**2 + len(cluster2)*(cluster2_mean-overall_mean)**2
            ratio = within/between
            
            img_smooth = LevelsetMedianSmoother(img)
            
            FOMs.append(calculate_fom(img_gold, img_smooth))
            withins.append(within)
            betweens.append(between)
            ratios.append(ratio)
        

stats1 = np.array([FOMs, withins, betweens, ratios]).T

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

for i, name in enumerate(['Withins', 'Betweens', 'Ratios']):
    if i==2:
        stats = stats1[stats1[:,3]<0.001,]
    else:
        stats = stats1
    stats = stats[stats[:, i+1].argsort()]
    plt.figure()
    plt.xlabel(name[:-1])
    plt.ylabel('FOM')
    plt.plot(stats[:,i+1],stats[:,0])
    plt.show()
