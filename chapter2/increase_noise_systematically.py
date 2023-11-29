import numpy as np
from matplotlib import pyplot as plt

'''
This script generates a simple black and white image with a diagonal edge.
The two areas seperated by the edge is then treated as clusters. The within
clusters variation is systematically increase as well as the between cluster
various decreased, the ratio of this is calculted, the image plotted with the
ratio as the title
'''

#Specify the range of the black and white regions in the image
blacks = [(0,20),   (0,30),   (0,40),  (0,50),  (0,60),
          (30,50),  (25,60),  (20,70), (15,80), (10,90),
          (60,80),  (55,90),  (50,100), (45,110), (40,120),
          (90,110), (85,120), (80,130), (75,140),(80,155)]

whites = [(230,255),(220,255),(210,255),(200,255), (190,255),
          (200,225),(190,230),(180,235),(170,240), (160,245),
          (170,195),(160,200),(150,205),(140,210), (130,215),
          (145,165),(135,170),(125,175),(115,180), (100,175)]


for c_idx in range(len(whites)):
    white = whites[c_idx]
    black = blacks[c_idx]
    img = np.random.randint(black[0],black[1],(30,30))
    cluster1_idx = []
    cluster2_idx = []
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i <= j:
                img[i,j] = np.random.randint(white[0],white[1],1)
                cluster1_idx.append((i,j))
            else:
                cluster2_idx.append((i,j))
            
    
    cluster1 = [img[idx] for idx in cluster1_idx]
    cluster2 = [img[idx] for idx in cluster2_idx]
    
    cluster1_mean = np.mean(cluster1)
    cluster2_mean = np.mean(cluster2)
    overall_mean = (cluster1_mean + cluster2_mean)/2
    
    
    within = np.var(cluster1) + np.var(cluster2)
    between = len(cluster1)*(cluster1_mean-overall_mean)**2 + len(cluster2)*(cluster2_mean-overall_mean)**2
    ratio = within/between
    
    plt.figure()
    plt.imshow(img,'gray')
    plt.axis('off')
    plt.title(r'$\beta$'+ f': {round(within,2)}')
    plt.show()
    