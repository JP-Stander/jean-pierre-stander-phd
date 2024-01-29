import os
import csv
import pandas as pd
import numpy as np
from skimage.util import random_noise

working_directory = "<enter working directory here>"
os.chdir(working_directory)
from utils import calculate_fom, LevelsetMedianSmoother

'''
This script investigates the effect of the hyperparameters of proposed smoother
by changing the hyperparameters for various image sizes and then generating an
images, adding noise and then smoothing the image. The size of the images and
hyperparameters are added to the list in line 62 as tuples in the form
(image_size, nmax, pmax)
The function to add and smooth the noise is then called for each combination
and the results saved to a file called hyper_parameter_investigation2.csv in 
the working directory. The script writes to the csv without opening it so it
can be updated in parallel.
'''

def NoiseExample(tup):
    n, nmax, pmax = tup
    img = np.ones((n, n))*255
    for i in range(img.shape[0]):
        for j in np.arange(i, img.shape[1]):
            img[i,j] = 0

    rows_to_add = []
    
    saltpepper_20 = random_noise(img, mode='s&p', amount=0.2)*255    
    result = LevelsetMedianSmoother(saltpepper_20, pmax=pmax, nmax=nmax)
    
    fom = calculate_fom(img, result)
    rows_to_add += [[None, n, 'SaltAndPepper', nmax, pmax, fom]]

    gaussian = np.random.normal(0, 20, img.shape)
    gaussian_20 = np.clip(img + gaussian, 0, 255)
    result = LevelsetMedianSmoother(gaussian_20, pmax=pmax, nmax=nmax)
    
    fom = calculate_fom(img, result)
    rows_to_add += [[None, n, 'Gaussian', nmax, pmax, fom]]
    
    gumbel = np.random.gumbel(0,20, img.shape)
    gumbel_20 = np.clip(img + gumbel, 0, 255)
    result = LevelsetMedianSmoother(gumbel_20, pmax=pmax, nmax=nmax)
    
    fom = calculate_fom(img, result)
    rows_to_add += [[None, n, 'Gumbel', nmax, pmax, fom]]

    with open("hyper_parameter_investigation2.csv", "a") as out_file:
        writer = csv.writer(out_file)
        for row in rows_to_add:
            writer.writerow(row)
    return 1
    

if not os.path.exists('hyper_parameter_investigation2.csv'):
    df = pd.DataFrame(columns = ['ImgSize', 'NoiseType', 'n', 'p', 'FOM'])
    df.to_csv('hyper_parameter_investigation2.csv')

Results = pd.read_csv(r'hyper_parameter_investigation2.csv') 



tuples = [
    (10, 1, 1),
    (10, 1, 3),
    (10, 1, 5),
    (10, 2, 1),
    (10, 2, 3),
    (10, 2, 5)
    ]

[NoiseExample(tup) for tup in tuples]

    
    

