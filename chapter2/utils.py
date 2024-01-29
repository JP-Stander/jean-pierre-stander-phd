import cv2
import numpy as np
import pandas as pd
from math import log
import statistics as stats
from skimage import measure
from scipy.ndimage import distance_transform_edt

'''
This script defines all the functions were frequently used for this paper's
results and investigation. All functions that start with an underscore (_) are
helper functions used by other functions
'''

#Peak signal to noise ratio
def calculate_psnr(img, img_smoothed):
    '''
    Calculate the peak signal to noise ratio between two images
    Parameters
    ----------
    img : numpy array
        the true image.
    img_smoothed : numpy array
        the image compared to the truth.

    Raises
    ------
    Exception
        Images should be the same size.

    Returns
    -------
    PSNR : float
        the peak signal to noise ratio.

    '''
    if img.shape != img_smoothed.shape:
        raise Exception ("Image should be the same size")
    n,m = img.shape
    max_t = 255 if img.flatten().max() > 1 else 1
    MSE = ((img - img_smoothed)**2).flatten().sum()/(n*m)
    PSNR = 20*log(max_t,10) - 10*log(MSE,10)
    return PSNR


def _padding(img, pad):
    '''
    Padds image with zeros
    Parameters
    ----------
    img : numpy array
        the image to be padded.
    pad : int
        the number of rows of padding to add.

    Returns
    -------
    padded_img : numpy array
        padded image.

    '''

    padded_img = np.zeros((img.shape[0]+2*pad,img.shape[1]+2*pad))
    padded_img[pad:-pad,pad:-pad] = img
    return padded_img


def calculate_fom(img_original, img_smooth, lower=20, upper=50):
    '''
    Function to calculate the Pratt's Figure of Merit (FOM) between two images
    Parameters
    ----------
    img_original : numpy array
        The reference/true image.
    img_smooth : numpy array
        The smoothed/target image.
    lower : TYPE, optional
        Canny smoothing hyperparameter, lower. The default is 20.
    upper : TYPE, optional
        Canny smoothing hyperparameter, upper. The default is 50.

    Returns
    -------
    fom : float
        The calculated FOM value.

    '''
    
    # Determine where edges are in the original and gold standard
    edges_smooth = cv2.Canny(np.uint8(img_smooth), lower, upper) > 0
    edges_original = cv2.Canny(np.uint8(img_original), lower, upper) > 0
    
    # Calculate the distance from each element to the closest edge element in the original image
    dist = distance_transform_edt(np.invert(edges_original))
    
    # Calculate the number of edge elements in the original and smoothed image
    N_smooth = (edges_smooth).sum()
    N_original = (edges_original).sum()
    
    # Initialize fom quantity
    fom = 0
    
    # Dimensions of image
    N, M = img_original.shape
    
    # Calculating the summation part of the FOM metric
    for i in range(N):
        for j in range(M):
            if edges_smooth[i,j]:
                fom += 1.0/(1.0 + 1/9* dist[i,j]**2)
                
    # Divide by maximum number of edges
    fom /= max(N_smooth, N_original)
    
    return fom
    
    
    