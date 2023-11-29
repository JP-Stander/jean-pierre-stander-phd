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


# This function is required for the LULU median smoother function
def _find_neighbours(c, nmax, N, M, connectivity=4):
    '''
    Determines all the neighbours of a set of pixels

    Parameters
    ----------
    c : list
        Set of pixels which neighbourhood needs to be determined.
    nmax : int
        Maximum number of neighbours of each element in each direction.
    N : int
        Height of the image.
    M : int
        Width of the image.
    connectivity : int, optional
        The connectivity to use (4 or 8 connectivity). The default is 4.

    Returns
    -------
    c : list
        List of coordinates of all neighbours.

    '''

    w = []
    for i in range(nmax):
        for j in range(len(c)):
            if connectivity==4:
                i1, i2 = c[j]
                if i2-1 >= 0:
                    w.append((i1, i2-1))
                if i2+1 < M:
                    w.append((i1, i2+1))
                if i1-1 >= 0:
                    w.append((i1-1, i2))
                if i1+1 < N:
                    w.append((i1+1, i2))
            elif connectivity==8:
                i1, i2 = c[j]
                for y_chng in [-1,0,1]:
                    for x_chng in [-1,0,1]:
                        if i1+y_chng >= 0 and i1+y_chng < M:
                            if i2+x_chng >= 0 and i2+x_chng < N:
                                w.append((i1+y_chng, i2+x_chng))
            
        
        c = [a for a in pd.unique(c+w)]
    return c
                
def LevelsetMedianSmoother(image, pmax=3, nmax=2, pmin=1, connectivity=8,
                           use_og=False):
    '''
    Function to apply adaptive median smoother using level sets.
    Parameters
    ----------
    image : numpy array
        The greyscale images to be smooted.
    pmax : int, optional
        Maximum size of level sets to be smoothed. The default is 3.
    nmax : int, optional
        Maximum neighbourhood size. The default is 2.
    pmin : int, optional
        Minimum size of level sets to be smoothed. The default is 1.
    connectivity : int, optional
        The connectivity to use to determine neighbours(4 or 8 connectivity). 
        The default is 8.
    use_og : boolean, optional
        Wheter the original image should be used to determine smoothed value or
        the smooted image up to the current point, if set to False the smoothed
        image to this point is used. The default is False.

    Returns
    -------
    img : numpy array
        The smoothed image.

    '''
    
    img = image.copy()
    ref_img = img.copy()
    
    N, M = img.shape
    level_sets = measure.label(img+1,connectivity=1) 
    set_sizes = pd.value_counts(level_sets.flatten())
    
    
    for p in np.arange(pmax,pmin-1,-1):
        req_set_sizes = set_sizes.iloc[(set_sizes == p).values].index
        for set_label in req_set_sizes:
            n = 1
            while n < nmax:
                set_index = list(map(tuple, np.asarray(np.where(level_sets==set_label)).T.tolist()))
                set_value = img[set_index[0]]
                neighbours = _find_neighbours(set_index, n, N, M, connectivity)

                neighbour_values = [ref_img[idx] for idx in neighbours]
                median = stats.median(neighbour_values)
                replace_value = stats.median(neighbour_values)

                
                minimum = min(neighbour_values)
                maximum = max(neighbour_values)
                
                if minimum < median and median < maximum:
                    if not(minimum < set_value and set_value < maximum):
                        for idx in set_index:
                            img[idx] = replace_value
                            
                    n = nmax
                else:
                    for idx in set_index:
                        img[idx] = replace_value
                    n += 1
            if use_og is False:
                ref_img = img.copy()
    return img


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

def AdaptiveMedianSmoother(img, s=3, sMax=7):
    '''
    Smooths and image using the adaptive median approach
    Parameters
    ----------
    img : numpy array
        The image to be smoothed.
    s : int, optional
        Window size. The default is 3.
    sMax : int, optional
        Maximum neighbourhood size. The default is 7.

    Raises
    ------
    Exception
        Only gray scale images (single channel images) are supported.

    Returns
    -------
    numpy array
        The smoothed image.

    '''

    if len(img.shape) == 3:
        raise Exception ("Single channel image only")

    H,W = img.shape
    a = sMax//2
    padded_img = _padding(img,a)

    f_img = np.zeros(padded_img.shape)

    for i in range(a,H+a+1):
        for j in range(a,W+a+1):
            value = _Lvl_A(padded_img, i, j, s, sMax)
            f_img[i,j] = value

    return f_img[a:-a,a:-a] 

def _Lvl_A(mat, x, y, s, sMax):

    window = mat[x-(s//2):x+(s//2)+1,y-(s//2):y+(s//2)+1]
    Zmin = np.min(window)
    Zmed = np.median(window)
    Zmax = np.max(window)

    A1 = Zmed - Zmin
    A2 = Zmed - Zmax

    if A1 > 0 and A2 < 0:
        return _Lvl_B(window)
    else:
        s += 2 
        if s <= sMax:
            return _Lvl_A(mat,x,y,s,sMax)
        else:
             return Zmed

def _Lvl_B(window):

    h,w = window.shape
    Zmin = np.min(window)
    Zmed = np.median(window)
    Zmax = np.max(window)

    Zxy = window[h//2,w//2]
    B1 = Zxy - Zmin
    B2 = Zxy - Zmax

    if B1 > 0 and B2 < 0 :
        return Zxy
    else:
        return Zmed

    
    
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
    
    
    