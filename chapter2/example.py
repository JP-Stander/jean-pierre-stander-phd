import numpy as np
from PIL import Image
from skimage.util import random_noise

from lsamf import level_set_adaptive_median_filter, AdaptiveMedianSmoother

img = np.array(Image.open(file).convert('L'))
    
#10% Salt and Pepper noise
saltpepper_10 = random_noise(img, mode='s&p', amount=0.1)*255

saltpepper_10_smoothed1 = level_set_adaptive_median_filter(saltpepper_10, 10, connectivity=8)
saltpepper_10_smoothed2 = level_set_adaptive_median_filter(saltpepper_10, 10, connectivity=8, use_og=True)
saltpepper_10_smoothed3 = AdaptiveMedianSmoother(saltpepper_10)

