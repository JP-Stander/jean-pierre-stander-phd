import os
import glob
import json
import numpy as np
from PIL import Image
from skimage.util import random_noise

working_directory = "<enter working directory here>"
os.chdir(working_directory)
from utils import calculate_fom, LevelsetMedianSmoother, AdaptiveMedianSmoother, calculate_psnr

'''
This script perform the main simulation study of the article. This script reads all
files in all subdirectories within the folder lfw in the working directory. The images
are read in one for one, three types of noise (salt and pepper, gumbel, gaussian) is
then added to the images and these noisy images smoothed using the proposed smoother
with various parameters and the adaptive median filter. The results are saved per image
in json format in a folder called results in the working directory as well. This can be 
time consuming so by default a maximum of 5 images will be executed. This can be changed
in line 218 if needed.
'''
def add_and_smooth_noise(file):
    
    #Read in image
    img = np.array(Image.open(file).convert('L'))
    
    #10% Salt and Pepper noise
    saltpepper_10 = random_noise(img, mode='s&p', amount=0.1)*255

    saltpepper_10_smoothed1 = LevelsetMedianSmoother(saltpepper_10,10, connectivity=8)
    saltpepper_10_smoothed2 = LevelsetMedianSmoother(saltpepper_10,10, connectivity=8, use_og=True)
    saltpepper_10_smoothed3 = AdaptiveMedianSmoother(saltpepper_10)
    
    sp10_PSNR1 = calculate_psnr(img, saltpepper_10_smoothed1)
    sp10_FOM1 = calculate_fom(img, saltpepper_10_smoothed1)
    
    sp10_PSNR2 = calculate_psnr(img, saltpepper_10_smoothed2)
    sp10_FOM2 = calculate_fom(img, saltpepper_10_smoothed2)
    
    sp10_PSNR3 = calculate_psnr(img, saltpepper_10_smoothed3)
    sp10_FOM3 = calculate_fom(img, saltpepper_10_smoothed3)
    
    sp10_PSNR_OG = calculate_psnr(img, saltpepper_10)
    
    #20% Salt and Pepper noise
    saltpepper_20 = random_noise(img, mode='s&p', amount=0.2)*255
    
    saltpepper_20_smoothed1 = LevelsetMedianSmoother(saltpepper_20, 10, 
                                                     connectivity=8)
    saltpepper_20_smoothed2 = LevelsetMedianSmoother(saltpepper_20, 10, 
                                                     connectivity=8, use_og=True)
    saltpepper_20_smoothed3 = AdaptiveMedianSmoother(saltpepper_20)
    
    sp20_PSNR1 = calculate_psnr(img, saltpepper_20_smoothed1)
    sp20_FOM1 = calculate_fom(img, saltpepper_20_smoothed1)
    
    sp20_PSNR2 = calculate_psnr(img, saltpepper_20_smoothed2)
    sp20_FOM2 = calculate_fom(img, saltpepper_20_smoothed2)
    
    sp20_PSNR3 = calculate_psnr(img, saltpepper_20_smoothed3)
    sp20_FOM3 = calculate_fom(img, saltpepper_20_smoothed3)
    
    sp20_PSNR_OG = calculate_psnr(img, saltpepper_20)
    
    #N(0, 10^2) noise
    gaussian = np.random.normal(0, 10, img.shape)
    gaussian_10 = np.clip(img + gaussian, 0, 255)
    
    gaussian_10_smoothed1 = LevelsetMedianSmoother(gaussian_10, 10, 
                                                     connectivity=8)
    gaussian_10_smoothed2 = LevelsetMedianSmoother(gaussian_10, 10, 
                                                     connectivity=8, use_og=True)
    gaussian_10_smoothed3 = AdaptiveMedianSmoother(gaussian_10)
    
    gs10_PSNR1 = calculate_psnr(img, gaussian_10_smoothed1)
    gs10_FOM1 = calculate_fom(img, gaussian_10_smoothed1)
    
    gs10_PSNR2 = calculate_psnr(img, gaussian_10_smoothed2)
    gs10_FOM2 = calculate_fom(img, gaussian_10_smoothed2)
    
    gs10_PSNR3 = calculate_psnr(img, gaussian_10_smoothed3)
    gs10_FOM3 = calculate_fom(img, gaussian_10_smoothed3)
    
    gs10_PSNR_OG = calculate_psnr(img, gaussian_10)
    
    # #N(0, 20^2) noise
    gaussian = np.random.normal(0, 20, img.shape)
    gaussian_20 = np.clip(img + gaussian, 0, 255)
    
    gaussian_20_smoothed1 = LevelsetMedianSmoother(gaussian_20, 10, 
                                                     connectivity=8)
    gaussian_20_smoothed2 = LevelsetMedianSmoother(gaussian_20, 10, 
                                                     connectivity=8, use_og=True)
    gaussian_20_smoothed3 = AdaptiveMedianSmoother(gaussian_20)
    
    gs20_PSNR1 = calculate_psnr(img, gaussian_20_smoothed1)
    gs20_FOM1 = calculate_fom(img, gaussian_20_smoothed1)
    
    gs20_PSNR2 = calculate_psnr(img, gaussian_20_smoothed2)
    gs20_FOM2 = calculate_fom(img, gaussian_20_smoothed2)
    
    gs20_PSNR3 = calculate_psnr(img, gaussian_20_smoothed3)
    gs20_FOM3 = calculate_fom(img, gaussian_20_smoothed3)
    
    gs20_PSNR_OG = calculate_psnr(img, gaussian_20)
    
    #gumbel(0, 10) noise
    gumbel = np.random.gumbel(0,10,img.shape)
    gumbel_10 = np.clip(img + gumbel, 0, 255)
    
    gumbel_10_smoothed1 = LevelsetMedianSmoother(gumbel_10, 10, 
                                                     connectivity=8)
    gumbel_10_smoothed2 = LevelsetMedianSmoother(gumbel_10, 10, 
                                                     connectivity=8, use_og=True)
    gumbel_10_smoothed3 = AdaptiveMedianSmoother(gumbel_10)
    
    gm10_PSNR1 = calculate_psnr(img, gumbel_10_smoothed1)
    gm10_FOM1 = calculate_fom(img, gumbel_10_smoothed1)
    
    gm10_PSNR2 = calculate_psnr(img, gumbel_10_smoothed2)
    gm10_FOM2 = calculate_fom(img, gumbel_10_smoothed2)
    
    gm10_PSNR3 = calculate_psnr(img, gumbel_10_smoothed3)
    gm10_FOM3 = calculate_fom(img, gumbel_10_smoothed3)
    
    gm10_PSNR_OG = calculate_psnr(img, gumbel_10)
    
    # #gumbel(0, 20) noise
    gumbel = np.random.gumbel(0,20, img.shape)
    gumbel_20 = np.clip(img + gumbel, 0, 255)
    
    gumbel_20_smoothed1 = LevelsetMedianSmoother(gumbel_20, 10, 
                                                     connectivity=8)
    gumbel_20_smoothed2 = LevelsetMedianSmoother(gumbel_20, 10, 
                                                     connectivity=8, use_og=True)
    gumbel_20_smoothed3 = AdaptiveMedianSmoother(gumbel_20)
    
    gm20_PSNR1 = calculate_psnr(img, gumbel_20_smoothed1)
    gm20_FOM1 = calculate_fom(img, gumbel_20_smoothed1)
    
    gm20_PSNR2 = calculate_psnr(img, gumbel_20_smoothed2)
    gm20_FOM2 = calculate_fom(img, gumbel_20_smoothed2)
    
    gm20_PSNR3 = calculate_psnr(img, gumbel_20_smoothed3)
    gm20_FOM3 = calculate_fom(img, gumbel_20_smoothed3)
    
    gm20_PSNR_OG = calculate_psnr(img, gumbel_20)
    
    result = {}
    
    #Salt and Pepper 10
    result['sp10']={}
    result['sp10']['ls_psnr_sp10_8']   = sp10_PSNR1
    result['sp10']['ls_psnr_sp10_8_p'] = sp10_PSNR2
    result['sp10']['am_psnr_sp10']     = sp10_PSNR3
    result['sp10']['og_psnr_sp10']     = sp10_PSNR_OG
    result['sp10']['ls_fom_sp10_8']    = sp10_FOM1
    result['sp10']['ls_fom_sp10_8_p']  = sp10_FOM2
    result['sp10']['am_fom_sp10']      = sp10_FOM3
    #Salt and Pepper 20
    result['sp20']={}
    result['sp20']['ls_psnr_sp20_8']   = sp20_PSNR1
    result['sp20']['ls_psnr_sp20_8_p'] = sp20_PSNR2
    result['sp20']['am_psnr_sp20']     = sp20_PSNR3
    result['sp20']['og_psnr_sp20']     = sp20_PSNR_OG
    result['sp20']['ls_fom_sp20_8']    = sp20_FOM1
    result['sp20']['ls_fom_sp20_8_p']  = sp20_FOM2
    result['sp20']['am_fom_sp20']      = sp20_FOM3
    #Gaussian 10
    result['gs10']={}
    result['gs10']['ls_psnr_gs10_8']   = gs10_PSNR1
    result['gs10']['ls_psnr_gs10_8_p'] = gs10_PSNR2
    result['gs10']['am_psnr_gs10']     = gs10_PSNR3
    result['gs10']['og_psnr_gs10']     = gs10_PSNR_OG
    result['gs10']['ls_fom_gs10_8']    = gs10_FOM1
    result['gs10']['ls_fom_gs10_8_p']  = gs10_FOM2
    result['gs10']['am_fom_gs10']      = gs10_FOM3
    #Gaussian 20
    result['gs20']={}
    result['gs20']['ls_psnr_gs20_8']   = gs20_PSNR1
    result['gs20']['ls_psnr_gs20_8_p'] = gs20_PSNR2
    result['gs20']['am_psnr_gs20']     = gs20_PSNR3
    result['gs20']['og_psnr_gs20']     = gs20_PSNR_OG
    result['gs20']['ls_fom_gs20_8']    = gs20_FOM1
    result['gs20']['ls_fom_gs20_8_p']  = gs20_FOM2
    result['gs20']['am_fom_gs20']      = gs20_FOM3
    #Gumbel 10
    result['gm10']={}
    result['gm10']['ls_psnr_gm10_8']   = gm10_PSNR1
    result['gm10']['ls_psnr_gm10_8_p'] = gm10_PSNR2
    result['gm10']['am_psnr_gm10']     = gm10_PSNR3
    result['gm10']['og_psnr_gm10']     = gm10_PSNR_OG
    result['gm10']['ls_fom_gm10_8']    = gm10_FOM1
    result['gm10']['ls_fom_gm10_8_p']  = gm10_FOM2
    result['gm10']['am_fom_gm10']      = gm10_FOM3
    #Gumbel 20
    result['gm20']={}
    result['gm20']['ls_psnr_gm20_8']   = gm20_PSNR1
    result['gm20']['ls_psnr_gm20_8_p'] = gm20_PSNR2
    result['gm20']['am_psnr_gm20']     = gm20_PSNR3
    result['gm20']['og_psnr_gm20']     = gm20_PSNR_OG
    result['gm20']['ls_fom_gm20_8']    = gm20_FOM1
    result['gm20']['ls_fom_gm20_8_p']  = gm20_FOM2
    result['gm20']['am_fom_gm20']      = gm20_FOM3
 
    fname = file.split('.')[0].split('\\')[-1]
    with open(f'results/{fname}.json', 'w') as fp:
        json.dump(result, fp, indent=4)
        

data_path = 'lfw'
images = []
for folder in glob.glob(data_path+'\*'):
      images += glob.glob(folder+'\*')
      
processed = [a.split('\\')[-1].split('.')[0] for a in glob.glob('results\*')]
# images = [a for a in images if a.split('\\')[-1].split('.')[0] not in processed]
n_images = min(len(images), 5)
images = images[:n_images]

[add_and_smooth_noise(image) for image in images] 

    