import os
import json
import glob
import numpy as np
import pandas as pd

'''
This script summarises the results from the main in tabular form. The results were
saved in json format in the folder results. This script searches for all json files
in the folder results. All the results are the added to a table and finally 
summarised by noise/smoother type
'''

working_directory = "<enter working directory here>"

os.chdir(working_directory)

files = glob.glob(r'results\*')
        
noise_types = ['sp10', 'sp20', 'gs10', 'gs20', 'gm10', 'gm20']
methods = ['ls', 'am']
target = [
 'ls_fom_sp10',
 'ls_fom_sp20',
 'ls_fom_ga10',
 'ls_fom_ga20',
 'ls_fom_gu10',
 'ls_fom_gu20',
 
 'ls_psnr_ga10',
 'ls_psnr_ga20',
 'ls_psnr_gu10',
 'ls_psnr_gu20',
 'ls_psnr_sp10',
 'ls_psnr_sp20',
 
 'am_fom_sp10',
 'am_fom_sp20',
 'am_fom_ga10',
 'am_fom_ga20',
 'am_fom_gu10',
 'am_fom_gu20',
 
 'am_psnr_sp10',
 'am_psnr_sp20',
 'am_psnr_ga10',
 'am_psnr_ga20',
 'am_psnr_gu10',
 'am_psnr_gu20',
 
 'og_psnr_ga10',
 'og_psnr_ga20',
 'og_psnr_gu10',
 'og_psnr_gu20',
 'og_psnr_sp10',
 'og_psnr_sp20']

Results = pd.DataFrame(columns=target)
for file in files:
    obs = pd.DataFrame(data = np.zeros((1,len(target))), columns=target)
    with open(file, 'r') as jf:
        result = json.load(jf)
    for noise in noise_types:
        for key in result[noise].keys():
            if key[-2:] != '_p' and key[:2] != 'og' and key[:2] != 'am':
                continue
            num_char = 12 if 'psnr' in key else 11
            print(key[:num_char].replace('gs', 'ga').replace('gm', 'gu'))
            obs[key[:num_char].replace('gs', 'ga').replace('gm', 'gu')] = result[noise][key]
    Results = pd.concat((Results, obs))

print('######### Mean ##########')    
Results.mean()
print('######### Variance ##########')    
Results.var()