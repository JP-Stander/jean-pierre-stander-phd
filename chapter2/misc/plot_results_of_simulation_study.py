import os
import json
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

'''
This script plots the results from the main simulation study. The results were
saved in json format in the folder results. This script searches for all json files
in the folder results. All the results are the added to a table and finally 
plotted by noise/smoother type
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

sp_p = [a for a in list(Results) if '_sp' in a and 'psnr' in a and 'og' not in a]
sp_f = [a for a in list(Results) if '_sp' in a and 'fom' in a  and 'og' not in a]
ga_p = [a for a in list(Results) if '_ga' in a and 'psnr' in a and 'og' not in a]
ga_f = [a for a in list(Results) if '_ga' in a and 'fom' in a  and 'og' not in a]
gu_p = [a for a in list(Results) if '_gu' in a and 'psnr' in a and 'og' not in a]
gu_f = [a for a in list(Results) if '_gu' in a and 'fom' in a  and 'og' not in a]

#Distribution of image noise and level for each algorithm
colors = ['red','green','yellow','blue']
names = ['sp_psnr', 'sp_fom', 'ga_psnr', 'ga_fom', 'gu_psnr', 'gu_fom']
label_name = ['Peak Signal to Noise Ratio', "Pratt's Figure of Merit"]*3

matplotlib.rcParams.update({'font.size': 16})
for j,ds in enumerate([sp_p, sp_f, ga_p, ga_f, gu_p, gu_f]):
    plt.figure()
    for i, name in enumerate(ds):
        if 'ls' in name and '20' in name: col='green'
        if 'ls' in name and '10' in name: col='red'
        if 'am' in name and '20' in name: col='blue'
        if 'am' in name and '10' in name: col='yellow'
        leg = f'{name[:2].upper()}: '
        if name[-4: -2] == 'sp':
            leg += f'S&P {name[-2:]}%'
        if name[-4: -2] == 'gu':
           leg += f'Gumbel(0, {name[-2:]})'
        if name[-4: -2] == 'ga':
            leg += f'Gaussian(0, {name[-2:]})'
        
        sns.distplot(Results[name],hist=False,color=col, label=leg, kde_kws={"shade": True})
    plt.xlabel(label_name[j])
    plt.show()

#histograms off differences
ogs = [a for a in list(Results) if 'og_' in a]

for og in ogs:
    diff_ls = Results['ls'+og[2:]] - Results[og]
    plt.figure()
    plt.hist(diff_ls,edgecolor='black', fill=None, rwidth=0.5)
    plt.show()

    diff_am = Results['am'+og[2:]] - Results[og]
    plt.figure()
    plt.hist(diff_am,edgecolor='black', fill=None, rwidth=0.5)
    plt.show()

#image specific
for noise in ['sp', 'ga', 'gu']:
    diff10 = Results[f'am_psnr_{noise}10'] - Results[f'ls_psnr_{noise}10']
    diff20 = Results[f'am_psnr_{noise}20'] - Results[f'ls_psnr_{noise}20']
    plt.figure()
    sns.distplot(diff10, hist=False, label = f'differnce_{noise}_10', color='red', kde_kws={"shade": True})
    sns.distplot(diff20, hist=False, label = f'differnce_{noise}_20', color='blue', kde_kws={"shade": True})
