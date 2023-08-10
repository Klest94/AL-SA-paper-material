# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:11:06 2023

@author:       Klest Dedja
@institution:  KU Leuven
"""
import os
import numpy as np
import pandas as pd
import pickle
import copy
import warnings
import matplotlib.pyplot as plt

# import Orange as orange
from scipy import stats
from utilities import format_plotting_info, return_performance_info, return_curve_summary_stats
from utilities import find_matching_substrings

MAX_ITERS = 200
NFOLDS = 5

# table_strategies = ['variance', 'dens_variance', 'random', 'uncertainty', 'dens_uncertainty']
table_strategies = ['random', 'uncertainty', 'variance', 'dens_uncertainty', 'dens_variance']

root_folder = os.getcwd()
store_query_data  = os.path.join(root_folder, "query-info-data")
dnames = os.listdir(store_query_data)
match_with = ['Framingham', 'grace', 'rott2', 'UnempDur', 'vlbw', 'simul']
# NHANES

dnames = find_matching_substrings(dnames, match_with)
dnames = [name.split('_')[0] for name in dnames if 'simul_data_4' not in name]

dnames = find_matching_substrings(dnames, match_with)
dnames = [*set(dnames)]

testing_dnames = dnames#[-1:]

min_bin = 0
max_bin = 8

M = len(table_strategies)

n_rows = 2
n_cols = (M+2)//2


for dname in testing_dnames:
    
    name = dname.split('_')[0]
    
    # Create subplots
    M = len(table_strategies)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(4.5*n_cols, 3.2*n_rows))
    
    exhaust_strat = {}
    
    for i, (ax, strat) in enumerate(zip(axes.ravel(), table_strategies + [None])):
            
        freq_counts = pd.Series(dtype='int32')
        exhaus_counts = pd.Series(dtype='int32')
        
        # final plot with proportion of exhausted, for each strat
    
        if i < (len(axes.ravel()) - 1):  # Check if this is NOT the last subplot
            
            for j in range(NFOLDS):
                
                filename = name + "_f" + str(j) + "_" + strat + ".csv"
                
                df = pd.read_csv(os.path.join(store_query_data, filename),index_col=0)
                freq_counts = pd.concat([freq_counts, df['used']])
                exhaus_counts = pd.concat([exhaus_counts, df['exhausted']])
                # we want to plot the average count across the 5 folds! (density=True)
                
            # Store corss-fold info here (for each strat)
            exhaust_strat[strat] = np.mean(exhaus_counts)
            
            bins = np.arange(min_bin, max_bin+1) - 0.5
    
            ax.hist(freq_counts, density=True, bins=bins, edgecolor="k")
            
            ax.set_xticks(bins[:-1] + 0.5)  # Adding 0.5 to center the ticks
            # Set the font size for the x-ticks and y-ticks
            ax.tick_params(axis='x', labelsize=12)  # Font size for x-ticks
            ax.tick_params(axis='y', labelsize=12)  # Font size for y-ticks
            
            ax.set_xlabel('times queried', fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
            ax.set_title(f'{strat}'.replace("dens_", "density +"), fontsize=16)
            
        else: # i == (len(axes.ravel()) - 1: last suboplot with common stats is plotted
                        
            exh_keys = list(exhaust_strat.keys())
            
            exh_keys = [k.replace("dens_", "density +\n") for k in exh_keys]
            exh_keys = [k.replace("uncertainty", "uncert.") for k in exh_keys]
            # exh_keys = [k.replace("variance", "var") for k in exh_keys]
            
            exh_values = list(exhaust_strat.values())
            ax.bar(exh_keys, exh_values, edgecolor="k")
            ax.set_title('Proportion of exhausted instances', fontsize=16)
            ax.set_xticks(exh_keys)
            ax.set_xticklabels(exh_keys, rotation=90, fontsize=14)
            
    # for row in range(n_rows):
    #     axes[row, 0].set_ylabel('Common Y-axis Label')
            
    fig.suptitle(f'Sampling frequencies for {name} data', fontsize=20, y=0.97)
    plt.tight_layout()
    plt.savefig(os.path.join(root_folder, "final-plots", 'stats-'+str(name)+'.png'))
    plt.show()
    
    print('#'*30)

