# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:36:06 2023

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
SAVE_OUTPUTS = False
FOLDS = 5

stat_strategies = ['variance', 'dens_variance', 'random', 'uncertainty', 'dens_uncertainty']
stat_strategies = ['random', 'uncertainty', 'variance', 'dens_uncertainty', 'dens_variance']


root_folder = os.getcwd()
store_dicts_folder  = os.path.join(root_folder, "plot-and-perform-info-June")
dnames = os.listdir(store_dicts_folder)
match_with = ['Framingham', 'grace', 'NHANES', 'rott2', 'UnempDur', 'vlbw', 'simul']

# match_with += ['csl', 'aids', 'hdfail', 'oldmort' ]
# TENTATIVE ADD: prostateSurvival
# DROP: grace

testing_dnames = find_matching_substrings(dnames, match_with)

testing_dnames = [name for name in testing_dnames if 'simul_data_4' not in name]

##%% here get tables from stored dicts (hopefully still alive)

df = pd.DataFrame(columns=[f'{elem}' for elem in stat_strategies])

for dname in testing_dnames:
    
    
    with open(os.path.join(store_dicts_folder, dname), 'rb') as f:
        dict_info = pickle.load(f)
        
    table_info = format_plotting_info(dict_info, stat_strategies)
    table_info2 = return_performance_info(table_info, stat_strategies, MAX_ITERS)
    
    for strat in stat_strategies + ['full_train']:
                
        # if strat == 'full_train':    
        #     strat_i = np.mean(table_info2['fold_'+str(i)][strat])
        
        for i in range(FOLDS):
            
            strat_i = np.mean(table_info2['fold_'+str(i)][strat])
            
            dname2 = dname.replace("my_", "").split('_full_info')[0]
            dname2 = dname2.replace("data_3", "data_2")
            
            df.loc[dname2+'_fold'+str(i), strat] = strat_i

df = df.astype(float)

#%%
import Orange as orange

def orange_plot(df, filename, smaller_is_better, save_outputs, exclude=[],
                plot_title="Nemenyi test", reverse_ranks=False,
                dpi_figure=100):
    
    # is ascending: for computing ranks. If true lowest ir ranked 1
    # reverse ranks: if true, greatest average rank is plotted to the left
    # instead of the right
    
    # drop columns that are excluded fro the analysis
    df.drop(exclude, axis=1, errors="ignore", inplace=True) # "overloading" the list, bad practise...
    
    list_friedman = []
    for col in df.columns:
        list_friedman.append(df[col])
    
    # before running post-hoc test, always run Friedman test first:
    Friedman_stat = stats.friedmanchisquare(*list_friedman)
    print('Friedman H0 p-value:', Friedman_stat[1])
    print('WARNING: Friedman test is NOT significant!') if Friedman_stat[1] > 0.05 else print('')
    print("deg. freedom:", Friedman_stat[0])
    
    ranks = df.T.rank(method="dense", ascending=smaller_is_better)
    avg_ranks = ranks.mean(axis=1)

    names = df.columns
    n_samples = df.shape[0]

    ### TODO consider Wilcoxon-Holm post-hoc procedure!

    cd = orange.evaluation.compute_CD(avg_ranks, n_samples)
    
    plot_top= int(min(avg_ranks))
    plot_bott= int(max(avg_ranks))+1
    
    orange.evaluation.graph_ranks(avg_ranks, names, cd=cd,
                                  lowv=plot_top, highv=plot_bott,
                                  width=3.7, textspace=1,
                                  reverse=reverse_ranks)
    plt.rcParams['figure.dpi'] = dpi_figure
    plt.tight_layout()
    plt.title(plot_title)
    figsize = plt.rcParams['figure.figsize']
        
    figsize[1] = figsize[1]*4
    
    plt.rcParams['figure.figsize'] = figsize

    if save_outputs == True: # some setting here (and the few lines above, are off)
        plt.savefig(os.path.join(root_folder, "final-plots", 'Nemenyi.png'))
        
        # plt.savefig(os.path.join(results_folder, SETUP.upper() + "_" + filename + ".pdf")
        #             , bbox_inches='tight', pad_inches=0.1, transparent=True
        #             )

    return plt.show()


# consider runnong it on the AVERAGE perf only!

orange_plot(df, "Performance", smaller_is_better=False, save_outputs=SAVE_OUTPUTS,
            exclude=[],
            plot_title="Performance comparison") # perf: higher is better except (mt)r

