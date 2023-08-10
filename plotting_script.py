# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:52:58 2023

@author:       Klest Dedja
@institution:  KU Leuven
"""
import os
import numpy as np
import pandas as pd
import copy
import warnings
import matplotlib.pyplot as plt
import pickle

from plot_utilities import plot_trajectories
from utilities import find_matching_substrings, format_plotting_info, return_performance_info

root_folder = os.getcwd()
data_folder = os.path.join(root_folder, "datasets", "processed-data")
plots_folder = os.path.join(root_folder, "preliminary-plots")
query_info_folder  = os.path.join(root_folder, "query-info-data")
store_dicts_folder  = os.path.join(root_folder, "plot-and-perform-info-June")
filename = os.path.join(root_folder, "AUC_results.csv")

MAX_ITER_POST = 200

dnames = os.listdir(data_folder)

match_with = ['Framingham', 'grace', 'NHANES', 'rott2', 'UnempDur', 
              'vlbw']

testing_dnames1 = find_matching_substrings(dnames, match_with)
testing_dnames2 = [df for df in dnames if "my_simul" in df][0::2]
testing_dnames = testing_dnames1 + testing_dnames2

#%%

all_strategies = ['variance', 'old_variance', 'random', 'uncertainty', 'old_uncertainty', 
              'dens_uncertainty', 'dens_variance', 
              'dens_uncertainty_sqrt', 'dens_variance_sqrt',
              'dens_uncertainty_inv_sqrt', 'dens_variance_inv_sqrt']

plot_strategies  = ['variance', 'dens_variance', 'random', 'uncertainty', 'dens_uncertainty']

# all_strategies =  plot_strategies

for data in testing_dnames:
    
    data_file = os.path.join(data_folder, data)
    
    for i in range(5):
        
        df_train = pd.read_csv(os.path.join(data_file, "df_train_fold_"+str(i+1)+".csv"))
        df_test = pd.read_csv(os.path.join(data_file, "df_test_fold_"+str(i+1)+".csv"))

        y_train_all = df_train[["event", "time"]]
        X_train_all = df_train[[col for col in df_train.columns 
                             if col not in ["event", "time"]]]    
        y_test = df_test[["event", "time"]]
        X_test = df_test[[col for col in df_test.columns 
                          if col not in ["event", "time"]]] 
    
    dict_name = str(data) + '_full_info.pkl'

    with open(os.path.join(store_dicts_folder, dict_name), 'rb') as fp:
        dict_info = pickle.load(fp)
        
    # adjust length of lists and stuff in the dict_info dict
    plot_info = format_plotting_info(dict_info, plot_strategies)
    plot_info2 = return_performance_info(plot_info, plot_strategies, MAX_ITER_POST)

    x_axis = plot_info2['rounds']
    
    BASE_FONT = 20
    
    fig, ax = plt.subplots(figsize=(10,6), dpi=80)
    plt.rcParams.update({'font.size': BASE_FONT})

    plot_trajectories(fig, ax, plot_info2, x_axis, plot_strategies,
                      fold_traject=False, 
                      conf_bound=False)
    plt.legend(fontsize=BASE_FONT-4)
    
    ymin, ymax = plt.ylim()
    ymin = ymin - 0.2*(ymax - ymin)
    plt.ylim((ymin,ymax))
    
    plt.xlabel("N rounds")
    plt.ylabel("performance (C-index)")
    
    data_name = copy.copy(data)
    if data_name == "my_simul_data_3":
        data_name = "my_simul_data_2"
    
    data_name = data_name.replace("my_sim", "sim")
    data_name = data_name.replace("_", " ")

    plt.title("data:"+str(data_name)+" max train size:"+str(X_train_all.shape[0]))
    plt.title(str(data_name), fontsize=BASE_FONT+6)
    name = str(data) + "_r" + str(MAX_ITER_POST) + "_partial.png"
    # plt.savefig(os.path.join(plots_folder, name))
    plt.show()