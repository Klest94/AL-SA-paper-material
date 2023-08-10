# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:01:15 2023

@author:       Klest Dedja
@institution:  KU Leuven
"""
import os
import random
import numpy as np
import pandas as pd
import copy
import warnings
import matplotlib.pyplot as plt

from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split

RANDSEED = 0
random.seed(a=RANDSEED) #overrides random seed within random library

VERBOSE = 1

root_folder = os.getcwd()
data_folder = os.path.join(root_folder, "datasets", "processed-data")
plots_folder = os.path.join(root_folder, "preliminary-plots")
query_info_folder  = os.path.join(root_folder, "query-info-data")

dnames = os.listdir(data_folder)
testing_dnames = dnames#[-15:-13]#[::-1]
testing_dnames2 = [df for df in dnames if "my_simul" in df]#[:0]

# testing_dnames = testing_dnames2 + testing_dnames1

REL_TRAIN_SIZES = [0.02, 0.05, 0.1, 0.2] #test different ones, selec most useful threshold
MIN_EVENTS = 3
N_FOLDS = 5

MIN_START = 0.65
MIN_GAIN = 0.02

#%%

full_dict = {}

for data in testing_dnames:
        
    # Reaches the single dataset folder with all the .csv files
    data_file = os.path.join(data_folder, data)
    
    full_perf = []
    init_perf = {}
           
    for i in range(N_FOLDS):
        
        print('++++ Data: {:s}, fold: {:d} ++++'.format(data, i))

        df_train = pd.read_csv(os.path.join(data_file, "df_train_fold_"+str(i+1)+".csv"))
        df_test = pd.read_csv(os.path.join(data_file, "df_test_fold_"+str(i+1)+".csv"))

        y_train_all = df_train[["event", "time"]]
        X_train_all = df_train[[col for col in df_train.columns 
                             if col not in ["event", "time"]]]
        
        y_test = df_test[["event", "time"]]
        X_test = df_test[[col for col in df_test.columns 
                          if col not in ["event", "time"]]]
        
        from utilities import ys_to_recarray

        for j, TRAIN_SIZE in enumerate(REL_TRAIN_SIZES):
            
            if TRAIN_SIZE < 1 and TRAIN_SIZE*X_train_all.shape[0] < 10:
                TRAIN_SIZE = 10
            
            
            full_dict[str(data)+'_size'+str(j)] = {}
        
            X_train, X_mask, y_train, y_mask = train_test_split(X_train_all, y_train_all,
                                                                train_size=TRAIN_SIZE,
                                                                stratify=y_train_all['event'],
                                                                random_state=i)
            
            from utilities import upsample_obs_events
            
            X_train, y_train, X_mask, y_mask = upsample_obs_events(X_train, y_train,
                                                               X_mask, y_mask,
                                                               MIN_EVENTS, #-N-events
                                                               random_state=0)
            
            y_train_all, y_train, y_test = ys_to_recarray(y_train_all,
                                                          y_train, y_test)
            
            if j == 0:
                rsf0 = RandomSurvivalForest(n_estimators=20, max_depth=10, random_state=0)
                rsf0.fit(X_train_all, y_train_all) # full train dataset, upper benchmark            
                full_perf.append(rsf0.score(X_test, y_test))
            
            rsf0.fit(X_train, y_train) # init train dataset
            if 'size_'+str(j) not in init_perf.keys():
                init_perf['size'+str(j)] = []
            try:
                init_perf['size'+str(j)].append(rsf0.score(X_test, y_test))
            except:
                init_perf['size'+str(j)].append(np.nan)
        
            
    avg_full_perf = np.mean(np.array(full_perf))
    
    for key in init_perf: #keys are the size
        
        if np.mean(np.array(init_perf[key])) >= MIN_START:
            print(key+":    {:.4f}".format(np.mean(np.array(init_perf[key]))))
            full_dict[str(data)+'_'+ key]['start'] = [np.round(np.mean(np.array(init_perf[key])), 4),
                                                        True]
        else:
            print(key+":    {:.4f}".format(np.mean(np.array(init_perf[key]))))
            full_dict[str(data)+'_'+ key]['start'] = [np.round(np.mean(np.array(init_perf[key])), 4),
                                                        False]
            
        if avg_full_perf - np.mean(np.array(init_perf[key])) >= MIN_GAIN:
            print('full_train: {:.4f}'.format(avg_full_perf))
            full_dict[str(data)+'_'+ key]['gain'] = [np.round(avg_full_perf - np.mean(np.array(init_perf[key])), 4),
                                                       True]
        else:
            print('full_train: {:.4f}'.format(avg_full_perf))
            full_dict[str(data)+'_'+ key]['gain'] = [np.round(avg_full_perf - np.mean(np.array(init_perf[key])), 4),
                                                       False]
        
#%%

import pickle
# save dictionary to data_filter_info.pkl file
with open('data_filter_full_info.pkl', 'wb') as fp:
    pickle.dump(full_dict, fp)
    print('Dictionary saved successfully to file')
    
    

#%%

with open('data_filter_full_info.pkl', 'rb') as fp:
    full_dict = pickle.load(fp)

filter_dict = {}
    

for key in full_dict.keys():
    if 'start' in full_dict[key]:
        if full_dict[key]['start'][1] == True and full_dict[key]['gain'][1] == True:
            filter_dict[key] = True
#%%

with open('data_filtered.pkl', 'wb') as fp:
    pickle.dump(filter_dict, fp)
    print('Dictionary saved successfully to file')
    
    
with open('data_filtered.pkl', 'rb') as fp:
    dicto2 = pickle.load(fp)
    print('Data filter dictionary')
    print(dicto2)
    
    