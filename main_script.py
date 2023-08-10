# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:35:46 2023

@author: 0135479u
"""

import numpy as np
import pandas as pd
import os, sys
from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from partial_RSF_class import ConditionalRandomSurvivalForest as ConditionalRSF
# import matplotlib.pyplot as plt
import random
import csv
import pickle
from utilities import find_matching_substrings, score_or_nan
RANDSEED = 0
random.seed(a=RANDSEED) #overrides random seed within random library

VERBOSE = 3

root_folder = os.getcwd()
data_folder = os.path.join(root_folder, "datasets", "processed-data")
plots_folder = os.path.join(root_folder, "preliminary-plots")
query_info_folder  = os.path.join(root_folder, "query-info-data")
store_dicts_folder  = os.path.join(root_folder, "plot-and-perform-info")

filename_out = 'AUC_risk'
suff = "_p2"

filename = os.path.join(root_folder, filename_out+ suff +".csv")

dnames = os.listdir(data_folder)
match_with = ['Framingham', 'grace', 'NHANES', 'rott2', 'UnempDur', 'vlbw']

# match_with += ['csl', 'aids', 'hdfail', 'oldmort' ]
# TENTATIVE ADD: prostateSurvival
# DROP: grace

testing_dnames1 = find_matching_substrings(dnames, match_with)[-1:]

testing_dnames2 = [df for df in dnames if "my_simul" in df][:0]

testing_dnames = testing_dnames2 + testing_dnames1

writing_mode = None #'a' to append, 'w' to overwrite, 'None'  for no touching at all

REL_TRAIN_SIZE = 0.05 # equivalent to size1 in the filter_data script

REL_BATCH_SIZE = 0.01 # size: 1% of the pool set (capped at N_TRAIN_MAX)
MIN_EVENTS = 3
N_FOLDS = 5
MAX_ITER = 200 # not interesting above 100 (no need to reach full train)

# memento: Beta is the weight of the abnormality score
BETA_k = np.ones(MAX_ITER)
BETA_sqrt = np.ones(MAX_ITER)*np.sqrt(np.arange(MAX_ITER)/25)
BETA_inv_sqrt = np.ones(MAX_ITER)*np.sqrt(1/(1e-4+np.arange(MAX_ITER)/25))

strategies = ['variance', 'old_variance', 'random', 'uncertainty', 'old_uncertainty',
              'dens_uncertainty', 'dens_variance',
              'sqrt_dens_uncertainty', 'sqrt_dens_variance',
              'inv_sqrt_dens_uncertainty', 'inv_sqrt_dens_variance']

strategies = ['variance', 'dens_variance', 'random', 'uncertainty', 'dens_uncertainty']

strategies = ['risk_dens_variance', 'risk_variance', 'random']

df_colnames = []
for strategy in strategies:
    df_colnames.append(f'{strategy}_avg')
for strategy in strategies:
    df_colnames.append(f'{strategy}_std')

results_df = pd.DataFrame(columns=df_colnames)

if writing_mode != 'a': # re-creating header if overwriting is needed
    with open(filename, 'w', newline="") as f: #initial file.csv setup
        writer = csv.writer(f)
        writer.writerow(["Dataset"] + df_colnames)
        
print('running on datasets:', testing_dnames)

#%%

for data in testing_dnames:
    
    data_file = os.path.join(data_folder, data)
    plot_info = {}
    perform_auc = {}
           
    for i in range(N_FOLDS):
        
        print('+++ Dataset: {:s}, fold: {:d} +++'.format(data, i))
            
        if 'fold_'+ str(i) not in plot_info.keys():
            plot_info['fold_'+ str(i)] = {}

        df_train = pd.read_csv(os.path.join(data_file, "df_train_fold_"+str(i+1)+".csv"))
        df_test = pd.read_csv(os.path.join(data_file, "df_test_fold_"+str(i+1)+".csv"))

        y_train_all = df_train[["event", "time"]]
        X_train_all = df_train[[col for col in df_train.columns 
                             if col not in ["event", "time"]]]    
        y_test = df_test[["event", "time"]]
        X_test = df_test[[col for col in df_test.columns 
                          if col not in ["event", "time"]]]

                        
        X_train, X_mask, y_train, y_mask = train_test_split(X_train_all, y_train_all,
                                                            train_size=REL_TRAIN_SIZE,
                                                            stratify=y_train_all['event'],
                                                            random_state=RANDSEED)
        
        ## guarantee at least MIN_EVENTS truly observed events
        from utilities import upsample_obs_events
        
        X_train, y_train, X_mask, y_mask = upsample_obs_events(X_train, y_train,
                                                           X_mask, y_mask,
                                                           MIN_EVENTS, #-N-events
                                                           random_state=RANDSEED)
        
        BATCH_SIZE = max(round(len(X_mask)*REL_BATCH_SIZE), 1)
        
        if VERBOSE > 2:
            print('initial train size: {:d}, batch size: {:d}'.format(len(X_train), BATCH_SIZE))
            
        '''
        - X_train: initially: fully labeled events. will be enriched
                with fully or partially labeled events with Active Learning
        - X_mask: fully masked events. Initially corresponds to the
                pool dataset (set from whoch to query from)
                
        X_train and X_mask will  be indistinguishable from now on,
        except for the fact that the first instance of X_train ( before 
        active learning rounds) is known to be fully used and therefore
        'exhausted' already. We append this piece of information on Z_query, 
        we furthermore keep track of which samples are 'used' 
        and if so, for how many times.
        '''
        
        query_size = X_train.shape[0] #initialize amount of queried instances
        N_round = 0
        max_rounds = 0 # x-axis for future plots
        
        from utilities import ys_to_recarray, init_pool_sets, init_round_info
        
        X_pool, y_pool, z_pool = init_pool_sets(X_train, X_mask,
                                                         y_train, y_mask)
        
        
        #initialising Z_query, mainly setting initial X_train as 'exhausted' 
        Z_query = init_round_info(X_train, X_pool,
                                  column_names=['used', 'exhausted'],
                                  start_idx=0)
            
        # all pd.DataFrames to recarrays
        y_train_all, y_train, y_pool, y_test = ys_to_recarray(y_train_all,
                                                      y_train, y_pool,
                                                      y_test)
                
        n_learners = 10 if N_FOLDS < 5 else 100 # for debugging set them = 10
        rsf = ConditionalRSF(n_estimators=n_learners, max_depth=10,
                             accept_censored_at_0=True,
                             n_jobs=5,
                             random_state=RANDSEED)
        
        rsf0 = RandomSurvivalForest(n_estimators=n_learners, max_depth=10, 
                                    n_jobs=5,random_state=RANDSEED)
        rsf0.fit(X_train_all, y_train_all) # full train dataset, upper benchmark
        
        del X_mask, y_mask
            
        from Sampling_script import ActiveLearning
        
        # X_train, X_pool, Z_query and similars should be re-initialized
        # at the beginning of every new for loop ( new strategy)
    
        active_learning = ActiveLearning(model=rsf,
                                         X_train=X_train,
                                         z_train=y_train,
                                         X_pool=X_pool,
                                         y_pool=y_pool,
                                         z_pool=z_pool,
                                         pool_info=Z_query,
                                         max_iter=MAX_ITER,
                                         mask_scenario='partial',
                                         reveal_scenario='partial',
                                         batch_size=BATCH_SIZE,
                                         verbose=VERBOSE) #also y_train, y_pool
        
        '''
        The next function guarantees that the amount of revealed information,
        albeit random, does not depend on the queirying order of the different instances.
        '''        
        
        full_reveal_masking = active_learning.create_masking_pattern(n_quantiles=5,
                                                                     p_reveal_all=None)
        
        '''let P be the probability of full reveal (p_reveal_all), then a
        sample is queried on average ~ (1-P)/P^2 times. '''
        
        for strategy in strategies:
            
            if VERBOSE > 0:
                print("Running strategy: {:s}".format(strategy))
            
            # if 'fold'+ str(i)+ '_' + strategy not in R_query.keys():
            #     R_query['fold'+ str(i)+ '_' + strategy] = {}
            
            N_round = 0
            
            # re-setting X_train, X_pool, Z_query, etc to initial conditions
            # stored as attributes of the ActiveLearning class
            
            X_train, z_train = active_learning.get_init_train_set()
            X_pool, y_pool, z_pool = active_learning.get_init_pool_set()
            Z_query = active_learning.get_init_pool_info()
            
            ''' initialising Z_query, mainly setting initial X_train as
            'exhausted' and updating X_pool as a consequence '''
            
            Z_query = init_round_info(X_train, X_pool,
                                      column_names=['used', 'exhausted'],
                                      start_idx=0)
            
            X_pool, z_pool, y_pool = active_learning.update_pool_set(X_pool, 
                                                        z_pool, y_pool,
                                                        Z_query,
                                                        filtering='exhaustion')
            
            # get score at iteration 0 (no active learning yet)
            current_score = score_or_nan(rsf, X_train, y_train, X_test, y_test)
            
            if strategy not in plot_info['fold_'+ str(i)].keys():
                plot_info['fold_'+ str(i)][strategy] = []
            plot_info['fold_'+ str(i)][strategy].append(current_score)
            
            if VERBOSE > 1:
                print("Original pool size (fold {:d}): {:4d}".format(i, len(X_pool)))
            
            while (X_pool.shape[0] > 0) and N_round < MAX_ITER: #in case smth goes wrong
                            
                N_round +=1    
                # all what we *really* need is the iloc_pool_idx
                if strategy == "random":
                    X_queried, z_queried, iloc_pool_idx = active_learning.random_sampling(X_pool, z_pool,
                                                                             random_state=N_round)
        
                elif strategy == "uncertainty":
                    X_queried, z_queried, iloc_pool_idx = active_learning.uncertainty_sampling(X_train, z_train, X_pool,
                                                            z_pool,
                                                            conditional=True)
                    
                elif strategy == "variance":                                        
                    X_queried, z_queried, iloc_pool_idx = active_learning.variance_sampling(X_train, z_train, X_pool,
                                                            z_pool,
                                                            conditional=True)
                
                elif strategy == "old_uncertainty":
                    X_queried, z_queried, iloc_pool_idx = active_learning.uncertainty_sampling(X_train, z_train, X_pool,
                                                            z_pool,
                                                            conditional=False)

                elif strategy == "old_variance":
                    X_queried, z_queried, iloc_pool_idx = active_learning.variance_sampling(X_train, z_train, X_pool,
                                                            z_pool,
                                                            conditional=False)
                    
                    
                elif strategy == "risk_variance":    
                    
                    X_queried, z_queried, iloc_pool_idx = active_learning.info_risk_sampling(X_train,
                                                            z_train, X_pool,
                                                            z_pool,
                                                            "variance",
                                                            conditional=True,
                                                            beta=BETA_k[N_round-1])
                    
                elif strategy == "risk_dens_variance":
                    
                    clf_density = IsolationForest(n_estimators=n_learners,
                                                  random_state=0).fit(X_train)
                    
                    X_queried, z_queried, iloc_pool_idx = active_learning.info_risk_density_sampling(X_train,
                                                            z_train, X_pool,
                                                            z_pool,
                                                            clf_density,
                                                            "variance",
                                                            conditional=True,
                                                            beta1=BETA_k[N_round-1],
                                                            beta2=BETA_k[N_round-1])
        
                    
                elif strategy == "dens_variance":    
                    clf_density = IsolationForest(n_estimators=n_learners,
                                                  random_state=0).fit(X_train)
                    
                    X_queried, z_queried, iloc_pool_idx = active_learning.info_density_sampling(X_train, z_train, X_pool,
                                                            z_pool, clf_density,
                                                            "variance",
                                                            conditional=True,
                                                            beta=BETA_k[N_round-1])
                    
                elif strategy == "dens_uncertainty":
                    clf_density = IsolationForest(n_estimators=n_learners,
                                                  random_state=0).fit(X_train)
          
                    X_queried, z_queried, iloc_pool_idx = active_learning.info_density_sampling(X_train, z_train, X_pool,
                                                            z_pool, clf_density,
                                                            "uncertainty",
                                                            conditional=True,
                                                            beta=BETA_k[N_round-1])
                    
                elif strategy == "sqrt_dens_variance":    
                    clf_density = IsolationForest(n_estimators=n_learners,
                                                  random_state=0).fit(X_train)
                    
                    X_queried, z_queried, iloc_pool_idx = active_learning.info_density_sampling(X_train, z_train, X_pool,
                                                            z_pool, clf_density,
                                                            "variance",
                                                            conditional=True,
                                                            beta=BETA_sqrt[N_round-1])
                    
                elif strategy == "sqrt_dens_uncertainty":
                    pop_average = rsf.predict(X_train).mean()
                    clf_density = IsolationForest(n_estimators=n_learners,
                                                  random_state=0).fit(X_train)
                    
                    X_queried, z_queried, iloc_pool_idx = active_learning.info_density_sampling(X_train, z_train, X_pool,
                                                            z_pool, clf_density,
                                                            "uncertainty",
                                                            conditional=True,
                                                            beta=BETA_sqrt[N_round-1])
                
                elif strategy == "inv_sqrt_dens_variance":    
                    clf_density = IsolationForest(n_estimators=n_learners,
                                                  random_state=0).fit(X_train)
                    
                    X_queried, z_queried, iloc_pool_idx = active_learning.info_density_sampling(X_train, z_train, X_pool,
                                                            z_pool, clf_density,
                                                            "variance",
                                                            conditional=True,
                                                            beta=BETA_inv_sqrt[N_round-1])
                    
                elif strategy == "inv_sqrt_dens_uncertainty":
                    pop_average = rsf.predict(X_train).mean()
                    clf_density = IsolationForest(n_estimators=n_learners,
                                                  random_state=0).fit(X_train)
                    
                    X_queried, z_queried, iloc_pool_idx = active_learning.info_density_sampling(X_train, z_train, X_pool,
                                                            z_pool, clf_density,
                                                            "uncertainty",
                                                            conditional=True,
                                                            beta=BETA_inv_sqrt[N_round-1])
                    
                else:
                    raise KeyError("check vector with scenarios name. Found:", strategy)
                                        
                # R_query['fold'+ str(i)+ '_' + strategy][N_round] = list(X_queried.index)
                
                select_col = Z_query['used'].values
                non_exhausted_mask = Z_query['exhausted'] == 0
                select_col = select_col[non_exhausted_mask].astype(int)

                reveal_masking = full_reveal_masking[non_exhausted_mask.values, :]
                reveal_masking = reveal_masking[np.arange(len(select_col)), select_col]
                
                z_revealed, is_idx_exhaust = active_learning.update_labels_pool(z_pool,
                                                        y_pool, iloc_pool_idx,
                                                        reveal_masking)
                
                Z_query = active_learning.update_round_info(Z_query, X_queried, N_round,
                                                            is_idx_exhaust)  
        
                
                z_revealed, is_idx_exhaust = active_learning.update_labels_pool(z_pool,
                                                        y_pool, iloc_pool_idx,
                                                        reveal_masking)
                
                # update z_pool info in the (remaining) pool set,\
                # this step is regardless of whether they are 'new' and appended 
                # to train, or 'already seen' and staying where they are
        
                z_pool[iloc_pool_idx] = z_revealed
            
                # identify new queries, they will be appended to X and z_train
                
                is_new_query = [idx not in X_train.index for idx in X_queried.index]
                # identify iloc (relative to X_train) of the queires that have already been observed
                iloc_to_update = [X_train.index.get_loc(idx) for idx in X_queried.index if idx in X_train.index]
                
                X_train = pd.concat([X_train, X_queried[is_new_query]], axis=0)
                z_train = np.concatenate((z_train, z_revealed[is_new_query]), axis=0) # also update by appending
                
                # already seen samples are not appended, labels are updated instead
                #if len(iloc_to_update) > 0: #not empty (otherwise there is ValueError)
                if any(x is False for x in is_new_query):
                    # print(z_revealed[[not x for x in is_new_query]])
                    z_train[iloc_to_update] = z_revealed[[not x for x in is_new_query]]
                              
                if N_round % min(MAX_ITER//5, 50) == 0 and VERBOSE > 2:
                    print('round: {:4d}: train: {:4d}, pool: {:4d}'.format(N_round,
                                                            len(X_train), len(X_pool)))
                    
                current_score = score_or_nan(rsf, X_train, z_train, X_test, y_test)
                
                if strategy not in plot_info['fold_'+ str(i)].keys():
                    plot_info['fold_'+ str(i)][strategy] = []
                plot_info['fold_'+ str(i)][strategy].append(current_score)
                
                # update 'admissible' pool set inside the while loop:
                    
                X_pool, z_pool, y_pool = active_learning.update_pool_set(X_pool, 
                                                            z_pool, y_pool,
                                                            Z_query,
                                                            filtering='exhaustion')
                
                query_size += BATCH_SIZE
                ## END while loop here, back to strategy loop
                
            ## store info for plotting, for every strategy   
            if N_round == MAX_ITER:
                print('Max. iterations {:d} reached'.format(N_round))
            
            # TODO: drop: probably useless given later corrections to the dict
            max_rounds = N_round if N_round > max_rounds else max_rounds
            
            from utilities import theor_max_queries
            if VERBOSE > 1:
                theor_tot_masking = theor_max_queries(full_reveal_masking)
                assert (Z_query['used'] <= theor_tot_masking).all()
            
            if writing_mode is not None:
                Z_query.to_csv(os.path.join(query_info_folder,
                            "Rounds_"+ data + "_f"+ str(i)+"_" + strategy + ".csv"))
                
                # pd.DataFrame(R_query).to_csv(os.path.join(query_info_folder,
                #             "Locs_" + data + "_f"+ str(i)+"_" + strategy + ".csv"))
            
            # store fold specific and strategy specific average performance
            # Area under the C-index curve atm. TODO: improve considered 
            # axis for more unbiased estimations (although, this is already OK)
            if strategy not in perform_auc.keys():
                perform_auc[strategy] = [] # create AUC key
            # store fold specific performance for given strategy
            perform_auc[strategy].append(np.mean(plot_info['fold_'+ str(i)][strategy]))
                
            # end of strategy loop (incl. 'while' loop) and re-entering fold loop

        # add list of rounds (common across folds), so to plot on x axis later
        
        # if 'round' not in plot_info['fold_'+ str(i)].keys():
        #     plot_info['fold_'+ str(i)]['round'] = list(np.arange(max_rounds))
    
        if 'full_train' not in plot_info['fold_'+ str(i)].keys():
            plot_info['fold_'+ str(i)]['full_train'] = []
        plot_info['fold_'+ str(i)]['full_train'].append(rsf0.score(X_test, y_test))


    # shouldn't round 0 be the same for everyone? 
    # exiting fold loop, back to data loop
    # compute fold-wise statistics for each stategy:
    for strategy in strategies:
        perform_auc[str(strategy)+'_avg'] = np.mean(np.array(perform_auc[strategy]))
        perform_auc[str(strategy)+'_std'] = np.std(np.array(perform_auc[strategy]))

    # save dictionary to data_filter_info.pkl file
    dict_name = str(data) + '_full_info_'+ filename_out +'.pkl'
    
    if writing_mode is not None:
        with open(os.path.join(store_dicts_folder, dict_name), 'wb') as fp:
            pickle.dump(plot_info, fp)
            
            # pd.DataFrame(R_query).to_csv(os.path.join(query_info_folder,
            #                 "Locs_" + filename_out + '_' + data + "_f"+ str(i) + ".csv"))

    from utilities import filter_dictionary
    
    perform_data = filter_dictionary(perform_auc, strings=['_avg', '_std'])
    perform_data_df = pd.DataFrame([perform_data])
    perform_data_df = perform_data_df[results_df.columns]  # Rearrange columns to match results_df
    results_df = pd.concat([results_df, perform_data_df], ignore_index=True)

    if writing_mode is not None:
        with open(filename, 'a', newline="") as f: #initial file.csv setup
            writer = csv.writer(f)
            writer.writerow([data] + list(perform_data_df.iloc[0]))

            
    # exit dataset loop
#%% manage write and read file
# calculate the average of each column and add it to the row
if writing_mode is not None: #we assume the filename exists
    results_df = pd.read_csv(filename, index_col=0)    
results_df.loc["average"] = results_df.mean(axis=0, numeric_only=True)


average = pd.DataFrame(results_df.loc["average"]).transpose()
if writing_mode is not None:
    average.to_csv(filename, mode="a", header=False, index=True)
    
if writing_mode is not None: #we assume the filename exists
    # Check if "average" already exists in the file 
    with open(filename, 'r') as file:
        lines = file.readlines()
        exists = any('average' in line for line in lines)

    if exists: # If "average" already exists, delete it
        with open(filename, 'w') as file:
            for line in lines:
                if 'average' not in line:
                    file.write(line)

    with open(filename, 'a') as file: # and then append at the end of the file
        average.to_csv(file, header=False, index=True)

if sys.platform == 'win32':
    import winsound
    winsound.Beep(800, 1500)
else:
    print("This code can only run on a Windows machine.")
