# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:16:20 2023

@author: u0135479
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

RANDSEED = 0    


def my_weibull(shape, scale=None, size=None, random_state=None):
    
    """
    generating a two-parameter Weibull distribution
    
    NOTE: given Weibull(a, b) we have:
        - a: shape. a < 1 hazard decreasing over time. a > 1: increasing over time
                    a = 1 becomes exponential distribution
        - b: scale. Simple multiplicative factor.
        
        EXP. VALUE: b * Gamma(1 + 1/a), minimum when a ~ 2 (?)
        VARIANCE: b^2 * [ Gamma(1 + 2/a) - Gamma(1 + 1/a)**2 ]

        
    """
    
    rng = np.random.default_rng(random_state)

    if scale is None:
        scale = 1
    
    if (isinstance(shape, (float, np.ndarray)) and np.any(shape <= 0)) or \
       (isinstance(scale, (float, np.ndarray)) and np.any(scale <= 0)):
        raise ValueError("Parameters must be > 0")

        
    # assert np.shape(scale) == np.shape(size)
    partial = rng.weibull(a=shape, size=size)
    return scale*partial


def sigmoid_transform(x, amplitude, stretch):
    assert stretch > 0 and amplitude > 0
    return amplitude / (1 + np.exp(-x/stretch)) - amplitude/2

def generate_synthetic_data(N, p, q=None, interaction_pairs=None,
                            noise_level=0.1, 
                            scaling_censoring=1, random_state=None):
    
    rng = np.random.default_rng(random_state)

    
    # Generate random covariates, with standard normal distribution
    X = rng.normal(loc=0, scale=1, size=(N, p))

    # Create interaction terms
    if interaction_pairs is not None:
        for i, j in interaction_pairs:
            X[:, i] *= X[:, j]

    # Create a vector of coefficients with varying importance
    beta = rng.uniform(-2, 2, p)
    beta_censor_indep = rng.uniform(-2, 2, p)
    

    # Calculate the linear predictor
    eta = np.dot(X, beta)
    eta = sigmoid_transform(eta, amplitude=5, stretch=2)
        
    # Add (gaussian) noise to the linear predictor
    eta += rng.normal(scale=noise_level*eta.std(), size=N)
    
    # Apply sigmoid transformation to control the range of the linear predictor

    # Set the baseline hazard and calculate the hazard for each observation
    h0 = 0.1
    h = h0 * np.exp(eta)
    
    event_times = my_weibull(1.2, scale=1/h, random_state=random_state)

    # Select q covariates that are in commmon with the event at hand:
    # Generate a random index set for selecting entries from beta
    index_set = rng.choice(range(len(beta)), size=q, replace=False)
    
    # Create a new array by copying selected entries from beta and the remaining entries from beta_censor_indep
    beta_censor = np.empty(len(beta))
    select_indices = index_set
    remain_indices = list(set(range(len(beta))) - set(select_indices))
    beta_censor[select_indices] = beta[select_indices]
    
    flip = rng.choice([-1, 1], size=len(select_indices), p=[0.5, 0.5])    
    beta[select_indices] = beta[select_indices]*flip
    
    beta_censor[remain_indices] = beta_censor_indep[remain_indices]
    
    # Calculate the linear predictor
    eta_censor = np.dot(X, beta_censor)
    # Add (gaussian) noise to the linear predictor
    eta_censor += rng.normal(scale=noise_level, size=N)
    
    # Apply sigmoid transformation to control the range of the linear predictor
    eta_censor = sigmoid_transform(eta_censor, amplitude=5, stretch=2)
    # is the *eta.std() missing ehre?
    
    corr_coeff = np.corrcoef(eta, eta_censor)[0, 1]
    
    plt.scatter(x=eta, y=eta_censor, s=8)
    plt.title(f'Correlation of event and censoring hazards: {corr_coeff:.3f}')
    plt.xlabel('eta event')
    plt.ylabel('eta censor')
    plt.show()
        
    # Calculate the hazard for each observation
    h_censor = 0.1 * np.exp(eta_censor)
    
    pct5 = np.percentile(h_censor, 5)
    pct10 = np.percentile(h_censor, 10)
    # Replace elements in the lowest 5th percentile with twice the amount of the 10th percentile
    h_censor[h_censor < pct5] = 2 * pct10

    # Generate censoring times (scale compared to event times??)
    # censoring_times = np.random.exponential(scale=np.maximum((5 - censoring_dependency * event_times),0))
    censoring_times  = my_weibull(1.1, scale=scaling_censoring/h_censor,
                                  random_state=random_state+1)
    
    # Create a DataFrame to hold the data
    data = pd.DataFrame(X, columns=[f'X{i + 1}' for i in range(p)])
    data['time'] = np.minimum(event_times, censoring_times)
    data['event'] = (event_times <= censoring_times).astype(bool)
    
    df_info = pd.DataFrame()
    df_info['event_time'] = event_times
    df_info['censoring_time'] = censoring_times
    df_info['h_events'] = h
    df_info['h_censor'] = h_censor

    y = data[['time', 'event']]
    
    # Return the data as a recarray
    return data, y.to_records(index=False), df_info


def generate_multivariate_data(N, p, q, interaction_pairs, noise_level,
                               scaling_censoring=2000,
                               event_names=['event1', 'event2', 'event3', 'death'],
                               censoring_event='death'):
    
    df_events = pd.DataFrame()
    i = 0
    for event in event_names:
        i+=1
        df_comp, _, all_info = generate_synthetic_data(N, p, q, interaction_pairs,
                                                       noise_level,
                                                       scaling_censoring,
                                                       random_state=i-1)
        all_info.rename(columns={'event_time': 'time_'+str(event),
                                 'censoring_time': 'censor_'+str(event)},
                        inplace=True)
        df_event = all_info[['time_'+str(event), 'censor_'+str(event)]]
        df_events = pd.concat([df_events, df_event], axis=1)

    df_covars = df_comp[[col for col in df_comp.columns if 'X' in col]]    
    df_tot = pd.concat([df_covars, df_events], axis=1)
    
    df = copy.copy(df_tot)
    # censor_cols = [col for col in df_tot.columns if 'censor_' in col]
        
    for event in event_names:
        df['censor_'+str(event)] = df[['censor_'+str(event),
                                       'time_'+str(censoring_event)]].min(axis=1)
    
    return df


if __name__ == '__main__':
    
    import os
    
    root_folder = os.getcwd()
    
    SAVE_DF = False
    idx = 4
    
    
    if 'augmentation' in root_folder:
        root_folder = os.path.dirname(root_folder) # go up one and make sure we are in the Bellatrex folder
        os.chdir(root_folder)

    store_data = os.path.join(root_folder, 'datasets', 'original-data')
    
    simul_name = "my_simul_data_"+str(idx) + ".csv"
    
    simul_full_name = os.path.join(store_data, simul_name)
    
    # if not os.path.exists(simul_dir):
    #     os.makedirs(simul_dir)
    
    
    N = 1000  # Number of samples
    p = 10    # Number of covariates
    q = 4     # Number of covariates in common bertween event time and censoring time
    interaction_pairs = [(0, 1), (2, 3)]  # Interactions between X1 and X2, and between X3 and X4
    # do not repeast indeces, always i < j
    noise_level = 0.1  # Amount of noise
    # 1: 0.01   # 2: 0.1    # 3: 0.3    # 4: 1
    # flip_p = 0.5
    
    df_surv, ys, all_info = generate_synthetic_data(N, p, q, interaction_pairs,
                                                    noise_level)
    
    df_surv_multi = generate_multivariate_data(2*N, 3*p, 3*q, interaction_pairs, noise_level)
    
    if SAVE_DF:
        df_surv.to_csv(simul_full_name, index=False)


    
    #%%
    
    data_folder = os.path.join(root_folder, "datasets", "processed-data")
    dnames = os.listdir(data_folder)
    # testing_dnames = dnames[:5]
    testing_dnames = [df for df in dnames if "my_simul" in df]
    
    df_perform = pd.DataFrame() # initialize empty daN = 10taframe for performances. Must store 3D array, sort of
    REL_TRAIN_SIZE = 0.10
    N_TRAIN_MAX = 1000
    # with N_TRAIN_MAX >= 300, the real plots (multi-strategy) are plotted, otherwise random stuff comes up (debug mode)
    REL_BATCH_SIZE = 0.01
    MIN_EVENTS = 3
    N_FOLDS = 3
    
    MAX_ITER = 1000
    
    list_strategies = ['uncertainty', 'old_uncertain', 'random', 'old_variance', 'variance_based'] #"variance_based"
    # list_strategies = ["random", "variance_based"]
    
    label_scenario = ['classic', 'masked_once', 'query_window']

    from sksurv.ensemble import RandomSurvivalForest
    from partial_RSF_class import ConditionalRandomSurvivalForest as ConditionalRSF

    
    for data in testing_dnames:
        
        full_train_perf = []
    
        data_file = os.path.join(data_folder, data) #so far correct: reaches the single dataset folder with all the .csv files
        
        print("Starting with dataset: "+str(data))
            
        for i in range(N_FOLDS):
            
            df_train = pd.read_csv(os.path.join(data_file, "df_train_fold_"+str(i+1)+".csv"))[:N_TRAIN_MAX]
            df_test = pd.read_csv(os.path.join(data_file, "df_test_fold_"+str(i+1)+".csv"))[:int(N_TRAIN_MAX//4)]

            y_train0 = df_train[["event", "time"]]
            X_train0 = df_train[[col for col in df_train.columns if col not in ["event", "time"]]]        
            y_test = df_test[["event", "time"]]
            X_test = df_test[[col for col in df_test.columns if col not in ["event", "time"]]] 

            if y_train0.shape[1] == 2: #survival case
    
                event_and_time_col_names = (y_train0.columns[0], y_train0.columns[1])
                
                y_train0 = y_train0.astype({'event': 'bool', 'time': 'float32'})
                y_test = y_test.astype({'event': 'bool', 'time': 'float32'})
                y_train0 = y_train0.to_records(index=False) #builds the structured array, needed for RSF
                y_test = y_test.to_records(index=False) #builds the structured array, needed for RSF
    
            rsf = RandomSurvivalForest(n_estimators=100, max_depth=10,
                                       random_state=0)
            
            rsf.fit(X_train0, y_train0) # full train dataset, upper benchmark
            fold_score = rsf.score(X_test, y_test)
            
            full_train_perf.append(fold_score)
            
            #close fold loop
        
        avg = np.array(full_train_perf).mean()
        std = np.array(full_train_perf).std()
        print("Score: {:.3f} (+- {:.3f})".format(avg, std))
                    
                # rsf = ConditionalRSF(n_estimators=100, max_depth=10,
                #                      accept_censored_at_0=True, 
                #                      random_state=0)
    
        
