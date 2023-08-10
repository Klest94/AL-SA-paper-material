

#%%
import os
import pandas as pd
import warnings
import numpy as np
# from SurvSet.data import SurvLoader
# from sksurv.util import Surv

root_folder = os.getcwd()

original_data_folder = os.path.join(root_folder, "datasets", "original-data")
data_processed_folder = os.path.join(root_folder, "datasets", "processed-data")



#%%

from sklearn.model_selection import StratifiedKFold

# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from preProcessing_class import OHE_Imputer
# Create the OHEIterativeImputer instance
ohe_imputer = OHE_Imputer(threshold_missing=0.2,
                          min_frequency=0.02,
                          add_indicator=True,
                          random_state=0,
                          max_iter=2000,
                          #tol=0.001 # default
                          n_nearest_features=20,
                          handle_unknown='infrequent_if_exist',
                          verbose=0)

for file in files:
        
    df = pd.read_csv(os.path.join(original_data_folder, file))
    
    if "pid" in df.columns:
        df = df.drop("pid", axis=1, errors="ignore")
    
    y = df[["time", "event"]]
    X = df[[col for col in df.columns if col not in ["time", "event"]]]
    
    print('dataset:', file)
    print('size:', X.shape)
    print('censoring rate: {:.2f}'.format(100-100*y['event'].mean()))
    


#%%

                       
    if X.shape[0] < 500: # size too small to be meaningful for Active Learning: skipping.
        print("(skipping dataset:", file, ")")
    
    elif X.shape[0] >= 500:
        data_dir = os.path.join(data_processed_folder , file.split(".")[0])
        dir_exists = os.path.exists(data_dir)
    
        if not dir_exists:
            os.makedirs(data_dir)
           
        y_strat = np.array(y["event"])
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
        inner_folds = kf.split(X, y_strat)
    
        i = 0
        
        for train_index, test_index in inner_folds:
            
            i += 1
            df_train, df_test = df.iloc[train_index,:], df.iloc[test_index,:] # N-1 + 1 split
            X_train0, X_test0 = X.iloc[train_index,:], X.iloc[test_index,:] # N-1 + 1 split
            try:
                y_train, y_test = y[train_index], y[test_index]
            except KeyError:
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
    
            X_train = ohe_imputer.fit_transform(X_train0)
            X_test = ohe_imputer.transform(X_test0)

            
            df_train = pd.concat([X_train.reset_index(drop=True),
                                  y_train.reset_index(drop=True)], axis=1)
            df_train.index = X_train.index
    
            df_test = pd.concat([X_test.reset_index(drop=True),
                                 y_test.reset_index(drop=True)], axis=1)
            df_test.index = X_test.index
            
            assert len(df_test) == len(X_test) # check concatenation successful
    
            if df_train.isnull().sum().sum() + df_train.isnull().sum().sum() > 0:
                ValueError("found missing values in train or test... !")
                
            df_train.to_csv(os.path.join(data_dir, "df_train_fold_"+str(i)+".csv"),index=False)
            df_test.to_csv(os.path.join(data_dir, "df_test_fold_"+str(i)+".csv"),index=False)
            print('completed fold: ', i)
    
            
        print('done with dataset', file)
        print('with size:', df.shape)        