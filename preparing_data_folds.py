'''
preparing and fixing folds here

'''


#%%
import os
import pandas as pd
import numpy as np

root_folder = os.getcwd()

original_data_folder = os.path.join(root_folder, "datasets", "original-data")

data_folder = os.path.join(root_folder, "datasets", "processed-data")

files = os.listdir(original_data_folder)
files = [f for f in files if ".csv" in f][-6:]

for file in files:

    df = pd.read_csv(os.path.join(original_data_folder, file),
                     low_memory=False)  #index=False




#%%



def preparing_data(set_up, datapath, folder, n_folds=5, stratify=True):
    
        
    df1 = pd.read_csv(datapath + folder.split(".csv")[0] +  "/" + "new_train1.csv") 
    df2 = pd.read_csv(datapath + folder.split(".csv")[0] +  "/" + "new_test1.csv") 
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    #old version (until December 8th)
    #df = pd.read_csv(datapath + folder)
    df[df.columns[-2]] = df[df.columns[-2]].astype('bool') # needed for correct recarray
    X = df[df.columns[:-2]].astype('float64') #astype apparently needed
    
    y = df[df.columns[-2:]]#.to_numpy()
    y = y.to_records(index=False) #builds the structured array, needed for RSF
    y_strat = np.array([i[0] for i in y])
        

        
    if stratify == True:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        inner_folds = kf.split(X, y_strat)
        
    elif stratify == False:      
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        inner_folds = kf.split(X, y)
        
    assert X.isnull().sum().sum() == 0 # cannot handle missing vlaues for now

    return X, y, stratify, inner_folds
# %%
