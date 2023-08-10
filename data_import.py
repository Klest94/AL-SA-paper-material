# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:52:35 2023

@author: u0135479
"""
import os
import numpy as np
import pandas as pd
from SurvSet.data import SurvLoader
# from sksurv.util import Surv

root_folder = os.getcwd()

store_data_folder = os.path.join(root_folder, "datasets", "original-data")

loader = SurvLoader()
# List of available datasets and meta-info
print(loader.df_ds.head())
# Load dataset and its reference
df, ref = loader.load_dataset(ds_name='ova').values()
# print(df.head())

# senc = Surv()
loader = SurvLoader()
# ds_lst = loader.df_ds[~loader.df_ds['is_td']]['ds'].to_list()  # Remove datasets with time-varying covariates
ds_lst = loader.df_ds['ds'].to_list()  # Remove datasets with time-varying covariates
ds_lst1 = loader.df_ds[~loader.df_ds['is_td']]['ds'].to_list()  # Remove datasets with time-varying covariates

ds_diff = [df for df in ds_lst if df not in ds_lst1]
n_ds = len(ds_lst)

#%%
# holder_cindex = np.zeros([n_ds, 3])
k = 0
for i, ds in enumerate(ds_lst):
    k +=1
    df, ref = loader.load_dataset(ds).values()
    # if df.shape[0] >= 500 and df.shape[0] < 20000:
    print("selected df:", ds, "size:", df.shape)
    
    if df.isnull().sum().sum() > 0:
        print("df: {} has missing values".format(ds))
    
    df.to_csv(os.path.join(store_data_folder, ds + ".csv"), index=False)
print("tot df loaded:", k)
    
        
    
# %%
