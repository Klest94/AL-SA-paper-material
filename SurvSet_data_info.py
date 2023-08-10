# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:01:24 2023

@author:       Klest Dedja
@institution:  KU Leuven
"""
import os
import pandas as pd
import numpy as np
# from SurvSet.data import SurvLoader
# from sksurv.util import Surv
import csv
import os

root_folder = os.getcwd()

original_data_folder = os.path.join(root_folder, "datasets", "original-data")
data_processed_folder = os.path.join(root_folder, "datasets", "processed-data")

files = os.listdir(original_data_folder)

files = [f for f in files if 'simul_' not in f and "NHANES" not in f]

df_info = pd.DataFrame()

for file in files:
    
    df = pd.read_csv(os.path.join(original_data_folder, file))

    if "pid" in df.columns:
        df = df.drop("pid", axis=1, errors="ignore")

    y = df[["time", "event"]]
    X = df[[col for col in df.columns if col not in ["time", "event"]]]

    dataset_name = file
    n_instances = X.shape[0]
    n_covars = X.shape[1]
    censoring_rate = 100-100*y['event'].mean()
    
    df_info = df_info.append({'dataset_name': dataset_name.rpartition('.')[0],
                              'n_instances': n_instances,
                              'n_covars': n_covars,
                              'censoring_rate': censoring_rate},
                             ignore_index=True)

df_info.set_index('dataset_name', inplace=True)

#%%
import matplotlib.pyplot as plt

# Define color mapping for censoring rate
color_map = plt.cm.get_cmap('coolwarm')

plt.figure(figsize=(17, 10), dpi=120)

plt.scatter(df_info['n_covars'], df_info['n_instances'], 
            c=df_info['censoring_rate'], 
            s=60,
            cmap=color_map)

plt.xlabel('N. features (log scale)', fontsize=20)
plt.ylabel('N. observations (log scale)', fontsize=20)
plt.title('Overview of the SurvSet datasets', fontsize=24)

plt.yscale('log')
plt.xscale('log')

xlim = plt.xlim()
new_xlim = (xlim[0], xlim[1]*2)
# Set the updated ylim values
plt.xlim(new_xlim)

# Add colorbar legend
cbar = plt.colorbar()
cbar.set_label('Censoring Rate (%)')

# Add dataset name annotations
# for index, row in df_info.iterrows():
#     plt.annotate(index.strip('.csv'), (row['n_covars'], row['n_instances']), ha='left', va='center')

TOL = 0.18
added_annotations = set()
for index, row in df_info.iterrows():
    skip_annotation = False
    for added_annotation in added_annotations:
        row_dist = row['n_covars']/added_annotation[0]
        col_dist = row['n_instances']/ added_annotation[1]
        if 1-6*TOL < row_dist < 1+6*TOL and 1-TOL < col_dist < 1+TOL:
            skip_annotation = True
            print('skipped: ', index)
    if not skip_annotation:
        plt.annotate(index, (row['n_covars'], row['n_instances']), fontsize=20,
                      ha='left', va='center', annotation_clip=True)
        added_annotations.add((row['n_covars'], row['n_instances']))

# Reverse the y-axis direction
# plt.gca().invert_yaxis()

# Display the plot
plt.savefig("SurvSet_data_plot_some_annotated.jpg")
plt.show()


#%%

# with open('SurvSet_info2.csv', 'w', newline='') as outfile:
    
#     writer = csv.writer(outfile)
#     writer.writerow(['Dataset', 'Size', 'Censoring rate percent'])  # Write the header

    for file in files:
        
        df = pd.read_csv(os.path.join(original_data_folder, file))
    
        if "pid" in df.columns:
            df = df.drop("pid", axis=1, errors="ignore")
    
        y = df[["time", "event"]]
        X = df[[col for col in df.columns if col not in ["time", "event"]]]
    
        dataset_name = file
        dataset_size = X.shape
        censoring_rate = '{:.2f}'.format(100-100*y['event'].mean())
    
        # Write row to CSV
        writer.writerow([dataset_name, dataset_size, censoring_rate])

print('Done')
