# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:18:19 2023

@author: u0135479
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay


N_SAMPLES = 100
N_TRAIN = 10

names = [
    # "Nearest Neighbors",
    "weak Decision Tree",
    "Random Forest",
]

clfs = [
    # KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=2),
    RandomForestClassifier(max_depth=5, n_estimators=20, max_features=1),
]

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=0),
    make_circles(n_samples=N_SAMPLES, noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
    make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=0),

]

datasets_dict = {}

for ds_cnt, ds in enumerate(datasets):
    
    X, y = ds

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    
    X_train, X_pool, y_train, y_pool = train_test_split(
        X_train, y_train, train_size=N_TRAIN, random_state=0)

    datasets_dict[ds_cnt] = [X_train, X_pool, X_test, y_train, y_pool, y_test]
    
    
#%%
    

strategy = "random"

for clf in clfs:
    clf.fit(X_train, y_train)

rf = clfs[-1]
rf.fit(X_train, y_train)

# in theory, loop along here:
pop_average = rf.predict(X_train).mean()

from Sampling_script import SamplingQuery

# for dataset in datasets_dict.values():
      

ActiveLearn = SamplingQuery(X_train, y_train, X_pool, y_pool) #also y_train, y_pool

X_train, y_train = ActiveLearn.return_train_set()
X_pool, y_pool = ActiveLearn.return_pool_set()

dataset_full = [X_train, X_pool, X_test, y_train, y_pool, y_test]


from utilities import return_snapshot

return_snapshot(dataset_full, clfs, names, the_figsize=(13, 4))


M = X_pool.shape[0] # original pool shape
N_ITER = 0

# for strategy in ["random", "uncertainty", "variance"]: 

# perf_dict = list() # empty list, appending FOLD specific (dictionary) performances here
perf_dict = {}

while X_pool.shape[0] > 0:
    
    for clf in clfs:
        clf.fit(X_train, y_train)

    rf = clfs[-1]
    rf.fit(X_train, y_train)
    # print("N* instances:", X_train.shape)
    current_score = rf.score(X_test, y_test)
    # print("performance: {:.5f}".format(current_score))
    
    try:
        perf_dict[X_train.shape[0]].append(current_score) # append score of the fold
    except KeyError: #fails when key does not exist: must create empty list
        perf_dict[X_train.shape[0]] = []
    
    # perf_dict[X_train.shape[0]] will look like a list of length N_FOLDS in the end
    
    
    if strategy == "random":
        X_queried, y_queried = ActiveLearn.random_sampling_query(random_state=0)

    elif strategy == "uncertainty":
        population_average = rf.predict(X_train).mean()
        X_queried, y_queried = ActiveLearn.uncertainty_sampling_query(rf, population_average)
        
    elif strategy == "variance":
        X_queried, y_queried = ActiveLearn.variance_sampling_query(rf,
                                                        normalize=True)
    else:
        raise KeyError("check vector with scenarios name")

    ActiveLearn.update_train_pool(X_queried, y_queried) # df_queried should also privde the queried label
    ActiveLearn.update_query_pool(X_queried, y_queried) # df_queried should also privde the queried label
    
    X_train, y_train = ActiveLearn.return_train_set()
    X_pool, y_pool = ActiveLearn.return_pool_set()
    
    N_ITER +=1
    
    
    data_current = [X_train, X_pool, X_test, y_train, y_pool, y_test]
    
    if X_pool.shape[0] % 20 == 0:
        
        return_snapshot(data_current, clfs, names,
                        the_figsize=(13, 4), print_iter=N_ITER)
        
        
print("finished")


# perf_dict_fold is now a dict with key: train_size, value: performance (on the fold)

''' perf_dict is a list that stores all performances, each element 
    of the list correspods to one fold ( we will build confidence bounds)
'''        


    
        
    
    

