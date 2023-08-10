import pandas as pd
import numpy as np
import copy
from typing import Union

from plot_utilities import find_max_length

def find_matching_substrings(list1, list2):
    return [item for item in list1 if any(substring in item for substring in list2)]

def score_or_nan(clf, X_train, y_train, X_test, y_test):
    try: #should never fail
        clf.fit(X_train, y_train)
        current_score = clf.score(X_test, y_test)
        # valid_batch = True # exit while loop: everything is alright
    except: # AxisError, be specific if possible!
        current_score = np.nan
        warnings.warn("Model could not be trained, adding random instances")
    return current_score


def format_plotting_info(base_info, strategies):
    
    # find global maximum number of iterations
    global_max = find_max_length(base_info)
    
    # global_max= 0 # first find max length across all (folds x strategies)
    # for key in base_info.keys():
    #     if 'fold' in key: # make sure keys are folds
    #         lengths = [len(base_info[key][strat]) for strat in base_info[key].keys()]
    #         max_length = np.max(lengths)
    #         global_max = np.max([max_length, global_max])
            
    # extend the non max length parts by appending np.nan, the underlying idea
    # is to extend common x_axis for plotting 

    for key in base_info.keys(): # iterate over all (folds x strategies)
        if 'fold' in key: # make sure keys are folds
            for strategy in strategies: #filling with nans where necessary
                strat_max = len(base_info[key][strategy])
                base_info[key][strategy] = base_info[key][strategy] + (global_max-strat_max)*[np.nan]
    
    return base_info


def return_performance_info(plot_info, strategies, MAX_ITER=200):
    
    base_info = copy.copy(plot_info)
    
    # return only average and std
    base_info['average'] = {} # store here for each strategy (sub dictionary)
    base_info['std_dev'] = {}
    
    max_length = find_max_length(plot_info)
    
    MAX_ITER = np.minimum(MAX_ITER, max_length)

    for strategy in strategies:  
        arrays = [] # store performance array (for each fold x strategy)
        
        for key in base_info.keys(): #loop across folds
            if 'fold' in key:# include round 0 and round MAX_ITER
                arrays.append(np.array(base_info[key][strategy])[:MAX_ITER]) # , dtype=object
                
        # arrays now is a list of length N_FOLDS, of max length MAX_ITER
                
        base_info['average'][strategy] = np.mean(arrays, axis=0)
        base_info['std_dev'][strategy] = np.std(arrays, axis=0)
        # close strategy loop now    
        
    train_perfs = [base_info[key]['full_train'][0] for key in base_info.keys() if 'fold' in key]
    
    base_info['rounds'] = np.arange(MAX_ITER)
    
    base_info['average']['full_train'] = np.mean(np.array(train_perfs))
    base_info['std_dev']['full_train'] = np.std(np.array(train_perfs))

    return base_info


def return_curve_summary_folds(plot_info, strategies):
    
    base_info = copy.copy(plot_info)

    for strategy in strategies:
        
        base_info['average'][strategy+'_avg'] = np.mean(base_info['average'][strategy])
        base_info['std_dev'][strategy+'_std'] = np.mean(base_info['std_dev'][strategy])



def return_curve_summary_stats(plot_info, strategies):
    
    base_info = copy.copy(plot_info)

    for strategy in strategies:
        
        base_info['average'][strategy+'_avg'] = np.mean(base_info['average'][strategy])
        base_info['std_dev'][strategy+'_std'] = np.mean(base_info['std_dev'][strategy])
        del  base_info['average'][strategy], base_info['std_dev'][strategy]
        
    base_info['std_dev']['full_train_std'] = np.mean(base_info['std_dev']['full_train'])
    base_info['average']['full_train_avg'] = np.mean(base_info['average']['full_train'])
    del base_info['average']['full_train'], base_info['std_dev']['full_train']

    return base_info    
    
            
        
def ys_to_recarray(*list_of_dfs, 
                         output_dtypes={'event': 'bool', 'time': 'float32'},
                         verbose=0):
    
    recarrays = []

    for df in list_of_dfs:
        if isinstance(df, pd.DataFrame):
            df = df.astype(output_dtypes)
            recarray = df.to_records(index=False)
            recarrays.append(recarray)
            
        elif isinstance(df, np.recarray):
            if verbose > 0:
                warnings.warn('found np.recarray already, adding it as it is')
            recarrays.append(df)
        else:
            raise ValueError('pd DataFrames are accepted only')
    
    return recarrays

def filter_dictionary(dictionary, strings=['_avg', '_std']):
    
    filtered_dict = {}
    substrings = strings

    for key in dictionary.keys():
        if any(substring in key for substring in substrings):
            filtered_dict[key] = dictionary[key]

    return filtered_dict
    

def init_pool_sets(X_train: pd.DataFrame,
                   X_mask: pd.DataFrame,
                   y_train: pd.DataFrame,
                   y_mask: pd.DataFrame) -> tuple[pd.DataFrame,
                                                  pd.DataFrame,
                                                  pd.DataFrame]:
    
    X_pool = pd.concat([X_train, X_mask], axis=0)
    y_pool = pd.concat([y_train, y_mask], axis=0)
    
    # make sure stypes are correct for z_mask and z_pool
    y_train = y_train.astype({'event': 'bool', 'time': 'float32'})

    z_mask = np.zeros(y_mask.shape[0], dtype=[('event', 'bool'), ('time', 'float32')])
    z_pool = pd.concat([y_train, pd.DataFrame(z_mask, columns=y_train.columns)],
                       axis=0).to_records(index=False)

    return X_pool, y_pool, z_pool #y_train = z_train


def upsample_obs_events(X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    X_pool: pd.DataFrame,
                    y_pool: pd.DataFrame,
                    min_events: Union[int,None],
                    random_state: Union[int,None]=None) -> tuple[pd.DataFrame,
                                                                 pd.DataFrame,
                                                                 pd.DataFrame,
                                                                 pd.DataFrame]:
    
    import random
    random.seed(a=random_state) #overrides random seed within random library
    
    if isinstance(y_train, np.recarray):
        y_train = pd.DataFrame(y_train)
        assert y_train.shape[1] == 2
            
    if isinstance(y_pool, np.recarray):
        y_pool = pd.DataFrame(y_pool)
        assert y_pool.shape[1] == 2
            
    
    if min_events is None: #do nothing, no min_events constraint...
        return X_train, y_train, X_pool, y_pool
        
    if min_events < 0:
        raise ValueError('min_events must be >= 0, found {:d}'.format(min_events))
    
    y_pool_obs = pd.DataFrame.from_records(y_pool)
    y_pool_obs = y_pool_obs[y_pool_obs["event"] == 1]
    
    n_events = y_train["event"].sum()
    n_upsample_events = min_events - n_events
    
    if n_upsample_events >= 1: #else do nothing and return original train/test
    
        # WARNING: original indexes are lost in the y_pool. Use .iloc
        upsample_events = list(random.sample(list(y_pool_obs.index), n_upsample_events))
    
        X_train_upsample = X_pool.iloc[upsample_events, :]
        y_train_upsample = y_pool.iloc[upsample_events, :]
        
        X_train = pd.concat([X_train, X_train_upsample], axis=0)
        y_train = pd.concat([y_train, y_train_upsample], axis=0)
        
        # dropping based on .iloc as well
        X_pool = X_pool.drop(X_pool.index[upsample_events], axis=0, inplace=False)
        y_pool = y_pool.drop(y_pool.index[upsample_events], axis=0, inplace=False)
    else:
        pass #no need to upsample: there are enough events
    
    return X_train, y_train, X_pool, y_pool


def init_round_info(X_train: pd.DataFrame,
                    X_pool: pd.DataFrame,
                    column_names: list=['used', 'exhausted'],
                    start_idx: int=0) -> pd.DataFrame:
    
    ''' assumption is that X_pool contains X_train'''

    Z_query = pd.DataFrame(np.zeros([X_pool.shape[0], 2]),
                          index=X_pool.index,
                          columns=['used', 'exhausted'])
    
    Z_query['rounds'] = np.empty((len(Z_query), 0)).tolist()
    
    
    Z_query.loc[Z_query.index.isin(X_train.index),
                ['used', 'exhausted']] = 1   
    
    Z_query.loc[Z_query.index.isin(X_train.index),
                'rounds'] = Z_query.loc[Z_query.index.isin(X_train.index), 
                                    'rounds'].apply(lambda _: [start_idx])
        
    return Z_query


def init_round_info_old(X_train: pd.DataFrame,
                    X_mask: pd.DataFrame,
                    column_names: list=['used', 'exhausted'],
                    start_idx: int=0) -> pd.DataFrame:

    Z_query = pd.DataFrame(np.zeros([X_mask.shape[0], 2]),
                          index=X_mask.index,
                          columns=['used', 'exhausted'])
    
    Z_query['rounds'] = np.empty((len(Z_query), 0)).tolist()

    
    Z_train_r0 = pd.DataFrame(np.ones([X_train.shape[0], 2]),
                          index=X_train.index,
                          columns=['used', 'exhausted'])
    # Z_train_r0['exhausted'] = np.ones(len(X_train))
    Z_train_r0['rounds'] = [[start_idx] for i in range(len(X_train))]
    
    # merging the info about and X_train and X_mask
    # no distinction from now on.
    Z_query = pd.concat([Z_train_r0, Z_query], axis=0)
    Z_query = Z_query.astype({'used': int, 'exhausted': int})
    
    return Z_query


# def update_round_info(Z_query, n_round, idx_queried)
def count_non_nan(arrays):
    result = []

    for a in arrays:
        non_nan_count = np.sum(~np.isnan(a))
        result.append(non_nan_count)

    return result

def theor_max_queries(array):
    # due to adding np.ones at MAX_ITER+1, we guarantee the presence of a '1'
    return np.argmax(array == 1, axis=1)+1



def append_non_duplicates(a, b, along_column=None, ignore_index=False,
                          return_duplicates_list=True):
    if ((a is not None and type(a) is not pd.core.frame.DataFrame) or (b is not None and type(b) is not pd.core.frame.DataFrame)):
        raise ValueError('a and b must be of type pandas.core.frame.DataFrame.')
    if (a is None):
        return(b)
    if (b is None):
        return(a)
    if(along_column is not None):
        aind = a.iloc[:,along_column].values
        bind = b.iloc[:,along_column].values
    else:
        aind = a.index.values
        bind = b.index.values
    take_rows_b = list(set(bind)-set(aind))
    take_rows_b = [i in take_rows_b for i in bind] #boolean list of non-duplicate entries
    
    return pd.concat([a, b.iloc[take_rows_b,:]], ignore_index=ignore_index), take_rows_b


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
# from sklearn.inspection import DecisionBoundaryDisplay #requires scikit-learn >=1.1





def return_snapshot(data: Union[list, tuple], clfs: list, names: list,
                    the_figsize: tuple=(12, 3.5), print_iter: int=0):
    

    # figure = 
    plt.figure(figsize=the_figsize)
    plt.suptitle("Iteration: "+str(print_iter))
    i = 1

    X_train, X_pool, X_test, y_train, y_pool, y_test = data
    
    X = np.concatenate([X_train, X_pool, X_test], axis=0)

    x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(1, len(clfs) + 1, i)
    ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=cm_bright,
               s=14, edgecolors="k")
    
    ax.scatter(
        X_pool.iloc[:, 0], X_pool.iloc[:, 1], 
        c="w", edgecolors= "k", s=10)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, clfs):
        ax = plt.subplot(1, len(clfs) + 1, i)

        with warnings.catch_warnings(): #FutureWarning for sklearn._neighbors
            warnings.filterwarnings("ignore")
  
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )

        # Plot the training points
        ax.scatter(
            X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=cm_bright, 
            s=12, edgecolors="k"
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1
    
    plt.tight_layout()
    plt.show()

    return

