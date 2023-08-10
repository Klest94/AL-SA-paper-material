# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:42:09 2023

@author: u0135479
"""

import pandas as pd
import warnings
import copy
import os
from typing import Union, Optional
from inspect import signature
import threading


from functools import partial

from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import (
    BaseForest,
    _accumulate_prediction,
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
    _parallel_build_trees,
)
from sklearn.tree._tree import DTYPE
from sklearn.utils.validation import check_is_fitted, check_random_state


from sklearn.base import BaseEstimator, ClassifierMixin
from sksurv.ensemble import RandomSurvivalForest
from sksurv.tree import SurvivalTree
from sksurv.functions import StepFunction


def _array_to_step_function(x_times, survival_curves):
    n_samples = survival_curves.shape[0]
    funcs = np.empty(n_samples, dtype=np.object_)
    for i in range(n_samples):
        funcs[i] = StepFunction(x=x_times,
                                y=survival_curves[i])
    return funcs


def _accumulate_conditional_prediction(predict, X, y, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, y)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


class ConditionalSurvivalTree(SurvivalTree):
    
    def __init__(self,
                 splitter="best",
                 max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 epsilon_chf=0,
                 accept_censored_at_0=True,
                 ):
        
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.epsilon_chf = epsilon_chf
        self.accept_censored_at_0 = accept_censored_at_0
        #super().__init__(*args, **kwargs)
    
    def _check_recarray(self, y_array):
        if type(y_array) != np.recarray:
            raise TypeError("y is not a recarray as expected")
            
        if not (y_array.dtype[0] == '?' and y_array.dtype[1] == '<f4'):
            raise ValueError("y recarray 1st field must be boolean, 2nd field float32")
                                     
    
    def _check_no_events(self, y_array):
        
        if np.sum(np.array([i[0] for i in y_array])) > 0:
            raise ValueError(
                'y recarray can contain only censored observations, '
                'found {:d} events'.format(np.sum(np.array([i[0] for i in y_array]))))                                      
        return
    
    
    
    def _find_maximal_index(self,
                           y_pred: np.ndarray,
                           time_events) -> list:
        
        indeces = np.searchsorted(time_events, y_pred, side='left')

        return list(indeces)
    
    def _conditional_info(self, 
                         X_samples: Union[np.ndarray, pd.DataFrame],
                         y_samples: np.ndarray):
        
        self._check_recarray(y_samples)
        self._check_no_events(y_samples)
        
        y_times = np.array([i[1] for i in y_samples])
        
        if self.accept_censored_at_0 == False:
            if min(y_times) == 0.0:
                raise ValueError('set \'accept_censored_at_0\'=True to be able to handle fully non-informative instances')
                
        return self._find_maximal_index(y_times, self.unique_times_)
    
    
    
    def conditional_predict(self, 
                        X_samples: Union[np.ndarray, pd.DataFrame],
                        y_samples: Optional[np.ndarray]=None):
                
        if y_samples is None:
            updated_hazards = self.predict(X_samples)
        
        elif y_samples is not None:    
            indices = self._conditional_info(X_samples, y_samples)
            hazard_matrix = self.predict_cumulative_hazard_function(X_samples,
                                                                    return_array=True)
            
            # CARE: when index = -1, take empty sum, and NOT the last element
            integr_cumul_hazard = [np.sum(row[:d]) if d >= 0 else 0
                                   for row, d in zip(hazard_matrix, indices)]
            
            updated_hazards = self.predict(X_samples) - np.array(integr_cumul_hazard)
            
        return updated_hazards
    
    def predict_conditional_cumulative_hazard(self, 
                                            X_samples: Union[np.ndarray, pd.DataFrame],
                                            y_samples: Optional[np.ndarray]=None,
                                            return_array=True):
        
        hazard_matrix = self.predict_cumulative_hazard_function(X_samples,
                                                                return_array=True)

        if y_samples is None: # nothing to do, respect return_array parameter
            updated_hazards = hazard_matrix
        
        elif y_samples is not None: # call conditional prediction, and have two cases
            indeces = self._conditional_info(X_samples, y_samples)
            ''' we now use _conditional_info to push the cumulated hazard to 0 
            until the observed time point given by the index, so that we can be
            consistent with the observed time-to-event being with certainy greater 
            than the unique_times_[idx]
            CAREFUL: when index = -1, then take 0, and NOT the last element'''
            
            #assumes if return_array = True until the last line:            
            cumul_hazard = [row[d] if d >= 0 else 0 for row, d in zip(hazard_matrix, indeces)]
            chf_matrix = self.predict_chf(X_samples, return_array=True)
            #creates 2-d array out of the 1-d one, replicating the values along the row
            updated_hazards = chf_matrix - np.array(cumul_hazard).reshape(-1, 1)
        
            updated_hazards = np.clip(updated_hazards, a_min=self.epsilon_chf, a_max=None)
            
            if return_array == False: #output StepFunction instead
                updated_hazards = _array_to_step_function(self.clf.unique_times_, updated_hazards)
                
            return updated_hazards
        
        
    def predict_conditional_survival(self, X_samples: Union[np.ndarray, pd.DataFrame],
                                  y_samples: Optional[np.ndarray]=None,
                                  return_array=True):
        
        # consdier using partial(function, return_array=True), shorter code?
        
        if y_samples is None:
            updated_survivals = self.predict_survival_function(X_samples,
                                                               return_array=True)    
        else:
            indeces = self._conditional_info(X_samples, y_samples)
            survival_matrix = self.predict_survival_function(X_samples,
                                                             return_array=True)
            
            cumul_survival = [row[d] if d >= 0 else 1 for row, d in zip(survival_matrix, indeces)]

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                # ignore warning, we treat it in the following lines
                updated_survivals = survival_matrix/(np.array(cumul_survival).reshape(-1, 1))
            
            # replace np.nan's with 0.5, +np.inf are eventually mapped to 1
            updated_survivals = np.where(np.isnan(updated_survivals), 0.5, updated_survivals)
            updated_survivals = np.clip(updated_survivals, a_min=0, a_max=1) # S(t) <=1
        
        if return_array == False: # previously, all was assuming return_array = True
            updated_survivals = _array_to_step_function(self.unique_times_, updated_survivals)
            
        return updated_survivals
    
    
    def expected_conditional_tte(self, X_samples, y_samples, confidence_bound=False):
        
        survival_curves = self.predict_conditional_survival(X_samples, y_samples,
                                                            return_array=False)
        
        time_to_event = np.array([np.trapz(curve.y, curve.x) for curve in survival_curves])
        
        return time_to_event
        
        

class ConditionalRandomSurvivalForest(RandomSurvivalForest, BaseEstimator):
    
    def __init__(self,
                 n_estimators=100,
                 *,
                 max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.0,
                 max_features="sqrt",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None,
                 epsilon_chf=0,
                 accept_censored_at_0=True
                 ):
        
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )
        
        self.epsilon_chf = epsilon_chf
        self.accept_censored_at_0=accept_censored_at_0
        self.estimator = ConditionalSurvivalTree()
        
    def _predict(self, predict_fn, X, y=None):
        
        check_is_fitted(self, "estimators_")
        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if predict_fn in ("predict", "conditional_predict"):
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)

        def _get_fn(est, name):
            fn = getattr(est, name)
            if name in (#"predict_cumulative_hazard_function", 
                        #"predict_survival_function",
                        "predict_conditional_cumulative_hazard",
                        "predict_conditional_survival"):
                fn = partial(fn, return_array=True)
            return fn

        # Parallelising prediction from every (base, underlying) estimator
        lock = threading.Lock()
        if "conditional" in predict_fn:
            Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
                delayed(_accumulate_conditional_prediction)(_get_fn(e, predict_fn), X, y, [y_hat], lock) for e in self.estimators_
            )
        elif predict_fn == 'predict':
            Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
                delayed(_accumulate_prediction)(_get_fn(e, predict_fn), X, [y_hat], lock) for e in self.estimators_
            )
        else:
            KeyError('check prediction method name')

        y_hat /= len(self.estimators_)

        return y_hat
    
    
    def conditional_predict(self, X, y):
        return self._predict("conditional_predict", X, y)
    
    def expected_conditional_tte(self, X, y, with_std=False):
        
        survival_curves = self.predict_conditional_survival(X, y,
                                                            return_array=False)
        
        time_to_event = np.array([np.trapz(curve.y, curve.x) for curve in survival_curves])
        
        if with_std:
            time_to_event_boots = np.zeros([len(X), self.n_estimators])
            for i, tree in enumerate(self.estimators_):
                time_to_event_boots[:,i] = tree.expected_conditional_tte(X.values,
                                                y, confidence_bound=False)
                
            time_to_event_std = np.std(np.array(time_to_event_boots), axis=1)
            df = pd.DataFrame(time_to_event, columns=['avg'])
            df['std'] = time_to_event_std
            return df
        else:
            return time_to_event

    

    def predict_conditional_cumulative_hazard(self, X, y, return_array=True):
        arr = self._predict("predict_conditional_cumulative_hazard", X, y)
        if return_array:
            return arr
        return _array_to_step_function(self.unique_times_, arr)
    

    def predict_conditional_survival(self, X, y, return_array=True):
        arr = self._predict("predict_conditional_survival", X, y)
        if return_array:
            return arr
        return _array_to_step_function(self.unique_times_, arr)

    
    def predict_chf(self, X, y, return_array): #renaming method with shorter name
        return self.predict_conditional_cumulative_hazard(X, y, return_array)
       
    ### MOVED ALL CONDITIONAL PREDICTIONS TO THE SINGLE TREE LEARNER
    ### will inherit from there (TODO: double check it works as expected)
 
       
#%%

''' testing here, comment out when finished'''


# root_folder = os.getcwd()
# data_folder = os.path.join(root_folder, "datasets", "processed-data")
# data_file = os.path.join(data_folder, 'acath')
# df_train = pd.read_csv(os.path.join(data_file, "df_train_fold_1.csv"))[:300]
# df_test = pd.read_csv(os.path.join(data_file, "df_test_fold_1.csv"))[:23]

# y_train = df_train[["event", "time"]]
# X_train = df_train[[col for col in df_train.columns if col not in ["event", "time"]]]        
# y_test = df_test[["event", "time"]]
# X_test = df_test[[col for col in df_test.columns if col not in ["event", "time"]]] 


# event_and_time_col_names = (y_train.columns[0], y_train.columns[1])

# y_train = y_train.astype({'event': 'bool', 'time': 'float32'})
# y_test = y_test.astype({'event': 'bool', 'time': 'float32'})
# y_train = y_train.to_records(index=False) #builds the structured array, needed for RSF
# y_test = y_test.to_records(index=False) #builds the structured array, needed for RSF


# rsf0 = RandomSurvivalForest(n_estimators=20, max_depth=5, random_state=0)
# rsf0.fit(X_train, y_train)

# rsf1 = ConditionalRandomSurvivalForest(n_estimators=20, max_depth=5, 
#                                         random_state=0, accept_censored_at_0=True)
# rsf1.fit(X_train, y_train)

# tr0 = SurvivalTree(max_depth=5, max_features=0.5, random_state=0)
# ctr0 = ConditionalSurvivalTree(max_depth=5, max_features=0.5, accept_censored_at_0=True,
#                     random_state=0)


# # TODO: Extend ConditionalTrees so that it also call on y_test with fully
# # observed events (does it make sense?:) in that case... give step function or smth??

# y_pred_new = rsf1.predict(X_test)

# y_testing = y_test[y_test.event == 0]
# y_testing.time[0] = 0
# X_testing = X_test[y_test.event == 0]

# y_zeros = copy.copy(y_testing)
# y_zeros.time = np.ones(len(y_zeros))*(-0.1)
# y_zeros.event = [False for i in range(len(y_zeros))]

# y_pred0 = rsf1.predict(X_testing)
# y_pred2 = rsf1.conditional_predict(X_testing, y_testing)
# y_pred21 = rsf1.predict_conditional_survival(X_testing, y_testing)

# y_pred3 = rsf1.expected_conditional_tte(X_testing, None)
# y_pred4 = rsf1.expected_conditional_tte(X_testing, y_testing)
# y_pred5 = rsf1.expected_conditional_tte(X_testing, y_testing, with_std=True)

# tte_boots = np.zeros([len(X_testing), rsf1.n_estimators])
# for i, tree in enumerate(rsf1.estimators_):
#     tte_boots[:,i] = tree.expected_time_to_event(X_testing.values,
#                                     y_testing, confidence_bound=False)
    
# ichf_boots = np.zeros([len(X_testing), rsf1.n_estimators])
# for i, tree in enumerate(rsf1.estimators_):
#     ichf_boots[:,i] = tree.conditional_predict(X_testing.values,
#                                     y_testing)
    
# tte_std = np.std(np.array(tte_boots))


# preds0 = [rsf0[i].predict(X_testing.values) for i in range(rsf0.n_estimators)]
# preds1 = [rsf1[i].predict(X_testing.values) for i in range(rsf1.n_estimators)]
# preds2 = [rsf1[i].conditional_predict(X_testing.values, y_testing) for i in range(rsf1.n_estimators)]

# y_pred1 = rsf1.conditional_predict(X_testing, y_testing)
# y_pred2 = np.mean(np.stack(preds2, axis=0), axis=0)


#%%

if __name__ == '__main__':
    # Example values and their corresponding cumulative probabilities
    values = np.array([1, 2, 3, 4, 5])
    cumulative_probabilities = np.array([0.1, 0.3, 0.6, 0.8, 1.0])
    
    def sample_from_cumulative_probability(values, cumulative_probabilities):
        r = np.random.rand()
        index = np.searchsorted(cumulative_probabilities, r)
        return values[index]
    
    # Sample from the cumulative probability distribution
    sample = sample_from_cumulative_probability(values, cumulative_probabilities)
    # print("Sample:", sample)


