# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:38:52 2023

@author: 0135479u
"""

from typing import Optional, Union
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

''' structure of the code:
    
- class BaseActiveLearning: basic parent class (not much in it), with RANDSEED and little more
- class ActiveLearning # does most of the job

consider separating some tasks (e.g. sampling strategy related tasks) to new classes

'''

class BaseActiveLearning:
    
    import random
    import numpy
    
    RANDSEED = 0
    MASK_KEYS = ["full", "partial"]
    REVEAL_KEYS = ["full", "partial"]
    
    STRATEGY_FULL_LIST = ['variance', 'old_variance',
                          'random', 
                          'uncertainty', 'old_uncertainty', 
                          'dens_uncertainty', 'dens_variance', 
                          'dens_uncertainty_sqrt', 'dens_variance_sqrt',
                          'dens_uncertainty_inv_sqrt', 'dens_variance_inv_sqrt']
    
    def __init__(self,
                 mask_scenario: str = "simple",
                 reveal_scenario: str = "simple",
                 column_names: list = ['event', 'time'],
                 random_state: Optional[int]=None
                 ):

        self.mask_scenario = mask_scenario
        self.reveal_scenario = reveal_scenario
        self.column_names = column_names      
        
        if self.mask_scenario not in self.MASK_KEYS:  # Access SCENARIO_KEYS through class name
            raise KeyError("Masking scenario key not recognized")
            
        if self.reveal_scenario not in self.REVEAL_KEYS:  # Access SCENARIO_KEYS through class name
            raise KeyError("Reveal scenario key not recognized")
            
      

class ActiveLearning(BaseActiveLearning): #sampling strategy, update train and pool sets

    def __init__(self,
                 model,
                 X_train: pd.DataFrame,
                 z_train: np.recarray,
                 X_pool: pd.DataFrame,
                 y_pool: np.recarray,
                 z_pool: np.recarray,
                 pool_info: pd.DataFrame,
                 max_iter: int,
                 mask_scenario=None,
                 reveal_scenario=None,
                 random_state=BaseActiveLearning.RANDSEED,
                 batch_size: int = 1,
                 verbose: int = 0
                 ):
        self.model=model
        self.X_train= X_train
        self.z_train = z_train
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.z_pool = z_pool
        self.pool_info = pool_info
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        
        if not hasattr(model, 'n_estimators'):
            raise ValueError('input model does not seem to be a RSF-like ensemble')
        
            
        if mask_scenario is None:
            mask_scenario = "simple"  # Set default value for mask_scenario
        if reveal_scenario is None:
            reveal_scenario = "simple"  # Set default value for reveal_scenario
            
        super().__init__(mask_scenario, reveal_scenario)
        
    def set_train_set(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return

    def set_pool_set(self, X_pool, y_pool):
        self.X_pool = X_pool
        self.y_pool = y_pool
        return
            
        
    def get_init_train_set(self):
        return self.X_train, self.z_train
    
    def get_init_pool_set(self):
        return self.X_pool, self.y_pool, self.z_pool
    
    def get_init_pool_info(self):
        return self.pool_info
    
    
    def create_masking_pattern(self,
                               n_quantiles: int = 4, 
                               p_reveal_all: float = 0.5,
                               weights=None) -> np.ndarray:
        
        '''
        This function gurantees that the amount of revealed information is a 
        (random) function of the number of times an instance is queried, 
        rather than a function of the number of the current iteration 
        round of the Active Learning step
        '''
    
        if self.max_iter is None:
            print('Setting max. iterations automatically to 10000')
            max_iters = int(1e5)
        else:
            max_iters = int(self.max_iter)
        
        # prepare for MAX_ITER+1 full of np.ones
        full_reveal_masking = np.zeros([len(self.y_pool), max_iters+1])
        
        for k in range(max_iters): 
            full_reveal_masking[:,k] = self.sample_quantiles(n_quantiles=n_quantiles,
                                                             p_reveal_all=p_reveal_all,
                                                             size=len(self.y_pool),
                                                             weights=weights,
                                                             random_state=k)
            
        full_reveal_masking[:,-1] = np.ones(len(self.y_pool)) # MAX_ITER+1 reveal all
    
        return full_reveal_masking
    
        
    def sample_quantiles(self, n_quantiles=5, p_reveal_all=None,
                         size=1, weights=None, random_state=None):
        
        ''' This function samples the amount of 'revealed' information from the
        (queried) labels in the pool set. We sample, for each y_i in y_pool, a 
        number r_i from a distribution of quantiles in the (0, 1] interval, with
                                                     0 excluded and 1 included
        
        Labels assigned to a number 0 < r <= 1 indicate that
        if the original label (T_i, delta_i) is now mapped to 
        (a*T_i, delta_i)
        That is, only in case mask = 1, then the final time to event 
        (with delta 0 or 1), is revealed.
        '''
                
        rng = np.random.default_rng(random_state)
        
        # generate q-ntiles within (0, 1]
        step_size = 1/(n_quantiles)
        quantiles = np.arange(step_size, 1 + step_size, step_size)
        
        if p_reveal_all is None: #default value
            p_reveal_all = 1/n_quantiles
            
        
        if weights is None: #assue they are unform, except for first and last weight in case
            weights = list((1-p_reveal_all)/(n_quantiles-1)*np.ones(n_quantiles-1))\
                + [p_reveal_all]
    
        if len(quantiles) != len(weights):
            raise ValueError("Quantiles and weights must have the same length")
            

        #normalise weights in case they aren't already
        weights = np.array(weights)/np.array(weights).sum() # normalize in case something goes wrong
        
        return rng.choice(quantiles, size=size, replace=True, p=weights)
        
        
        
    def theor_max_queries(arr, guaranteed_one=True):
        result = np.argmax(arr == 1, axis=1)+1
        if not guaranteed_one:
            result[result == 0] = -1 # return -1 if nothing is found

        return result
    
    def tournament_selection(self, list_selected, n_batch, r_ratio,
                             random_state=None):
        
        np.random.seed(random_state)  # Set the random_state for reproducibility
        
        if n_batch > len(list_selected):
            raise ValueError("n_batch should be less or equal to n_selected")
    
        weights = [r_ratio**i for i in range(len(list_selected))]  # create the probability distribution
        norm_factor = sum(weights)  # normalize the distribution
        weights = [w / norm_factor for w in weights]

        # Sample n_batch items without replacement
        batch_list = np.random.choice(list_selected, size=n_batch, replace=False, p=weights)
    
        return batch_list
            
        
    def update_pool_set(self,
                     X_pool: pd.DataFrame,
                     z_pool: np.ndarray,
                     y_pool: np.ndarray,
                     Z_query: pd.DataFrame,
                     filtering: str = 'exhaustion',
                     max_queries_instance: Optional[int]=None)-> tuple[pd.DataFrame, np.ndarray]:
            
        if max_queries_instance is None:
            max_queries_instance = np.inf

        # Z_query has always the original size, whereas X_pool and X_train not
        # cut Z_query accordingly here:
        Z_query = Z_query.loc[X_pool.index] #assumes same indexing (correct?)
        
        if filtering == 'exhaustion':
            mask = Z_query['exhausted'] == 0
        
        elif filtering in ['strictly_once', 'strictly once']:
            mask = Z_query['used'] == 0
            
        elif filtering == 'multiple':
            mask = Z_query['used'] <= max_queries_instance

        else:
            KeyError('Scenario \'{}\' not implemented'.format(filtering))
        
        # gets iloc positions of idx still in X_pool after filtering
        # X_pool.iloc[i] should has corresponding label in z_pool[i], right?
    
        X_pool = X_pool[mask] 
        z_pool = z_pool[mask]
        y_pool = y_pool[mask]

        assert len(X_pool) == len(z_pool) and len(X_pool) == len(y_pool) 
        
        return X_pool, z_pool, y_pool

    
                
    def return_train_set(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.X_train, self.z_train
    
    #output should be DataFrame in the beginning, but then might become Series or single element...
    def return_pool_set(self) -> tuple[Union[pd.DataFrame, pd.Series],
                                       Union[pd.DataFrame, pd.Series]]:
        return self.X_pool, self.y_pool    
    
            
    
    def update_train_set(self, X_train: pd.DataFrame,\
                         z_train: np.ndarray,
                         X_queried: Union[pd.DataFrame, pd.Series],
                         z_revealed: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        ''' given the selected instance(s) (X_queried, z_revealed), update 
        (X_train, z_train) accordingly by or either updateing or appending'''
        
        if isinstance(X_queried, pd.Series): # should not happen, but just in case...
            X_queried = X_queried.to_frame().T
            
        # normally, the list of index with .iloc generates a pd.DataFrame, ready to append
        
        from utilities import append_non_duplicates
        
        #is_new_observed: boolean list of non-duplicate entries
        X_train, is_new_observed = append_non_duplicates(X_train, X_queried, 
                                             along_column=None)
        
        X_loc_duplicates = [X_queried.index[i] for i in range(len(z_revealed)) if 
                           not is_new_observed[i]]
        
        X_iloc_duplicates = [X_train.index.get_loc(loc_index)
                           for loc_index in X_loc_duplicates]
        
        z_revealed_new = z_revealed[is_new_observed]
            
        z_train[X_iloc_duplicates] = z_revealed[~np.array(is_new_observed)]
        
        z_train = np.append(z_train, np.array(z_revealed_new,
                                                dtype=z_train.dtype))
            
        assert len(z_train) == len(X_train)
            
        if (self.verbose > 0 and X_train.shape[0] % 100 == 0) or self.verbose > 1:
            print("updated train pool. Size:", X_train.shape[0])
    
        if self.verbose > 1:
            print("Added queried sample:", X_queried)
            
            
        return X_train, z_train


    
    def return_queried_label(self, y_queried):
        if self.reveal_scenario == 'simple':
            #y_output = y_queried
            print('to be implemented')
            
        if self.reveal_scenario == "partial":
            print('to be implemented')
        return
        
    
    def update_round_info(self, pool_info, X_queried, N_round, exhaust_list):
        
        idx_last_query = list(X_queried.index)
        
        exhaust_list = [idx_last_query[i] for i in range(len(idx_last_query))
                        if exhaust_list[i]]        
        # in the 'simple' scenario, one query leads to full info reveal
        if self.reveal_scenario == 'simple':
            exhaust_list = idx_last_query
            
        # normally, exhaust_list is a subset of idx_last_query
        assert set(exhaust_list) <= set(idx_last_query)
        
        for idx in idx_last_query: #.at faster than .loc in this case (access and set single values)
            pool_info.at[idx, "used"] += 1
            pool_info.at[idx, "rounds"].append(N_round)  # append round
            
            # remember: set(exhaust_list) <= set(idx_last_query)
            if idx in exhaust_list:
                pool_info.at[idx, "exhausted"] = 1
        
        return pool_info


    # querying strategies from here onwards    
    
    def random_sampling(self, X_pool: pd.DataFrame,
                              z_pool: np.ndarray,
                              random_state:Optional[int]=None) -> tuple[pd.DataFrame, np.ndarray]:
        random.seed(a=self.RANDSEED) #overrides random seed within random library
    
        #select random row, output as single raw df or index only or...
        
        # (random) select index based on position (iloc style)
        query_list = list(range(X_pool.shape[0]))
        query_index = random.sample(query_list, #if batch size > X_pool
                                    np.minimum(self.batch_size, len(query_list)))
        # always outputs a list
    
        return X_pool.iloc[query_index], z_pool[query_index], query_index
    
    
    def uncertainty_sampling(self,
                             X_train,
                             z_train, 
                             X_pool: pd.DataFrame,
                             z_pool: np.ndarray,
                             conditional: bool=True) -> tuple[pd.DataFrame, pd.Series]:
        
        clf = self.model.fit(X_train, z_train)
        
        pop_avg = clf.predict(X_train).mean()
        
        if conditional == True:
            proximity_to_avg = np.abs(clf.conditional_predict(X_pool, z_pool) - pop_avg) 
            # the smaller in abs value, the most uncertain (in a way)
        else:
            proximity_to_avg = np.abs(clf.predict(X_pool) - pop_avg)
            # the smaller, the most uncertain (in a way)

        
        # top #batch_size closest to average (smallest L1 distance), output is list
        query_index = np.argsort(proximity_to_avg)[:self.batch_size]
        
        if self.verbose > 3:
            print('sampled ilocs:', query_index)
            print('with proximity:', proximity_to_avg[query_index])

        return X_pool.iloc[query_index], z_pool[query_index], query_index
    
    def variance_sampling(self,
                          X_train,
                          z_train,
                          X_pool: pd.DataFrame,
                          z_pool: np.ndarray,
                          conditional: bool=True) -> tuple[pd.DataFrame, pd.Series]:
        
        
        clf = self.model.fit(X_train, z_train)
        
        train_pop_ranks = np.sort(np.array(clf.predict(X_train)))

        if conditional == True:
            tree_preds = np.array([clf[i].conditional_predict(X_pool.values,
                                                                     z_pool) 
                          for i in range(clf.n_estimators)])
        else:
            tree_preds = np.array([clf[i].predict(X_pool.values) 
                          for i in range(clf.n_estimators)])
        
        # tree_preds = np.transpose(tree_preds)
        ''' tree_preds is an array of shape (n_trees, n_samples) ''' 
        
        # Count instances in train_pop_ranks that are lower or equal to each element in tree_preds
        tree_ranks = np.searchsorted(train_pop_ranks, tree_preds, side='right')
        
        # consider normalizing... probabaly not needed
        uncertainty_ranks = tree_ranks.std(0)

        #argosrt (reversed) to take top elements, output will be a list 
        query_index = np.argsort(uncertainty_ranks)[::-1][:self.batch_size] 

        if self.verbose > 3:
            print('sampled ilocs:', query_index)
            print('with proximity:', uncertainty_ranks[query_index])

        return X_pool.iloc[query_index], z_pool[query_index], query_index


    def alt_variance_sampling(self,
                              X_train,
                              z_train,
                          X_pool: pd.DataFrame,
                          z_pool: np.ndarray,
                          qbc_member,
                          n_members_committee=20,
                          conditional: bool=True) -> tuple[pd.DataFrame, pd.Series]:
        
        clf = self.model.fit(X_train, z_train)
        
        train_pop_ranks = np.sort(np.array(clf.predict(X_train)))
        
        committee_members = [] # list of commitee members (small ConditionalRSF)
        
        for i in range(n_members_committee):
            qbc_member.random_state = i
            print(i)
            qbc_member.fit(X_train, self.z_train)
            committee_members.append(qbc_member)
        
        if conditional == True:
            qbc_preds = np.array([member.conditional_predict(X_pool, z_pool) 
                          for member in committee_members])
        else:
            qbc_preds = np.array([member.predict(X_pool.values) 
                          for member in committee_members])
        
        # tree_preds = np.transpose(tree_preds)
        ''' tree_preds is an array of shape (n_trees, n_samples) ''' 
        
        # Count instances in train_pop_ranks that are lower or equal to each element in tree_preds
        commitee_ranks = np.searchsorted(train_pop_ranks, qbc_preds, side='right')
        
        # consider normalizing... probabaly not needed
        uncertainty_ranks = commitee_ranks.std(0)

        #argosrt (reversed) to take top elements, output will be a list 
        query_index = np.argsort(uncertainty_ranks)[::-1][:self.batch_size] 

        if self.verbose > 3:
            print('sampled ilocs:', query_index)
            print('with proximity:', uncertainty_ranks[query_index])

        return X_pool.iloc[query_index], z_pool[query_index], query_index    
    

    def info_risk_sampling(self,
                            X_train,
                            z_train,
                            X_pool: pd.DataFrame,
                            z_pool: np.ndarray,
                            info_measure,
                            conditional:bool=True,
                            beta: float=1.0) -> tuple[pd.DataFrame, pd.Series]:
        
        clf = self.model.fit(X_train, z_train)
        risk_score = clf.predict(X_pool)
        min_val = risk_score.min()
        max_val = risk_score.max()
        # normality score to abnormality in the [0,1] interval (reversed mapping)
        norm_risk_score = (risk_score - min_val) / np.maximum((max_val - min_val), 1e-8)
        
        if info_measure == "variance":
                                
            if conditional:
                tree_preds = [clf[i].conditional_predict(X_pool.values, z_pool) 
                              for i in range(clf.n_estimators)]
            elif conditional == False: #if self.conditional is False
                tree_preds = [clf[i].predict(X_pool.values) 
                              for i in range(clf.n_estimators)]
            else:
                raise ValueError("\'conditional\' parameter not found")
                
            train_pop_ranks = np.sort(np.array(clf.predict(X_train)))

            # Count instances in train_pop_ranks that are lower or equal to each element in tree_preds
            tree_ranks = np.searchsorted(train_pop_ranks, tree_preds, side='right')
            base_inform =  tree_ranks.std(0)
            
                
        elif info_measure == "uncertainty":
            
            pop_avg = clf.predict(X_train).mean()

            if conditional:
                base_inform = np.abs(clf.conditional_predict(X_pool, z_pool) - pop_avg) 
            else: #if self.conditional is False
                base_inform = np.abs(clf.predict(X_pool) - pop_avg)
            
        # regardless of the base_inform measure, normalise it so that the
        # beta parameter has a meaningful adn controllable influence
        # we ensure that density and informativeness has comparable scales
        
        min_val = base_inform.min()
        max_val = base_inform.max()
        # normality score in [0,1] interval
        base_inform = (base_inform - min_val) / np.maximum((max_val - min_val), 1e-8)
        
        # we compute the log of the Equation in page 5 from Settles and Craven
        # An Analysis of Active Learning Strategies for Sequence Labeling Tasks (2008)
        log_info_density = base_inform + beta*norm_risk_score
        
        #argsort (reversed) and take top (highest) elements, output will be list 
        query_index = np.argsort(log_info_density)[::-1][:self.batch_size]
        
        if self.verbose > 3:
            print('sampled ilocs:', query_index)
            print('with score:', log_info_density[query_index])
        
        
        return X_pool.iloc[query_index], z_pool[query_index], query_index
    
    
    
    def info_density_sampling(self,
                              X_train,
                              z_train,
                          X_pool: pd.DataFrame,
                          z_pool: np.ndarray,
                          clf_density,
                          info_measure,
                          conditional:bool=True,
                          beta: float=1.0) -> tuple[pd.DataFrame, pd.Series]:
        
        clf = self.model.fit(X_train, z_train)
        normality_score = clf_density.score_samples(X_pool)
        min_val = normality_score.min()
        max_val = normality_score.max()
        # normality score to abnormality in the [0,1] interval (reversed mapping)
        abnormality_score = 1 - ((normality_score - min_val) / np.maximum((max_val - min_val), 1e-8))
        
        if info_measure == "variance":
                                
            if conditional:
                tree_preds = [clf[i].conditional_predict(X_pool.values, z_pool) 
                              for i in range(clf.n_estimators)]
            elif conditional == False: #if self.conditional is False
                tree_preds = [clf[i].predict(X_pool.values) 
                              for i in range(clf.n_estimators)]
            else:
                raise ValueError("\'conditional\' parameter not found")
                
            train_pop_ranks = np.sort(np.array(clf.predict(X_train)))

            # Count instances in train_pop_ranks that are lower or equal to each element in tree_preds
            tree_ranks = np.searchsorted(train_pop_ranks, tree_preds, side='right')
            base_inform =  tree_ranks.std(0)
            
                
        elif info_measure == "uncertainty":
            
            pop_avg = clf.predict(X_train).mean()

            if conditional:
                base_inform = np.abs(clf.conditional_predict(X_pool, z_pool) - pop_avg) 
            else: #if self.conditional is False
                base_inform = np.abs(clf.predict(X_pool) - pop_avg)
            
        # regardless of the base_inform measure, normalise it so that the
        # beta parameter has a meaningful adn controllable influence
        # we ensure that density and informativeness has comparable scales
        
        min_val = base_inform.min()
        max_val = base_inform.max()
        # normality score in [0,1] interval
        base_inform = (base_inform - min_val) / np.maximum((max_val - min_val), 1e-8)
        
        # we compute the log of the Equation in page 5 from Settles and Craven
        # An Analysis of Active Learning Strategies for Sequence Labeling Tasks (2008)
        log_info_density = base_inform + beta*abnormality_score
        
        #argsort (reversed) and take top (highest) elements, output will be list 
        query_index = np.argsort(log_info_density)[::-1][:self.batch_size]
        
        if self.verbose > 3:
            print('sampled ilocs:', query_index)
            print('with info-density:', log_info_density[query_index])
        
        
        return X_pool.iloc[query_index], z_pool[query_index], query_index
    
    
    
    def info_risk_density_sampling(self,
                                   X_train,
                                   z_train,
                            X_pool: pd.DataFrame,
                            z_pool: np.ndarray,
                            clf_density,
                            info_measure,
                            conditional:bool=True,
                            beta1: float=1.0,
                            beta2: float=1.0) -> tuple[pd.DataFrame, pd.Series]:
        
        clf = self.model.fit(X_train, z_train)
        
        normality_score = clf_density.score_samples(X_pool)
        min_val = normality_score.min()
        max_val = normality_score.max()
        # normality score to abnormality in the [0,1] interval (reversed mapping)
        abnormality_score = 1 - ((normality_score - min_val) / np.maximum((max_val - min_val), 1e-8))
        
        
        risk_score = clf.predict(X_pool)
        min_val = risk_score.min()
        max_val = risk_score.max()
        # risk score to to the [0,1] interval
        norm_risk_score = (risk_score - min_val) / np.maximum((max_val - min_val), 1e-8)
        
        
        if info_measure == "variance":
                                
            if conditional:
                tree_preds = [clf[i].conditional_predict(X_pool.values, z_pool) 
                              for i in range(clf.n_estimators)]
            elif conditional == False: #if self.conditional is False
                tree_preds = [clf[i].predict(X_pool.values) 
                              for i in range(clf.n_estimators)]
            else:
                raise ValueError("\'conditional\' parameter not found")
                
            train_pop_ranks = np.sort(np.array(clf.predict(X_train)))

            # Count instances in train_pop_ranks that are lower or equal to each element in tree_preds
            tree_ranks = np.searchsorted(train_pop_ranks, tree_preds, side='right')
            base_inform =  tree_ranks.std(0)
            
                
        elif info_measure == "uncertainty":
            
            pop_avg = clf.predict(X_train).mean()

            if conditional:
                base_inform = np.abs(clf.conditional_predict(X_pool, z_pool) - pop_avg) 
            else: #if self.conditional is False
                base_inform = np.abs(clf.predict(X_pool) - pop_avg)
            
        # regardless of the base_inform measure, normalise it so that the
        # beta parameter has a meaningful adn controllable influence
        # we ensure that density and informativeness has comparable scales
        
        min_val = base_inform.min()
        max_val = base_inform.max()
        # normality score in [0,1] interval
        base_inform = (base_inform - min_val) / np.maximum((max_val - min_val), 1e-8)
        
        # we compute the log of the Equation in page 5 from Settles and Craven
        # An Analysis of Active Learning Strategies for Sequence Labeling Tasks (2008)
        log_info_density = base_inform + beta1*abnormality_score +beta2*norm_risk_score
        
        #argsort (reversed) and take top (highest) elements, output will be list 
        query_index = np.argsort(log_info_density)[::-1][:self.batch_size]
        
        if self.verbose > 3:
            print('sampled ilocs:', query_index)
            print('with info-density:', log_info_density[query_index])
        
        
        return X_pool.iloc[query_index], z_pool[query_index], query_index

        
    

    def reveal_labels(self, z_mask, y_pool, reveal_sampling) -> np.ndarray:
        
        # z_mask are the currently available labels,
        # y_pool are the underlying true labels (not always \delta_i = 1)
        z_mask = pd.DataFrame(z_mask, columns=self.column_names)

        
        if (len(reveal_sampling) != len(y_pool)) or (len(reveal_sampling) != len(z_mask)):
            raise ValueError('Length of sampling distribution must be the same \
                             as the length of the masked labels in the pool set')
                             
        if list(z_mask.columns) != ['event', 'time']:
            raise ValueError('column names are not \'event\' and \'time\' as expected')
            
            
        if self.reveal_scenario == 'partial':            
            z_mask['time'] = z_mask['time']*(1-reveal_sampling) + reveal_sampling*(y_pool['time'])
            z_mask['event'] = np.where(reveal_sampling == 1,
                                               y_pool['event'], False)
            
            z_mask = z_mask.astype({'event': 'bool', 'time': 'float32'})
            z_mask = z_mask.to_records(index=False) #builds the structured array, needed for RSF
            
        elif self.reveal_scenario == 'full':
            z_mask = y_pool # full reveal y_pool, and it is already a recarray
        else:
            raise ValueError('reveal scenario not recognized')
            

        return z_mask


    def update_labels_pool(self, z_mask, y_pool, iloc_idx, reveal_masking):
        
        z_mask_round = pd.DataFrame(z_mask[iloc_idx],
                                    columns=self.column_names)
        y_pool_round = pd.DataFrame(y_pool[iloc_idx],
                                    columns=self.column_names)
        
        z_reveal_round = self.reveal_labels(z_mask_round, y_pool_round,
                                           reveal_masking[iloc_idx])
        
        exhaust_idx = [True if reveal_masking[iloc_idx][i] == 1  else False 
                   for i in range(len(reveal_masking[iloc_idx]))]

        return z_reveal_round, exhaust_idx
    
    
    # X_pool, z_pool, clf_density, "variance", conditional=True, beta=BETA_k[N_round-1])
    def sample_strategy(self, strategy, X_train, X_pool, z_pool, N_round, 
                        n_learners, submethod, conditional, gamma):
        
        if strategy not in self.STRATEGY_FULL_LIST:
            raise ValueError('stategy: \'{}\' not found in list'.format(strategy))
            
        
        if strategy == "random":
            X_queried, z_queried, iloc_pool_idx = self.random_sampling(X_pool, z_pool,
                                                                      random_state=N_round)
            
        elif strategy == "uncertainty":
            X_queried, z_queried, iloc_pool_idx = self.uncertainty_sampling(X_pool,
                                                    z_pool,
                                                    conditional=True)

        elif strategy == "variance":                                        
            X_queried, z_queried, iloc_pool_idx = self.variance_sampling(X_pool,
                                                    z_pool,
                                                    conditional=True)
        
        elif strategy == "old_uncertainty":
            X_queried, z_queried, iloc_pool_idx = self.uncertainty_sampling(X_pool,
                                                    z_pool,
                                                    conditional=False)

        elif strategy == "old_variance":
            X_queried, z_queried, iloc_pool_idx = self.variance_sampling(X_pool,
                                                    z_pool,
                                                    conditional=False)

            
        elif strategy == "dens_variance":
            
            clf_density = IsolationForest(n_estimators=n_learners,
                                          random_state=0).fit(X_train)
            
            X_queried, z_queried, iloc_pool_idx = self.info_density_sampling(X_pool,
                                                    z_pool, clf_density,
                                                    "variance",
                                                    conditional=True,
                                                    beta=gamma)
            
        elif strategy == "dens_uncertainty":
            clf_density = IsolationForest(n_estimators=n_learners,
                                          random_state=0).fit(X_train)
  
            X_queried, z_queried, iloc_pool_idx = self.info_density_sampling(X_pool,
                                                    z_pool, clf_density,
                                                    "uncertainty",
                                                    conditional=True,
                                                    beta=gamma)
            
        elif strategy == "dens_variance_sqrt":    
            clf_density = IsolationForest(n_estimators=n_learners,
                                          random_state=0).fit(X_train)
            
            X_queried, z_queried, iloc_pool_idx = self.info_density_sampling(X_pool,
                                                    z_pool, clf_density,
                                                    "variance",
                                                    conditional=True,
                                                    beta=gamma)

        return X_queried, z_queried, iloc_pool_idx

      
