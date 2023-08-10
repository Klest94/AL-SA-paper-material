# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:54:57 2023

@author: u0135479
"""
import numpy as np
import random

    

def additional_censoring(y_pool, amount_masking=1, intensity=1,
                        column_names=["Survival", "Status"],
                        random_state=None):

    ''' 
    - y_pool: np.recarray, it's the pool dataset for active learning querying, part of it
        ( or all of it) might be masked ( partially or fully masked)
    - amount_masking:
        - if float, or == 1 -> probability of masking instances from y_pool. amount_masking=1 -> all instances are masked
        - if int > 1 -> total quantity of masked elements in the dataframe, has to be less than np.shape(y_pool)[0]  

    - intensity: amount of the 'information loss' due to masking
            - if (float, int) -> information loss is contant acorss masked instances, given an instance
            with observed values (T_i, \delta) the new label will be (T_i*(1-intensity), 0)
            - if (list, tuple) -> needs to be of the form (e1, e2), length 2. The information loss 
            in this case is not constant but we have intensity ~ Uniform(e1, e2) instead.  

    -random_state: for deterministic generation, good for debugging

    - column_names: event times column name, followed by (boolean) event type. List or tuple of length 2 (for now)
    '''
    
    import random
    import copy
    import numpy as np

    random.seed(a=random_state) #overrides random seed within random library
    np.random.seed(random_state) #overrides random seed in numpy library

    assert isinstance(y_pool, np.recarray)

    assert isinstance(column_names, (list, tuple)) and len(column_names) == 2

    N = y_pool.shape[0]

    if amount_masking <= 1 and amount_masking > 0:
        mask_vector = np.random.rand(N, ) < amount_masking #>= amount_masking #False if still visible, True if masked
    elif isinstance(amount_masking, int) and amount_masking > 1 and amount_masking <= N:
        mask_vector = np.array([i in random.sample(range(N), amount_masking) for i in range(N)])
    else:
        raise KeyError("amount_masking is not a float or int in the expected range")

    # checking validity of intensity input. 
    # TODO: distributons with parameters should be accepted as inputs
    if isinstance(intensity, (float, int)):
        intens_censoring = np.ones(N)*intensity # censor everything (with fixed intensity)
    
    elif isinstance(intensity, (tuple, list)):
        if len(intensity) != 2:
            raise KeyError("Intensity variable is a tuple/list of \
                           length: {:d}, expected 2 instead".format(len(intensity)))
        else:
            intens_censoring = np.random.uniform(intensity[0], intensity[1], N) #vector with entries in [0, intensity)

    # censor_intervals = intens_censoring*y_pool[column_names[0]] #not robust to name choice, TODO: improve 

    y_pool2 = copy.copy(y_pool) 
    y_pool2[column_names[0]] = y_pool[column_names[0]]*(1-intens_censoring*(mask_vector))
    y_pool2[column_names[1]] = y_pool[column_names[1]]*(1-mask_vector) # sets to False all masked observations


    return y_pool2, intens_censoring