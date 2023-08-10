# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:19:55 2023

@author: u0135479
"""
import numpy as np
import pandas as pd
from inspect import signature
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = {
    'col1': [1, 2, 3, 6, 5, 6, 7, 8, 9, 11, 0],
    'animals': ['cat', 'dog', 'cat', 'dog', 'cat', 'cat', 'dog', 'cat', 'cat', 'mouse', 'horse'],
    'col3': [4.1, np.nan, 3.2, 5.6, 7.8, 11, -0.4, np.nan, 2, 5, 3.14],
    'col4': [4.1, np.nan, 3.2, np.nan, np.nan, 1, np.nan, np.nan, 2, 15, 3.14],

}

data_test = {
    'col1': [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 11, 0],
    'animals': ['cat', 'dog', 'cat', 'dog', 'cat', 'cat', 'dog', 'fox', 'fox', 'fox', 'human'],
    'col3': [4.1, 3.8, 3.2, 5.6, 7.8, 11, -0.4, 11, 2, 5, 3.14],
    'col4': [4.1, np.nan, 3.2, np.nan, np.nan, 1, np.nan, np.nan, 2, 15, 3.14],

}


df = pd.DataFrame(data)
df_test = pd.DataFrame(data_test)



class OHE_Imputer(BaseEstimator, TransformerMixin):
    
    """
    A custom transformer that combines OneHotEncoder and IterativeImputer.
    
    This class performs one-hot encoding on categorical columns, drops columns with a proportion of missing
    values greater than the specified threshold, and then applies iterative imputation to the remaining columns.
    
    Parameters
    ----------
    threshold_drop_miss : float, optional, default=0.2
    The proportion of missing values threshold for dropping columns. Columns with a proportion
    of missing values greater than this threshold will be dropped.
    
    **kwargs : dict, optional
    Keyword arguments to be passed to both OneHotEncoder and IterativeImputer. In case there are keys in common,
    the same value is passed to both. 
    
    Attributes
    ----------
    cat_columns : pandas Index
    The categorical columns in the input DataFrame.
    
    non_cat_columns : pandas Index
    The non-categorical columns in the input DataFrame.
    
    cols_to_drop : pandas Index
    The columns with a proportion of missing values greater than the specified threshold.
    
    ohe : OneHotEncoder
    The OneHotEncoder instance used to encode categorical columns.
    
    preprocessor : ColumnTransformer
    The ColumnTransformer instance used to preprocess the input DataFrame.
    
    imputer : IterativeImputer
    The IterativeImputer instance used to impute missing values on the columns
    whose amount of missingness does not exceed the threshold.
    """

    
    def __init__(self, threshold_drop_miss=0.2,**kwargs):
        
        self.kwargs = kwargs
        self.threshold_drop_miss = threshold_drop_miss
        
        ohe_signature = signature(OneHotEncoder).parameters
        self.ohe_kwargs = {k: v for k, v in self.kwargs.items() if k in ohe_signature}
        
        transf_signature = signature(ColumnTransformer).parameters
        self.transf_kwargs = {k: v for k, v in self.kwargs.items() if k in transf_signature}
        
        imputer_signature = signature(IterativeImputer).parameters
        self.imputer_kwargs = {k: v for k, v in self.kwargs.items() if k in imputer_signature}
        
        
    def update_col_names(self, all_cols_kept):
        
        """
        Update the column names of the input DataFrame after applying the imputation.
        If the 'add_indicator' parameter is set to True for the IterativeImputer, this function
        adds new column names for the missing value indicators, with the suffix "_was_missing".
        
        Parameters
        ----------
        all_cols_kept : list
            A list containing the column names of the input DataFrame after applying
            the OneHotEncoder and dropping columns with high proportions of missing values.
        
        Returns
        -------
        updated_cols_kept : list
            A list containing the updated column names of the input DataFrame after
            applying the imputation and adding missing value indicator column names if needed.
        """
        
        missing_indicator = self.imputer.indicator_
        missing_indicator_cols = missing_indicator.features_
        missing_col_names = np.array(all_cols_kept)[missing_indicator_cols]
        missing_col_names = [col + "_was_nan" for col in missing_col_names]
        all_cols_kept.extend(missing_col_names)
        
        return all_cols_kept
        

    def fit(self, X, y=None):
        
        """
        Fit the transformer to the input DataFrame.
        
        Parameters
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
            The input DataFrame to fit the transformer.
            
        y : None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.
        
        Returns
        -------
        self : OHEIterativeImputer
            Returns self.
        """
        
        self.cat_columns = X.select_dtypes(include=['object']).columns
        self.non_cat_columns = X.select_dtypes(exclude=['object']).columns
        
        missing_freqs = X.isnull().mean()
        self.cols_to_drop = missing_freqs[missing_freqs > self.threshold_drop_miss].index
        
        self.ohe = OneHotEncoder(**self.ohe_kwargs)
        self.ohe.fit(X[self.cat_columns])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', self.ohe, self.cat_columns),
                ('drop', 'drop', self.cols_to_drop),
            ],
            remainder='passthrough',
            **self.transf_kwargs)
        
        X_transformed = self.preprocessor.fit_transform(X)

        self.imputer = IterativeImputer(**self.imputer_kwargs)
        self.imputer.fit(X_transformed)

        return self

    def transform(self, X):
        
        """
        Transform the input DataFrame using the fitted transformer.
        
        Parameters
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
            The input DataFrame to transform.
        
        Returns
        -------
        X_imputed : pandas DataFrame, shape (n_samples, n_transformed_features)
            The transformed DataFrame with one-hot encoded categorical columns, dropped columns
            with high proportions of missing values, and imputed missing values.
        """
        
        
        X_transformed = self.preprocessor.transform(X)
        X_imputed = self.imputer.transform(X_transformed)

        cat_cols_transformed = self.ohe.get_feature_names_out(self.cat_columns)
        all_cols_transformed = np.concatenate([cat_cols_transformed, self.non_cat_columns])
        
        mask_cols_kept = np.isin(all_cols_transformed, self.cols_to_drop,
                                   invert=True)
        
        all_cols_kept = all_cols_transformed[mask_cols_kept]

        all_cols_kept = [col.rstrip('_sklearn') if col.endswith('_sklearn') else col
                            for col in all_cols_kept]
        
        
        if self.imputer_kwargs.get('add_indicator', False):
            all_cols_kept = self.update_col_names(all_cols_kept)


        return pd.DataFrame(X_imputed, columns=all_cols_kept)


# Create the OHEIterativeImputer instance
ohe_iterative_imputer = OHE_Imputer(estimator=None,
                                    threshold_drop_miss=0.2,
                                    min_frequency=0.1,
                                    add_indicator=True,
                                    random_state=0,
                                    max_iter=1000,
                                    n_nearest_features=None,
                                    handle_unknown='infrequent_if_exist',
                                    verbose=2)

# Fit and transform the DataFrame using the OHEIterativeImputer
df2 = ohe_iterative_imputer.fit_transform(df)
df2_test = ohe_iterative_imputer.transform(df_test)

