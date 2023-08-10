# AL-SA-paper-material
reproducibility results for [IAL2023 conference](https://www.activeml.net/ial2023/) paper, we are excited to share our findings and discusse more ideas!

# Upcoming soon!
A detailed giude on how to run this messy code.

# How to run

Step 0: make sure you have a Python environment with basic libraries. Check ``requirements.txt`` for more details.

Step 1: generate raw data with ``data_import.py`` importing from the SurvSet library is needed.
Use the ``pip install SurvSet==0.2.6`` command to install the same version of the library.

Step 2: generate final version of the datasets with the ``process_and_folds.py`` script.
This will also prepare 5-fold cross validation and impute missing values (do we? double check): the RSF model 
(as for ``scikit-survival==0.21``) cannot handle missing values. 

Step 3: Select relevant datasets for the analysis (see paper for more details), use the ``filter_datasets.py`` script for the task.
The current paramters in the script are the ones used in the paper, and represernt a trade-off between having only 'good' datasets and having 'enough' datasets. 

Step 4: Run the ``main_script.py`` for the whole active learning procedure. This will call functions from other files such as ``utillities.py``.
It will populate the ``query-info-data`` and the ``plot-and-perfomr-info`` folders.

Step 5: Analysis of the data and plotting was performed with the ``stat_testing.py`` and the ``plotting_script.py``.

:warning:We initially  generated 4 simulated datasets, and we later dropped two. We share the results for the 1st and 3rd datasets in the paper, and they are renamed as 
``simul_data_1`` and ``simul_data_2`` respectively, but in the project folder they are genearted under the ``my_simul_data_1`` and ``my_simul_data_3`` respectively.



