# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:11:06 2023

@author:       Klest Dedja
@institution:  KU Leuven
"""
import os
import numpy as np
import pandas as pd
import pickle
import copy
import warnings
import matplotlib.pyplot as plt

# import Orange as orange
from scipy import stats
from utilities import format_plotting_info, return_performance_info, return_curve_summary_stats
from utilities import find_matching_substrings

MAX_ITERS = 200

table_strategies = ['variance', 'dens_variance', 'random', 'uncertainty', 'dens_uncertainty']
table_strategies = ['random', 'uncertainty', 'variance', 'dens_uncertainty', 'dens_variance']


root_folder = os.getcwd()
store_dicts_folder  = os.path.join(root_folder, "plot-and-perform-info-June")
dnames = os.listdir(store_dicts_folder)
match_with = ['Framingham', 'grace', 'NHANES', 'rott2', 'UnempDur', 'vlbw', 'simul']

# match_with += ['csl', 'aids', 'hdfail', 'oldmort' ]
# TENTATIVE ADD: prostateSurvival
# DROP: grace

testing_dnames = find_matching_substrings(dnames, match_with)

testing_dnames = [name for name in testing_dnames if 'simul_data_4' not in name]

##%% here get tables from stored dicts (hopefully still alive)

df = pd.DataFrame(columns=[f'{elem}_avg' for elem in table_strategies] +\
                          [f'{elem}_std' for elem in table_strategies])

for dname in testing_dnames:
    
    with open(os.path.join(store_dicts_folder, dname), 'rb') as f:
        dict_info = pickle.load(f)
        
    table_info = format_plotting_info(dict_info, table_strategies)
    table_info2 = return_performance_info(table_info, table_strategies, MAX_ITERS)
    table_info3 = return_curve_summary_stats(table_info2, table_strategies)

    avg_values = table_info3['average']
    std_values = table_info3['std_dev']
    
    row_dict = {**avg_values, **std_values}

    df = df.append(pd.Series(row_dict, name=dname.split('_full_info')[0]))
    
mean_values = df.mean()   
df = df.append(pd.Series(mean_values, name='average'))

std_values = df.std()   
df = df.append(pd.Series(std_values, name='average_std'))


#%%


df2 = copy.copy(df)


def bold_max(row, avg_cols):
    max_val = row[avg_cols].max()
    return pd.Series([val == max_val for val in row[avg_cols]], index=avg_cols)

# Identify which values to bold
bold_flags = df.apply(lambda row: bold_max(row, [f'{col}_avg' for col in table_strategies]), axis=1)

def format_values(row, col):
    val = row[f'{col}_avg']
    std = row[f'{col}_std']
    if bold_flags.at[row.name, f'{col}_avg']:  # If this is a max value, wrap value_avg in \textbf{}
        val = '\\textbf{' +'{:.3f}'.format(val)+'}'
    else:
        val = '{:.3f}'.format(val)
    formatted = val + ' \\pm {:.3f}'.format(std)
    return formatted

# Apply the format function
for col in table_strategies:
    df[col] = df.apply(lambda row: format_values(row, col), axis=1)

# Only keep the formatted columns for the final table
df = df[table_strategies]

df.index = df.index.str.replace('my_', '', regex=True)
df.index = df.index.str.replace('my_', '_', regex=True)
df.index = df.index.str.replace('_', ' ', regex=True)

# Convert to LaTeX
latex_table = df.to_latex(index=True, escape=False)

print(latex_table)


#%%

# Let's assume avg_values_dict and std_values_dict are your dictionaries with average and std values for each element
# avg_values_dict = {f'{elem}_avg': avg_values for elem in list1}  # Replace 0.5 with actual avg values
# std_values_dict = {f'{elem}_std': std_values for elem in list1}  # Replace 0.1 with actual std values


#%% here get tables from csv file ( which got lost in the meantime)

SAVE_OUTPUTS = False

# from IPython import get_ipython
# from utilities import rename_data_index
# get_ipython().run_line_magic('matplotlib', 'inline')

FILENAME = 'AUC_results_22May.csv'

##########################################################################
#root_folder = "C:/Users/u0135479/Documents/GitHub/Explain-extract-trees/"
root_folder = os.getcwd()

df = pd.read_csv(os.path.join(root_folder, FILENAME), index_col=[0])

results_folder = os.path.join(root_folder, "Final_results")


data_names = df.index


def output_latex(df, col_format, highlight_min, drop_cols=[], exclude=[]): # {:,.4f}
    assert isinstance(col_format, str)
    
    df = df.drop(drop_cols, axis=1, errors="ignore", inplace=False)
    
    df = df.applymap(col_format.format) #'{:,.4f}' or '{:.2f}'
    #df2 = df.style.format('{:.2f}')
    for col in df.columns:
        df[col] = df[col].astype(float)
    
    latex_names = []#list(df.index)
    for name in df.index:
        latex_names.append(name.replace("_", " "))
    df.index = latex_names
    
    subsets = df.columns.drop(exclude, errors="ignore") # "overloading" the list, bad practise...
    
    if highlight_min == True:
        out_latex = df.style.highlight_min(axis=1, subset=subsets,
                                props='textbf:--rwrap;').to_latex(hrules=True)
    elif highlight_min == False:
        out_latex = df.style.highlight_max(axis=1, subset=subsets,
                                props='textbf:--rwrap;').to_latex(hrules=True)
        
    elif highlight_min in ["None", None]:
        out_latex = df.to_latex(hrules=True)
    else:
        KeyError("Highlight option not recognized")
        
    out_latex = out_latex.replace("_", " ") #test this... does it work?
    
    return out_latex
    
std_drop_cols = [col for col in df.columns if ("std" in col) or ('old' in col) or ("sqrt" in col)]

df2 = df.drop(std_drop_cols, axis=1)

colnames = [col.split("_avg")[0] for col in df2]

df2.columns = colnames

df2 = df2[['random', 'uncertainty', 'variance','dens_uncertainty','dens_variance']]

df2 = df2.rename(columns={'dens_uncertainty': 'dens. + uncertainty',
                         'dens_variance': 'dens. + variance'})

df2 = df2.rename(index={'my_simul_data_1': 'simul data 1', 'my_simul_data_3': 'simul data 2'})

indices = df.index.tolist()
indices[0], indices[1] = indices[1], indices[0]
df2 = df2.reindex(indices)

latex_perfs = output_latex(df2, '{:,.4f}', highlight_min=False,
                           exclude=[],
                           drop_cols=std_drop_cols)

import re
pattern = re.compile("(\.\d{2,4})(00)") #crop final 2 zeros from decimal floats

def regex_replacer(in_string, in_pattern):
    
    matches = in_pattern.finditer(in_string)
    for match in matches:
        in_string = in_string.replace(match.group(0), match.group(1))
        
    return in_string

latex_perfs2 = regex_replacer(latex_perfs, pattern)

#%%

def orange_plot(df, filename, is_ascending, save_outputs, exclude=[],
                plot_title="Nemenyi test", reverse_ranks=False,
                dpi_figure=100):
    
    # is ascending: for computing ranks. If true lowest ir ranked 1
    # reverse ranks: if true, greatest average rank is plotted to the left
    # instead of the right
    
    df.drop(exclude, axis=1, errors="ignore", inplace=True) # "overloading" the list, bad practise...
    
    list_friedman = []
    for col in df.columns:
        list_friedman.append(df[col])

    Friedman_stat = stats.friedmanchisquare(*list_friedman)
    print('Friedman H0 p-value:', Friedman_stat[1])
    print("df:", Friedman_stat[0])
    
    ranks = df.T.rank(method="dense", ascending=is_ascending)
    avg_ranks = ranks.mean(axis=1)

    names = df.columns
    n_samples = df.shape[0]

    ### TODO consider Wilcoxon-Holm post-hoc procedure!

    cd = orange.evaluation.compute_CD(avg_ranks, n_samples)
    
    plot_top= int(min(avg_ranks))
    plot_bott= int(max(avg_ranks))+1
    
    orange.evaluation.graph_ranks(avg_ranks, names, cd=cd,
                                  lowv=plot_top, highv=plot_bott,
                                  width=3.7, textspace=1,
                                  reverse=reverse_ranks)
    plt.rcParams['figure.dpi'] = dpi_figure
    plt.tight_layout()
    plt.title(plot_title)
    figsize = plt.rcParams['figure.figsize']
        
    figsize[1] = figsize[1]*1.1
    
    plt.rcParams['figure.figsize'] = figsize


    if SAVE_OUTPUTS == True: # some setting here (and the few lines above, are off)
        plt.savefig(os.path.join(root_folder, 'final-plots', filename))#,
                    #bbox_inches="tight")
        plt.savefig(os.path.join(root_folder, 'final-plots', filename)
                    , bbox_inches='tight', pad_inches=0.1, transparent=True)

    return plt.show()



#%% PERFORMANCE - INTERPRETABILITY TRADE OFF

performances = df_complete.loc["average"]
performances.rename({"Bellatrex, wt.": "Bellatrex"}, inplace=True, errors="ignore")
complexity = df_compare_rules.loc["average"]

perf_idx = performances.index
complex_idx = complexity.index

#%% intersect to find methods in common ( for which both PERF and COMPLEX is available)
perf_idx = perf_idx.intersection(complex_idx)

performances = performances.loc[perf_idx]
complexity = complexity.loc[perf_idx]

#%%

measures_dict = {"bin": "AUROC",
                "surv": "C-index",
                "mtr": "weighted MAE",
                "regress" : "MAE",
                "multi": "weighted AUROC"}

measure = measures_dict[SETUP]
#%% gwtting the data right ( indeces must correspond) for trade-off plotting
interpretability = 1/complexity

x = interpretability.values
y = performances.values
names = performances.index

#%% plot trade-off and Pareto Frontier

def plot_pareto_frontier(Xs, Ys, maxX=True, maxY=True, fontsize=12):
    
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    #sorting according to (descending) Xs. Highest Xs is guaranteed \in Pareto
    pareto_front = [sorted_list[0]] #initialise list with first member
    for pair in sorted_list[1:]:
        if maxY:        #if bigger is better
            if pair[1] >= pareto_front[-1][1]: #if y >= last y of the front, add
                pareto_front.append(pair)
        else:           # if smaller y is better
            if pair[1] <= pareto_front[-1][1]:  # add if y <= than last in Pareto
                pareto_front.append(pair)
    
    '''Plotting process'''
    plt.scatter(Xs,Ys)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    lines, = ax.plot(pf_X, pf_Y, color='red', marker='o', linestyle='dashed')
    lines.set_label("Pareto frontier")
    ax.legend(fontsize=fontsize)
    
    
    return None


fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

plt.rcParams['text.usetex'] = True
# fig.set_size_inches(7.5, 4)
# fig.set_dpi(100)
FIG_FONT = 15

#plt.figure(figsize=(5,3), dpi=80)
ax.scatter(x, y, s=FIG_FONT-6)
plt.title("Performance-interpretability trade-off", fontsize=FIG_FONT+2)

# We change the fontsize of minor ticks label 
ax.tick_params(axis='both', which='major', labelsize=FIG_FONT-2)
ax.tick_params(axis='both', which='minor', labelsize=FIG_FONT-4)

plot_pareto_frontier(x,y, maxX=True, maxY=(not smaller_is_better), fontsize=FIG_FONT-5)

if smaller_is_better:
    ax.invert_yaxis()
#plt.xlabel(r'\textbf{time (s)}')
plt.ylabel("{}".format(measure), fontsize=FIG_FONT-1)

y_lims = list(ax.get_ylim())
x_lims = list(ax.get_xlim())

pad_prop = 0.05 #add some padding to the plot ( increase axis range)
                # since the FIG_FONT is bigger than usual

# formula works also when axis are flipped (pads top and right corner anyway)
y_lims[1] = y_lims[1]+(y_lims[1]-y_lims[0])*(pad_prop) 
x_lims[1] = x_lims[1]+(x_lims[1]-x_lims[0])*(pad_prop/2)

text_pad = 1.5*1e-3

plt.xlabel('$1 / \mathcal{C}$', fontsize=FIG_FONT+1)
for i, txt in enumerate(names):
    ax.annotate(txt, xy=(x[i], y[i]), 
                xytext=(x[i]+text_pad, y[i]+text_pad),
                fontsize=FIG_FONT)

ax.set_ylim(tuple(y_lims))
ax.set_xlim(tuple(x_lims))

plt.savefig(os.path.join(results_folder, "Trade-off_"+SETUP.upper() +".pdf"))
plt.show()


#%%

'''
BETTER STRING MANAGEMENT WITH REGULAR EXPRESSIONS

STANDARDISE DATASET NAMES: shorten them consistently,
manually provide list of indeces
'''


import re
import copy


# re.compile to define the pattern to be searched

# find occurences with .finditer -> generates iterable with all matches

pattern = re.compile("(\.\d{2,4})(00)") #crop final 2 zeros from decimal floats

def regex_replacer(in_string, in_pattern):
    
    matches = in_pattern.finditer(in_string)
    for match in matches:
        in_string = in_string.replace(match.group(0), match.group(1))
        
    return in_string


latex_ablation = latex_ablation.replace(r"average", "\\midrule \nAverage")
latex_perfs = latex_perfs.replace(r"average", "\\midrule \nAverage")
latex_diss = latex_diss.replace(r"average", "\\midrule \nAverage")
latex_rules = latex_rules.replace(r"average", "\\midrule \nAverage")


latex_perfs = latex_perfs.replace(", wt.", "")


latex_diss2 = regex_replacer(latex_diss, pattern)
latex_perfs2 = regex_replacer(latex_perfs, pattern)
latex_rules2 = regex_replacer(latex_rules, pattern)
#repeat for latex_rules, since it has only 2 digits after the . separator
latex_rules2 = regex_replacer(latex_rules2, pattern)
latex_ablation2 = regex_replacer(latex_ablation, pattern)
    
    
#%%
#latex_rules3 = copy.copy(latex_rules2)
latex_rules3 = ""

pattern2 = re.compile("\d{1,2}\s(\&\s(\textbf{)?\d{1,2}\.\d{2}\}?\s\&)")


#  WANRING: CAL500 & 29.29 ... does not work as expected! 

for line in latex_rules2.split("\n"):
    #print(line)
    line_clean = line.replace("lrr", "lr", 1)
    line_clean = line.replace("& n. rules &", "(n. rules) &")
    
    matches = pattern2.finditer(line)
    matchn = list(matches)[0:1] #works beacuse n. rules is the 2nd column

    for match in matchn:
        #print(match.group(1))
        out_string =  copy.copy(match.group(1))
        out_string = out_string.replace("& ", "(", 1)
        out_string = out_string.replace(" &", ") &", 1)
        
        line_clean = line.replace(match.group(1), out_string)
        #print(line_clean)
    latex_rules3 += line_clean + "\n"

        
latex_rules2 = copy.copy(latex_rules3)

del latex_rules3

#df_ablations_extra = copy.copy(df_ablations)
#df_ablations_extra["delta step1"] = df_ablations_extra['Bellatrex'] - df_ablations_extra['no step 1']
#df_ablations_extra["delta step2"] = df_ablations_extra['Bellatrex'] - df_ablations_extra['no step 2']
#df_ablations_extra["delta step12"] = df_ablations_extra['Bellatrex'] - df_ablations_extra['no steps 1-2']
