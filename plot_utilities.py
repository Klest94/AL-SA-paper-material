# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:45:33 2023

@author:       Klest Dedja
@institution:  KU Leuven
"""
import os
import numpy as np
import pandas as pd
import copy
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

all_strategies = ['variance', 'old_variance', 'random', 'uncertainty', 'old_uncertainty', 
              'dens_uncertainty', 'dens_variance', 
              'sqrt_dens_uncertainty', 'sqrt_dens_variance',
              'inv_sqrt_dens_uncertainty', 'inv_sqrt_dens_variance']


def map_common_substrings_to_same(strat_list, group_by ='old_'):
    
    result_dict = {}
    unique_val = 0
    
    if isinstance(group_by, str):
        group_by = [group_by]
        
    for strategy in strat_list:
        
        matching_element = next((elem for elem in group_by if elem in strategy), None)
        #gets FIRST match substring, group_by has implied priority
        if matching_element:
            key = strategy.replace(matching_element, '')
            if key in result_dict:
                result_dict[strategy] = result_dict.get(key)
            else: #it's a new key after all
                result_dict[strategy] = unique_val+1
                unique_val += 1
        else:
            unique_val += 1
            result_dict[strategy] = unique_val
            
    return result_dict
        
    
mapped_strings = map_common_substrings_to_same(all_strategies, ['old_', 'inv_', 'sqrt_'])


#%%

def find_max_length(plot_dict):
    global_max_length = 0
    for key in plot_dict.keys():
        if 'fold' in key:
            fold_max = np.max(np.array([len(plot_dict[key][strat]) for strat in plot_dict[key].keys()]))
            if fold_max > global_max_length:
                global_max_length = fold_max
    return global_max_length
    

def plot_trajectories(fig, ax,
                      base_info, x_axis, strategies,
                      fold_traject=True,
                      conf_bound=False):
        
    # fig, ax = plt.subplots(figsize=(10,6), dpi=80)
    
    avg_train = base_info['average']['full_train']
    std_train = base_info['std_dev']['full_train']
    
    # draw horizontal line for full train data
    
    full_train_line = ax.plot([x_axis[0], x_axis[-1]],
                    [avg_train, avg_train], linestyle='dashdot',
                    linewidth=1.5,
                    color='k', label='upper benchmark')
    
    full_train_color = full_train_line[0].get_color()
    
    ax.plot([x_axis[0], x_axis[-1]],
            [avg_train-std_train, avg_train-std_train],
            linestyle='--', linewidth=0.5, color=full_train_color)
    
    ax.plot([x_axis[0], x_axis[-1]],
            [avg_train+std_train, avg_train+std_train],
            linestyle='--', linewidth=0.5, color=full_train_color)
    
    mapping_dict = map_common_substrings_to_same(strategies, ['dens_', 'inv_', 'sqrt_'])
    
    if np.max(np.array(list(mapping_dict.values()))) > 10:
        warnings.warning("More than 10 strategy colors are beign plotted, some will overlap")
    
    # Get the tab10 colormap
    tab10_colors = mcolors.TABLEAU_COLORS
    colors_list = list(tab10_colors.values()) #equivalent, most likely
    colors_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Map the strategies to colors
    mapped_colors = {strategy: colors_list[value % len(colors_list) -1] #index at 0
                     for strategy, value in mapping_dict.items()}
        
    linestyles = {}
    
    for key in mapped_colors:
        if 'old_' in key:
            linestyles[key] = 'dotted'
        elif 'dens_' in key:
            linestyles[key] = 'dashed'
        else:
            linestyles[key] = 'solid'


    for strategy in strategies: # excludes weird keys such as 'full_train'or 'rounds'
        
        strat_avg = base_info['average'][strategy]
        strat_std = base_info['std_dev'][strategy]
            
        strategy_line = ax.plot(x_axis, strat_avg, linewidth=1.5,\
                                label=strategy.replace('dens_', 'dens+'),
                                color=mapped_colors[strategy],
                                linestyle=linestyles[strategy])    
        
        if conf_bound:
            ax.fill_between(x_axis, 
                            strat_avg-strat_std, strat_avg+strat_std,
                            alpha=0.05 if "old" in strategy else 0.10)
        
        strategy_color = strategy_line[0].get_color()
        
        # if "old" in strategy: #dotted
        #     the_linestyle = ':'
        # elif "inv_sqrt" in strategy: #dashdot
        #     the_linestyle = '-.'
        # elif "_sqrt" in strategy: # and not ; 'inv_sqrt'  #dashed
        #     the_linestyle = '--'
        # else:
        #     the_linestyle = '-'
        
        if fold_traject:
            for key in base_info.keys():
                if 'fold' in key: # make sure keys are folds   
                    ax.plot(x_axis, base_info[key][strategy][:len(x_axis)], linewidth=0.5,
                            linestyle= linestyles[strategy],
                            color=strategy_color, alpha=0.4)
                    

