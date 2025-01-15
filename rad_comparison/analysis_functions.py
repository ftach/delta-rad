'''Functions for analyzing radiomics model performances. '''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_heatmap(results: dict, table: str, outcome: str, feat_sel_algo_list: list, pred_algo_list: list, metrics: list = ['roc_auc', 'sensitivity', 'specificity'], value: str = 'max'): 
    """
    Plots a heat map based on the given results.
    Args:
        results (dict): A dictionary containing the results to be plotted.
        metric (str, optional): The metric to be used for plotting. Defaults to 'roc_auc'. Options include: 'roc_auc', 'sensitivity', 'specificity'.
        value (str, optional): The value to be used for plotting. Defaults to 'max'. Options include: 'max', 'mean'. 
    Returns:
        None
    """
    heatmaps = []
    for m in metrics:
        heatmap_data = pd.DataFrame(index=pred_algo_list, columns=feat_sel_algo_list)
        for feat_sel_algo in feat_sel_algo_list:
            for pred_algo in pred_algo_list:
                try: 
                    values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features][m] for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                    # remove zeros from the list
                    filtered_values = [x for x in values if (x != 'N/A') and (x != 0) and (x != 'None')]
                    if len(filtered_values) == 0:
                        heatmap_data.loc[pred_algo, feat_sel_algo] = 0
                    else:
                        if value == 'max':
                            heatmap_data.loc[pred_algo, feat_sel_algo] = max(filtered_values)
                        elif value == 'mean':
                            heatmap_data.loc[pred_algo, feat_sel_algo] = np.mean(filtered_values)

                except ValueError:
                    print(results[table][feat_sel_algo][pred_algo][outcome].keys(), feat_sel_algo, pred_algo, outcome)
                
        heatmap_data = heatmap_data.astype(float)
        heatmaps.append(heatmap_data)   
        
    # Plot the heatmap
    plt.figure(figsize=(24, 8))
    for i, heatmap_data in enumerate(heatmaps):
        metric = metrics[i]
        plt.subplot(1, 3, i+1)
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Heatmap of {metric} {value} for {outcome} in {table}')
        if i == 0:
            plt.xlabel('Feature Selection Algorithm')
            plt.ylabel('Prediction Algorithm')
    plt.show()

def get_best_results_dict(results: dict, delta_rad_tables: list, feat_sel_algo_list: list, pred_algo_list: list, metric_list: list, outcome: str): 
    """ 
    Get the top 5 results for each table and each outcome in terms of sensitivity.
    Args:
        results (dict): A dictionary containing the results to be plotted.
        delta_rad_tables (list): A list of the radiomic tables.
        feat_sel_algo_list (list): A list of the feature selection algorithms.
        pred_algo_list (list): A list of the prediction algorithms.
        outcome (str): The outcome of interest.

    Returns:
        None
    """

    top_results = {}
    for metric in metric_list:
        top_results[metric] = {}
        for table in delta_rad_tables: 
            top_results[metric][table] = {}
            results_list = []
            for feat_sel_algo in feat_sel_algo_list:
                for pred_algo in pred_algo_list:
                    values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features][metric] for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                    filtered_values = [x for x in values if (x != 'N/A') and (x != 'None')]
                    if filtered_values:
                        max_value = max(filtered_values)
                        results_list.append(max_value)

            # Sort the results list by the max value in descending order and take the top 3
            results_list = sorted(results_list, key=lambda x: x, reverse=True)[:10]
            top_results[metric][table] = results_list

    return top_results

def get_best_results(results: dict, delta_rad_tables: list, feat_sel_algo_list: list, pred_algo_list: list, outcome: str, metric: str, k: int = 3): 
    """ 
    Get the top k results for each table and each outcome in terms of sensitivity.
    Args:
        results (dict): A dictionary containing the results to be plotted.
        delta_rad_tables (list): A list of the radiomic tables.
        feat_sel_algo_list (list): A list of the feature selection algorithms.
        pred_algo_list (list): A list of the prediction algorithms.
        outcome (str): The outcome of interest.

    Returns:
        None
    """

    top_results = {}
    print(f"Top 3 results for each table and {outcome} in terms of {metric}:")
    for table in delta_rad_tables: 
        top_results[table] = {}
        results_list = []
        for feat_sel_algo in feat_sel_algo_list:
            for pred_algo in pred_algo_list:
                values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features][metric] for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                filtered_values = [x for x in values if (x != 'N/A') and (x != 0) and (x != 'None')]
                if len(filtered_values) > 0:
                    max_value = max(filtered_values)
                    results_list.append((max_value, pred_algo, feat_sel_algo, len(filtered_values)))

        # Sort the results list by the max value in descending order and take the top k
        if len(results_list) > 0:
            results_list = sorted(results_list, key=lambda x: x[0], reverse=True)[:k]
            top_results[table][outcome] = results_list

        # Display the top k results for each table and each outcome
        print(f"Top {k} results for table {table}:")
        for result in results_list:
            print(f"{metric}: {result[0]}, Prediction Algorithm: {result[1]}, Feature Selection Algorithm: {result[2]}, Number of Features: {result[3]}")
        print("\n")
    return top_results

def find_perf_alg(results, delta_rad_tables, outcomes_list, feat_sel_algo_list, pred_algo_list, threshold: float = 0.7): 
    '''Find the algorithms with good performance based on a given threshold.
    Args:
        results (dict): A dictionary containing the results to be plotted.
        delta_rad_tables (list): A list of the radiomic tables.
        outcomes_list (list): A list of the outcomes.
        feat_sel_algo_list (list): A list of the feature selection algorithms.
        pred_algo_list (list): A list of the prediction algorithms.
    
    Returns:
        None
    '''

    for table in delta_rad_tables: 
        for outcome in outcomes_list: 
            for feat_sel_algo in feat_sel_algo_list:
                for pred_algo in pred_algo_list:
                    # get performances for each number of features, save them in a list
                    sub_outcome_results = results[table][feat_sel_algo][pred_algo][outcome]
                    for nb_features in sub_outcome_results.keys():
                        if sub_outcome_results[nb_features]['sensitivity'] != 'None':
                            if (sub_outcome_results[nb_features]['sensitivity'] > threshold) & (sub_outcome_results[nb_features]['specificity'] > threshold): 
                                print(f"Table: {table}, Outcome: {outcome}, Feature Selection Algorithm: {feat_sel_algo}, Prediction Algorithm: {pred_algo}, Number of Features: {nb_features}, '\n' \
                                      AUC: {sub_outcome_results[nb_features]['roc_auc']}, Sensitivity: {sub_outcome_results[nb_features]['sensitivity']}, Specificity: {sub_outcome_results[nb_features]['specificity']}, '\n' \
                                      Features: {sub_outcome_results[nb_features]['features']}")

def find_robust_alg(results, delta_rad_tables, outcomes_list, feat_sel_algo_list, pred_algo_list, threshold: float = 0.8): 
    '''Find the robust algorithms based on a given threshold.
    Args:
        results (dict): A dictionary containing the results to be plotted.
        delta_rad_tables (list): A list of the radiomic tables.
        feat_sel_algo_list (list): A list of the feature selection algorithms.
        pred_algo_list (list): A list of the prediction algorithms.
        outcomes_list (list): A list of the outcomes.
        threshold (float): The threshold to be used for selecting the robust algorithms.
    
    Returns:
        None
    '''

    for table in delta_rad_tables: 
        for outcome in outcomes_list: 
            for feat_sel_algo in feat_sel_algo_list:
                for pred_algo in pred_algo_list:
                    # get performances for each number of features, save them in a list
                    for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys():
                        if (results[table][feat_sel_algo][pred_algo][outcome][nb_features]['roc_auc'] != 'N/A'):
                            if results[table][feat_sel_algo][pred_algo][outcome][nb_features]['roc_auc'] > threshold: 
                                print(f"Table: {table}, Outcome: {outcome}, Feature Selection Algorithm: {feat_sel_algo}, Prediction Algorithm: {pred_algo}, Number of Features: {nb_features}")


def get_mispreds(results, delta_rad_tables, outcomes_list, feat_sel_algo_list, pred_algo_list): 
    '''Get the mispredicted IDs for each table, outcome, feature selection algorithm and prediction algorithm.
    Args:
        results (dict): A dictionary containing the results to be plotted.
        delta_rad_tables (list): A list of the radiomic tables.
        feat_sel_algo_list (list): A list of the feature selection algorithms.
        pred_algo_list (list): A list of the prediction algorithms.
        outcomes_list (list): A list of the outcomes.
    
    Returns:
        mispredicted_ids (dict): A dictionary containing the mispredicted IDs for each table, outcome, feature selection algorithm and prediction algorithm.
    '''

    mispredicted_ids = {}
    for table in delta_rad_tables: 
        mispredicted_ids[table] = {}
        for outcome in outcomes_list: 
            mispredicted_ids[table][outcome] = []
            for feat_sel_algo in feat_sel_algo_list:
                for pred_algo in pred_algo_list:
                    # get performances for each number of features, save them in a list
                    for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys():
                        mispreds = results[table][feat_sel_algo][pred_algo][outcome][nb_features]['mispreds']
                        if mispreds:
                            mispredicted_ids[table][outcome].extend(mispreds)

    return mispredicted_ids


def get_best_results_features_dict(outcomes_list, metric_list, results: dict, delta_rad_tables: list, feat_sel_algo_list: list, pred_algo_list: list, nb_results: int): 
    """ 
    Get the top 5 results for each table and each outcome in terms of sensitivity.
    Args:
        results (dict): A dictionary containing the results to be plotted.
        delta_rad_tables (list): A list of the radiomic tables.
        feat_sel_algo_list (list): A list of the feature selection algorithms.
        pred_algo_list (list): A list of the prediction algorithms.
        outcome (str): The outcome of interest.

    Returns:
        None
    """

    nb_feat_results = {}
    for table in delta_rad_tables: 
        nb_feat_results[table] = {}
        for outcome in outcomes_list: 
            nb_feat_results[table][outcome] = {}
            for metric in metric_list:
                nb_feat_results[table][outcome][metric] = {}
                values = []
                for feat_sel_algo in feat_sel_algo_list:
                    for pred_algo in pred_algo_list:
                        # get performances for each number of features, save them in a list
                        for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys():
                            if nb_features not in nb_feat_results[table][outcome][metric].keys():
                                nb_feat_results[table][outcome][metric][nb_features] = []

                            value = results[table][feat_sel_algo][pred_algo][outcome][nb_features][metric]
                            if value != 'N/A' and value != 'None':
                                nb_feat_results[table][outcome][metric][nb_features].append(value)
            
    # SORTING
    for table in delta_rad_tables:
        for outcome in outcomes_list:
            for metric in metric_list:
                for nb_features in nb_feat_results[table][outcome][metric].keys():
                    nb_feat_results[table][outcome][metric][nb_features] = sorted(nb_feat_results[table][outcome][metric][nb_features], key=lambda x: x, reverse=True)[:nb_results] # 

    return nb_feat_results
