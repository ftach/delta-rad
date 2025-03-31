'''Functions for analyzing radiomics model performances. '''

import math
from typing import Sequence
from dataset import get_dataset, min_max_scaling
from train import train_model

from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


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
        heatmap_data = pd.DataFrame(
            index=pred_algo_list, columns=feat_sel_algo_list)
        for feat_sel_algo in feat_sel_algo_list:
            for pred_algo in pred_algo_list:
                try:
                    values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features][m]
                              for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                    # remove zeros from the list
                    filtered_values = [x for x in values if (
                        x != 'N/A') and (x != 0) and (x != 'None')]
                    if len(filtered_values) == 0:
                        heatmap_data.loc[pred_algo, feat_sel_algo] = 0
                    else:
                        if value == 'max':
                            heatmap_data.loc[pred_algo, feat_sel_algo] = max(
                                filtered_values)
                        elif value == 'mean':
                            heatmap_data.loc[pred_algo, feat_sel_algo] = np.mean(
                                filtered_values)

                except ValueError:
                    print(results[table][feat_sel_algo][pred_algo]
                          [outcome].keys(), feat_sel_algo, pred_algo, outcome)

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

def get_best_results_to_plot(results: dict, delta_rad_tables: list, feat_sel_algo_list: list, pred_algo_list: list, outcome: str, metric: str, k: int = 3):
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
    print(f"Top {k} results for each table and {outcome} in terms of {metric}:")
    for table in delta_rad_tables:
        top_results[table] = {}
        for feat_sel_algo in feat_sel_algo_list:
            for pred_algo in pred_algo_list:
                if metric == 'train_auc': 
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['train_metrics']['auc']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                elif metric == 'train_brier_loss':
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['train_metrics']['brier_loss']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                elif metric == 'test_auc':
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['test_metrics']['auc']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                elif metric == 'test_brier_loss':
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['test_metrics']['brier_loss']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                elif metric == 'sensitivity':
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['test_metrics']['sensitivity']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                elif metric == 'specificity':
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['test_metrics']['specificity']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                else:
                    print("Metric not recognized. Please choose one of the following: train_auc, train_brier_loss, test_auc, test_brier_loss, sensitivity, specificity.")
                    
                mean_values = [np.mean(sublist) for sublist in all_values]
                filtered_values = [x for x in mean_values if (
                    x != 'N/A') and (x != 0) and (x != 'None')]
                
        # Sort the results list by the max value in descending order and take the top k
        if len(filtered_values) > 0:
            if metric == 'brier_loss':
                filtered_values = sorted(
                    filtered_values, key=lambda x: x, reverse=False)[:k]
            else:
                filtered_values = sorted(
                    filtered_values, key=lambda x: x, reverse=True)[:k]
            top_results[table] = filtered_values

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
    print(f"Top {k} results for each table and {outcome} in terms of {metric}:")
    for table in delta_rad_tables:
        top_results[table] = {}
        results_list = []
        for feat_sel_algo in feat_sel_algo_list:
            for pred_algo in pred_algo_list:
                if metric == 'train_auc': 
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['train_metrics']['auc']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                elif metric == 'train_brier_loss':
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['train_metrics']['brier_loss']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                elif metric == 'test_auc':
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['test_metrics']['auc']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                elif metric == 'test_brier_loss':
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['test_metrics']['brier_loss']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                elif metric == 'sensitivity':
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['test_metrics']['sensitivity']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                elif metric == 'specificity':
                    all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['test_metrics']['specificity']['values']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                    
                mean_values = [np.mean(sublist) for sublist in all_values]
                features = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['features']
                            for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                params = [results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params']
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                filtered_values = [x for x in mean_values if (
                    x != 'N/A') and (x != 0) and (x != 'None')]
                if len(filtered_values) > 0:
                    if metric == 'brier_loss':
                        max_value = min(filtered_values)
                    else:   
                        max_value = max(filtered_values)
                    # get index in the list of the max value
                    index = filtered_values.index(max_value)
                    params = params[index]
                    # get features for which the metric is the max
                    features = features[index]
                    results_list.append(
                        (max_value, pred_algo, feat_sel_algo, features, params))

        # Sort the results list by the max value in descending order and take the top k
        if len(results_list) > 0:
            if metric == 'brier_loss':
                results_list = sorted(
                    results_list, key=lambda x: x[0], reverse=False)[:k]
            else:
                results_list = sorted(
                    results_list, key=lambda x: x[0], reverse=True)[:k]
            top_results[table][outcome] = results_list

        # Display the top k results for each table and each outcome
        print(f"Top {k} mean results for table {table}:")
        for result in results_list:
            print(f"Mean {metric}: {result[0]}, Prediction Algorithm: {result[1]}, Feature Selection Algorithm: {result[2]}, Features: {result[3]}")
        print("\n")
    return top_results


def compute_best_metrics(fpr, tpr, thresholds, y_prob, y_true, youden_index: bool = True):
    '''Computes the best metrics for the best threshold according to Youden index.
    Args:
        fpr (array): The false positive rate.
        tpr (array): The true positive rate.
        thresholds (array): The thresholds.
        y_prob (array): The predicted probabilities.
        y_true (array): The true labels.

    Returns:
        Sensitivity (float): The sensitivity.
        Specificity (float): The specificity.
        Optimal threshold (float): The optimal threshold.
    '''
    thresholds = [x for x in thresholds if math.isfinite(x)]

    assert np.all(np.isfinite(tpr)), "tpr contient des valeurs non définies."
    assert np.all(np.isfinite(fpr)), "fpr contient des valeurs non définies."
    assert np.all(np.isfinite(thresholds)), print(thresholds)

    if youden_index:
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
    else: # maximize distance to the diagonal
        distances = np.sqrt((1 - tpr)**2 + fpr**2)
        optimal_idx = np.argmin(distances)

    optimal_threshold = thresholds[optimal_idx]
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    conf_matrix = confusion_matrix(y_true, y_pred_optimal)
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity, optimal_threshold

def compute_metrics_for_all_thresholds(fpr, tpr, thresholds, y_prob, y_true, table):
    '''Computes the metrics for all thresholds.
    Args:
        fpr (array): The false positive rate.
        tpr (array): The true positive rate.
        thresholds (array): The thresholds.
        y_prob (array): The predicted probabilities.
        y_true (array): The true labels.

    Returns:
        Sensitivity (float): The sensitivity.
        Specificity (float): The specificity.
        Optimal threshold (float): The optimal threshold.
    '''
    thresholds = [x for x in thresholds if math.isfinite(x)]

    assert np.all(np.isfinite(tpr)), "tpr contient des valeurs non définies."
    assert np.all(np.isfinite(fpr)), "fpr contient des valeurs non définies."
    assert np.all(np.isfinite(thresholds)), print(thresholds)

    sensitivities = []
    specificities = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return sensitivities, specificities, thresholds


def find_perf_alg(results, delta_rad_tables, outcomes_list, feat_sel_algo_list, pred_algo_list, threshold: float = 0.7, metric: str = 'test_auc'):
    '''Find the algorithms with good performance based on a given threshold.
    Args:
        results (dict): A dictionary containing the results to be plotted.
        delta_rad_tables (list): A list of the radiomic tables.
        outcomes_list (list): A list of the outcomes.
        feat_sel_algo_list (list): A list of the feature selection algorithms.
        pred_algo_list (list): A list of the prediction algorithms.
        threshold (float): The threshold to be used for selecting the robust algorithms.
        metric (str): The metric to be used for selecting the robust algorithms. Options are 'test_auc', 'sensitivity', 'specificity'.

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
                        if sub_outcome_results[nb_features]['test_metrics']['sensitivity']['values'] != 'None':
                            if metric == 'sens_spec':
                                if (np.mean(sub_outcome_results[nb_features]['test_metrics']['sensitivity']['values']) > threshold) & (np.mean(sub_outcome_results[nb_features]['test_metrics']['specificity']['values']) > threshold): 
                                    print(f"Table: {table}, Outcome: {outcome}, Feature Selection Algorithm: {feat_sel_algo}, Prediction Algorithm: {pred_algo}, Number of Features: {nb_features}, '\n' \
                                          TEST AUC: {np.mean(sub_outcome_results[nb_features]['test_metrics']['auc']['values'])}, Sensitivity: {np.mean(sub_outcome_results[nb_features]['test_metrics']['sensitivity']['values'])}, Specificity: {np.mean(sub_outcome_results[nb_features]['test_metrics']['specificity']['values'])}, '\n' \
                                          Features: {sub_outcome_results[nb_features]['features']}")
                            else:
                                print('Metric not recognized.')


