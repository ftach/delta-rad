'''Functions for analyzing radiomics model performances. '''

import math
from typing import Sequence
from utils.dataset import get_dataset, min_max_scaling
from utils.train import train_model

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


def get_best_results_dict(results: dict, delta_rad_tables: list, feat_sel_algo_list: list, pred_algo_list: list, metric_list: list, outcome: str, k: int = 10):
    """ 
    Get the top k results for each table and each outcome in terms of sensitivity.
    Args:
        results (dict): A dictionary containing the results to be plotted.
        delta_rad_tables (list): A list of the radiomic tables.
        feat_sel_algo_list (list): A list of the feature selection algorithms.
        pred_algo_list (list): A list of the prediction algorithms.
        outcome (str): The outcome of interest.
        k (int): The number of top results to be displayed.

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
                    values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features][metric]
                              for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                    filtered_values = [x for x in values if (
                        x != 'N/A') and (x != 'None')]
                    if filtered_values:
                        max_value = max(filtered_values)
                        results_list.append(max_value)

            # Sort the results list by the max value in descending order and take the top k
            results_list = sorted(
                results_list, key=lambda x: x, reverse=True)[:k]
            top_results[metric][table] = results_list

    return top_results

def get_best_results_dict2(results: dict, delta_rad_tables: list, feat_sel_algo_list: list, pred_algo_list: list, metric_list: list, outcome: str, k: int = 10):
    """ 
    Get the top k results for each table and each outcome in terms of sensitivity.
    Args:
        results (dict): A dictionary containing the results to be plotted.
        delta_rad_tables (list): A list of the radiomic tables.
        feat_sel_algo_list (list): A list of the feature selection algorithms.
        pred_algo_list (list): A list of the prediction algorithms.
        outcome (str): The outcome of interest.
        k (int): The number of top results to be displayed.

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
                    values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features][metric]
                              for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                    mean_values = [np.mean(sublist) for sublist in values]
                    filtered_values = [x for x in mean_values if (
                        x != 'N/A') and (x != 'None')]
                    if filtered_values:
                        max_value = max(filtered_values)
                        results_list.append(max_value)

            # Sort the results list by the max value in descending order and take the top k
            results_list = sorted(
                results_list, key=lambda x: x, reverse=True)[:k]
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
    print(f"Top {k} results for each table and {outcome} in terms of {metric}:")
    for table in delta_rad_tables:
        top_results[table] = {}
        results_list = []
        for feat_sel_algo in feat_sel_algo_list:
            for pred_algo in pred_algo_list:
                all_values = [results[table][feat_sel_algo][pred_algo][outcome][nb_features][metric]
                          for nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys()]
                # 
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


def retrain_best_models(result, X_train, y_train):
    # Get info from best models and retrain them
    # select features in X_train and X_test

    params = result[4]
    pred_algo = result[1]

    if pred_algo == 'RF':
        clf = RandomForestClassifier(**params)
    elif pred_algo == 'ADABOOST':
        clf = AdaBoostClassifier(**params)
    elif pred_algo == 'LOGREG':
        clf = LogisticRegression(**params)
    elif pred_algo == 'LOGREGRIDGE':
        clf = LogisticRegression(**params)
    elif pred_algo == 'DT':
        clf = DecisionTreeClassifier(**params)
    elif pred_algo == 'PSVM':
        clf = SVC(**params)
    elif pred_algo == 'KNN':
        clf = KNeighborsClassifier(**params)
    elif pred_algo == 'BAGG':
        clf = BaggingClassifier(**params)
    elif pred_algo == 'MLP':
        clf = MLPClassifier(**params)
    elif pred_algo == 'LDA':
        clf = LinearDiscriminantAnalysis(**params)
    elif pred_algo == 'QDA':
        clf = QuadraticDiscriminantAnalysis(**params)
    elif pred_algo == 'GaussianP':
        clf = GaussianProcessClassifier(**params)
    elif pred_algo == 'NaiveB':
        clf = GaussianNB(**params)

    clf = clf.fit(X_train, y_train)

    # retrain the model
    return clf


def retrain_best_models2(result, X_train, y_train):
    # Get info from best models and retrain them with cross-validation

    pred_algo = result[1]

    clf = train_model(pred_algo, X_train, y_train)

    # retrain the model
    return clf

def get_data_from_table(table, features, outcome):

    # get the data
    X_train, X_val, y_train, y_test, features_list = get_dataset("/home/tachennf/Documents/delta-rad/extracted_radiomics/"+table,
                                                                 "/home/tachennf/Documents/delta-rad/extracted_radiomics/outcomes.csv", selection_method='random', outcome=outcome)

    # normalize the data
    X_train, X_val = min_max_scaling(X_train, X_val)

    _, X_train_filtered, X_val_filtered = filter_dataset(
        X_train, X_val, features, features_list)

    return X_train_filtered, X_val_filtered, y_train, y_test


def filter_dataset(X_train: np.ndarray, X_val: np.ndarray, best_features: Sequence, feature_names: list):
    """
    Filters the dataset to retain only the features selected by the algorithms.

    Parameters:
    X_train (np.ndarray): Training data array.
    X_val (np.ndarray): Validation data array.
    best_features (Sequence): Sequence of best features, can be a list or a dictionary.
    nb_features (int): Number of top features to select.
    feature_names (list): List of feature names corresponding to the columns in X_train and X_val.

    Returns:
    tuple: A tuple containing:
        - selected_features (list): List of selected feature names.
        - X_train_filtered (np.ndarray): Filtered training data array.
        - X_val_filtered (np.ndarray): Filtered validation data array.
    """

    if type(best_features) == dict:
        best_features_dict = best_features
        best_features = list(best_features_dict.keys())
    elif type(best_features) == list:
        pass

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_val = pd.DataFrame(X_val, columns=feature_names)

    # Use indices to filter the arrays
    X_train_filtered = X_train[best_features]
    X_val_filtered = X_val[best_features]

    # transform back X_train_filtered to np.ndarray
    X_train_filtered = X_train_filtered.to_numpy()
    X_val_filtered = X_val_filtered.to_numpy()

    return best_features, X_train_filtered, X_val_filtered


def compare_roc_auc(top_results, outcome: str, nice_tables: list, cval: bool = True, youden_index: bool = True):
    '''Plots the ROC AUC of the top models for each table and outcome.
    Args:
        top_results (dict): A dictionary containing the top results.
        outcome (str): The outcome of interest.
        nice_tables (list): A list of the nice names of the tables.
        cval (bool): Whether to use cross-validation or not.

    Returns:
        None
    '''

    plt.figure(figsize=(12, 4))
    plt.plot([0, 1], [0, 1], color=(0.6, 0.6, 0.6), linestyle='--')
    plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':',
             color='black', label='Perfect classifier')
    k = 0
    for table in top_results.keys():
        result = top_results[table][outcome][0]
        features = result[3]
        X_train, X_test, y_train, y_test = get_data_from_table(
            table, features, outcome)  # normalizes the data

        if cval:
            model = retrain_best_models2(result, X_train, y_train)
        else:
            model = retrain_best_models(result, X_train, y_train)

        y_test = y_test.astype('int64')
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        sens, spe, opt_threshold = compute_best_metrics(
            fpr, tpr, thresholds, y_prob, y_test, youden_index)
        print(f"Table: {table}, Sensitivity: {sens}, Specificity: {spe}, Optimal threshold: {opt_threshold}")
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='%s | AUC = %0.3f' %
                 (nice_tables[k], roc_auc))
        k += 1

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR (1-ESP)')
    plt.ylabel('SEN')
    plt.title(f'ROC-AUC Curve for {outcome}')
    plt.legend(loc="lower right")
    plt.show()

    
def compare_sens_spe(top_results, outcome: str, cval: bool = True):
    '''
    Args:
        top_results (dict): A dictionary containing the top results.
        outcome (str): The outcome of interest.
        nice_tables (list): A list of the nice names of the tables.
        cval (bool): Whether to use cross-validation or not.

    Returns:
        None
    '''
    plt.figure(figsize=(14, 5))
    k = 0
    for table in top_results.keys():
        result = top_results[table][outcome][0]
        features = result[3]
        X_train, X_test, y_train, y_test = get_data_from_table(
            table, features, outcome)  # normalizes the data

        if cval:
            model = retrain_best_models2(result, X_train, y_train)
        else:
            model = retrain_best_models(result, X_train, y_train)

        y_test = y_test.astype('int64')
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        sens, spe, thresholds = compute_metrics_for_all_thresholds(
            fpr, tpr, thresholds, y_prob, y_test, table)
        plt.subplot(1, len(list(top_results.keys())), k+1)
        plt.plot(thresholds, sens, label='Sensitivity')
        plt.plot(thresholds, spe, label='Specificity')
        plt.legend()
        plt.xlabel('Threshold')
        plt.ylabel('Metric')
        plt.title(f'{table}')
        
        print(f"Table: {table}, Sensitivity: {sens}, Specificity: {spe}, Threshold: {thresholds}")
        k += 1
    print(len(list(top_results.keys()))+1, k)
    plt.title('Sensitivity and Specificity for different thresholds')
    plt.tight_layout()

    plt.show()


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

def find_perf_alg2(results, delta_rad_tables, outcomes_list, feat_sel_algo_list, pred_algo_list, threshold: float = 0.7, metric: str = 'test_auc'):
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
                        if sub_outcome_results[nb_features]['sensitivity'] != 'None':
                            if metric == 'test_auc': 
                                if (np.mean(sub_outcome_results[nb_features]['test_auc']) > threshold): # & (np.mean(sub_outcome_results[nb_features]['specificity']) > threshold):
                                    print(f"Table: {table}, Outcome: {outcome}, Feature Selection Algorithm: {feat_sel_algo}, Prediction Algorithm: {pred_algo}, Number of Features: {nb_features}, '\n' \
                                          TEST AUC: {np.mean(sub_outcome_results[nb_features]['test_auc'])}, Sensitivity: {np.mean(sub_outcome_results[nb_features]['sensitivity'])}, Specificity: {np.mean(sub_outcome_results[nb_features]['specificity'])}, '\n' \
                                          Features: {sub_outcome_results[nb_features]['features']}")
                            elif metric == 'sensitivity': 
                                if np.mean(sub_outcome_results[nb_features]['sensitivity']) > threshold: 
                                    print(f"Table: {table}, Outcome: {outcome}, Feature Selection Algorithm: {feat_sel_algo}, Prediction Algorithm: {pred_algo}, Number of Features: {nb_features}, '\n' \
                                          TEST AUC: {np.mean(sub_outcome_results[nb_features]['test_auc'])}, Sensitivity: {np.mean(sub_outcome_results[nb_features]['sensitivity'])}, Specificity: {np.mean(sub_outcome_results[nb_features]['specificity'])}, '\n' \
                                          Features: {sub_outcome_results[nb_features]['features']}")
                            elif metric == 'specificity':
                                if np.mean(sub_outcome_results[nb_features]['specificity']) > threshold: 
                                    print(f"Table: {table}, Outcome: {outcome}, Feature Selection Algorithm: {feat_sel_algo}, Prediction Algorithm: {pred_algo}, Number of Features: {nb_features}, '\n' \
                                          TEST AUC: {np.mean(sub_outcome_results[nb_features]['test_auc'])}, Sensitivity: {np.mean(sub_outcome_results[nb_features]['sensitivity'])}, Specificity: {np.mean(sub_outcome_results[nb_features]['specificity'])}, '\n' \
                                          Features: {sub_outcome_results[nb_features]['features']}")
                            elif metric == 'sens_spec':
                                if (np.mean(sub_outcome_results[nb_features]['sensitivity']) > threshold) & (np.mean(sub_outcome_results[nb_features]['specificity']) > threshold): 
                                    print(f"Table: {table}, Outcome: {outcome}, Feature Selection Algorithm: {feat_sel_algo}, Prediction Algorithm: {pred_algo}, Number of Features: {nb_features}, '\n' \
                                          TEST AUC: {np.mean(sub_outcome_results[nb_features]['test_auc'])}, Sensitivity: {np.mean(sub_outcome_results[nb_features]['sensitivity'])}, Specificity: {np.mean(sub_outcome_results[nb_features]['specificity'])}, '\n' \
                                          Features: {sub_outcome_results[nb_features]['features']}")
                            else:
                                print('Metric not recognized.')
                            
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
                                nb_feat_results[table][outcome][metric][nb_features] = [
                                ]

                            value = results[table][feat_sel_algo][pred_algo][outcome][nb_features][metric]
                            if value != 'N/A' and value != 'None':
                                nb_feat_results[table][outcome][metric][nb_features].append(
                                    value)

    # SORTING
    for table in delta_rad_tables:
        for outcome in outcomes_list:
            for metric in metric_list:
                for nb_features in nb_feat_results[table][outcome][metric].keys():
                    nb_feat_results[table][outcome][metric][nb_features] = sorted(
                        nb_feat_results[table][outcome][metric][nb_features], key=lambda x: x, reverse=True)[:nb_results]

    return nb_feat_results
