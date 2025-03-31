'''Functions to select features. '''
from sklearn.linear_model import Lasso
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import GenericUnivariateSelect, f_classif, chi2, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import train as train
import sklearn_utils as sku 
import pandas as pd 
import numpy as np 
import random 
from typing import Sequence


SCORER = 'roc_auc' # 'f1'


def select_best_features(importances, feat_names, n_features):
    """
    Select the best features based on their importances.

    Parameters:
    importances (list of float): A list of feature importances.
    feat_names (list of str): A list of feature names corresponding to the importances.
    n_features (int): The number of top features to select.

    Returns:
    dict: A dictionary containing the top n_features with their names as keys and importances as values.
    """

    # Combine importances and feature names into a list of tuples
    feature_importance = list(zip(feat_names, importances))
    
    # Sort the list of tuples based on importances in descending order
    sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    # Select top n_features from both input and sorted lists
    top_features = {k: v for k, v in sorted_features[:n_features] if v > 0}

    return top_features

def rf_feat_sel(znorm_scaled_x_train, y_train, features_list, max_features: int = 5): 
    """
    Perform feature selection using a Random Forest model in 5-fold cross-validation.

    Parameters:
    znorm_scaled_x_train (np.ndarrray): The normalized and scaled training data.
    y_train (np.ndarrray): The target variable for the training data.
    features_list (list): List of feature names corresponding to the columns in znorm_scaled_x_train.
    max_features (int, optional): The maximum number of features to select. Default is 5.

    Returns:
    tuple: A tuple containing:
        - selected_features (list): List of selected feature names.
        - best_model (RandomForestClassifier): The trained Random Forest model.
    """

    #print("Beginning feature selection with Random Forest...")

    best_model = train.train_rf(znorm_scaled_x_train, y_train)
    selected_features = select_best_features(best_model.feature_importances_, features_list, n_features=max_features)
    
    #print("Feature selection with Random Forest ended.")

    return selected_features, best_model

def adaboost_feat_sel(znorm_scaled_x_train, y_train, features_list, max_features: int = 5):
    """
    Perform feature selection using the AdaBoost algorithm in 5-fold cross-validation.

    Parameters:
    znorm_scaled_x_train (array-like): The normalized and scaled training data.
    y_train (array-like): The target values for the training data.
    features_list (list): List of feature names corresponding to the columns in znorm_scaled_x_train.
    max_features (int, optional): The maximum number of features to select. Default is 5.

    Returns:
    tuple: A tuple containing:
        - selected_features (list): List of selected feature names.
        - best_model (AdaBoostClassifier): The trained AdaBoost model.
    """

    #print("Beginning feature selection with AdaBoost...") 

    best_model = train.train_adaboost(znorm_scaled_x_train, y_train)
    selected_features = select_best_features(best_model.feature_importances_, features_list, n_features=max_features)

    #print("Feature selection with AdaBoost ended.")

    return selected_features, best_model

def lasso_feat_sel(znorm_scaled_x_train, y_train, features_list, max_features: int = 5):
    """
    Perform feature selection using Lasso regression.

    Parameters:
    znorm_scaled_x_train (numpy.ndarray): The normalized and scaled training data.
    y_train (numpy.ndarray): The target values for the training data.
    features_list (list): List of feature names corresponding to the columns in znorm_scaled_x_train.
    max_features (int, optional): The maximum number of features to select. Default is 5.

    Returns:
    tuple: A tuple containing:
        - selected_features (list): List of selected feature names.
        - best_estimator (Lasso): The best Lasso estimator found during hyperparameter tuning.
    """

    #print("Beginning feature selection with Lasso...")
    param_grid = {'alpha' : list(np.arange(0.01, 0.11, 0.01))} # regularizer tuning
    estimator = Lasso(random_state=42) 
    grid_lasso = sku.hyper_parameters_search(estimator, znorm_scaled_x_train, y_train, param_grid, scorer=SCORER, cv=5)

    selected_features = select_best_features(grid_lasso.best_estimator_.coef_, features_list, n_features=max_features)        

    #print("Feature selection with Lasso ended.")

    return selected_features, grid_lasso.best_estimator_


def nzv_feat_sel(znorm_scaled_x_train, features_list, threshold: float = 0.1): 
    """
    Perform Near Zero Variance (NZV) feature selection on the given dataset.
    This function uses the VarianceThreshold method to filter out features 
    with variance below the specified threshold. It is important that the 
    input data is not normalized, as normalization can affect the variance.

    Parameters:
    znorm_scaled_x_train (numpy.ndarray or pandas.DataFrame): The input data 
        for feature selection. It should be scaled but not normalized.
    features_list (list): A list of feature names corresponding to the columns 
        in znorm_scaled_x_train.
    threshold (float, optional): The variance threshold for feature selection. 
        Features with variance below this threshold will be removed. 
        Default is 0.1.

    Returns:
    tuple: A tuple containing:
        - selected_features (numpy.ndarray): An array of selected feature names 
          that have variance above the threshold.
        - None: Placeholder for compatibility with other feature selection 
          methods that might return additional information.
    """

    #print("Beginning feature selection with NZV ({})...".format(threshold))

    selector = VarianceThreshold(threshold=threshold)
    X_filtered  = selector.fit_transform(znorm_scaled_x_train) # data have to have variance different than 0!! Not normalized!! 

    selected_features = selector.get_feature_names_out(features_list)

    #print("Feature selection with NZV ({}) ended.".format(threshold))

    return selected_features, None

def gus_feat_sel(znorm_scaled_x_train, y_train, features_list, max_features: int = 5, method: str = 'ANOVA', mode: str = 'percentile'):
    """
    Perform feature selection with statistical tests of GenericUnivariateSelect class using the specified method and mode.

    Parameters:
    znorm_scaled_x_train (array-like): The normalized and scaled training data.
    y_train (array-like): The target values for the training data.
    features_list (list): List of feature names corresponding to the columns in znorm_scaled_x_train.
    max_features (int, optional): The maximum number of features to select. Default is 5.
    method (str, optional): The feature selection method to use. Options are 'ANOVA', 'CHI2', and 'MI'. Default is 'ANOVA'.

    Returns:
    tuple: A tuple containing:
        - selected_features (array-like): The names of the selected features.
        - selector (GenericUnivariateSelect): The fitted feature selector object.

    Raises:
    ValueError: If there is an issue with the input data during the fit_transform process.
    Notes:
    - If a ValueError is raised during the fit_transform process, the minimum and maximum values of znorm_scaled_x_train will be printed.
    """

    #print("Beginning feature selection with ...")

    if method == 'ANOVA': 
        score_func = f_classif
    elif method == 'CHI2':
        score_func = chi2
    elif method == 'MI':
        score_func = mutual_info_classif 

    if mode == 'percentile':
        selector = GenericUnivariateSelect(score_func=score_func, mode=mode)

    elif mode == 'k_best':
        selector = GenericUnivariateSelect(score_func=score_func, mode=mode, param=max_features)

    try: 
        X_selected = selector.fit_transform(znorm_scaled_x_train, y_train)
    except ValueError:
        # display min max de znorm_scaled_x_train
        print("Min and max values:", np.min(znorm_scaled_x_train), np.max(znorm_scaled_x_train))

    selected_features = selector.get_feature_names_out(features_list)

    #print("Feature selection with {} ({}) ended.".format(method, mode))

    return selected_features, selector 

# TODO: Add other algorithms: PCA 


    #feat_sel_algo_list = ['RF', 'ADABOOST', 'NZV_01', 'NZV_01', 'ANOVA_PERC', 'ANOVA_K_BEST', 'CHI2_PERC', 'CHI2_K_BEST', 'MI_PERC', 'MI_K_BEST', 'NO_SEL', 'RDM_SEL',
    #                'LASSO', 'PCA_7', 'PCA_8', 'PCA_9']
def get_best_features(X_train: np.ndarray, y_train: np.ndarray, feat_sel_algo: str, features_list: list, max_features: int = 5): 
    """
    Selects the best features from the training data using the specified feature selection algorithm.

    Parameters:
    X_train (np.ndarray): The training input samples.
    y_train (np.ndarray): The target values (class labels) as integers or strings.
    feat_sel_algo (str): The feature selection algorithm to use. Options include:
        - 'RF': Random Forest
        - 'ADABOOST': AdaBoost
        - 'LASSO': Lasso Regression
        - 'NZV_01': Near Zero Variance with threshold 0.1
        - 'NZV_001': Near Zero Variance with threshold 0.01
        - 'ANOVA_PERC': ANOVA F-test with percentile mode
        - 'ANOVA_K_BEST': ANOVA F-test with k-best mode
        - 'CHI2_PERC': Chi-squared test with percentile mode
        - 'CHI2_K_BEST': Chi-squared test with k-best mode
        - 'MI_PERC': Mutual Information with percentile mode
        - 'MI_K_BEST': Mutual Information with k-best mode
        - 'PCA_7': Principal Component Analysis (not yet implemented)
        - 'NO_SEL': No selection, return all features
        - 'RDM_SEL': Random selection of features, length determined by max_features
    features_list (list): List of feature names.
    max_features (int, optional): The maximum number of features to select. Default is 5.

    Returns:
    tuple: A tuple containing:
        - best_features (list): The list of selected feature names.
        - best_model: The model used for feature selection, if applicable. None if not applicable.
    """

    #print("Beginning feature selection with {}...".format(feat_sel_algo))
    if feat_sel_algo == 'RF': 
        best_features, best_model = rf_feat_sel(X_train, y_train, features_list, max_features)
    elif feat_sel_algo == 'ADABOOST':
        best_features, best_model = adaboost_feat_sel(X_train, y_train, features_list, max_features)

    elif feat_sel_algo == 'LASSO':
        best_features, best_model = lasso_feat_sel(X_train, y_train, features_list, max_features)

    elif feat_sel_algo == 'NZV_01':
        best_features, best_model = nzv_feat_sel(X_train, features_list, threshold=0.1)
    elif feat_sel_algo == 'NZV_001':
        best_features, best_model = nzv_feat_sel(X_train, features_list, threshold=0.01)

    elif feat_sel_algo == 'ANOVA_K_BEST':
        best_features, best_model = gus_feat_sel(X_train, y_train, features_list, max_features, method='ANOVA', mode='k_best')
    elif feat_sel_algo == 'CHI2_K_BEST':
        best_features, best_model = gus_feat_sel(X_train, y_train, features_list, max_features, method='CHI2', mode='k_best')
    elif feat_sel_algo == 'MI_K_BEST':
        best_features, best_model = gus_feat_sel(X_train, y_train, features_list, max_features, method='MI', mode='k_best')

    elif feat_sel_algo == 'PCA_7':
        # TODO: Add PCA
        pass 

    elif feat_sel_algo == 'NO_SEL':
        best_features, best_model = features_list, None 
    elif feat_sel_algo == 'RDM_SEL':
        best_features, best_model =  random.sample(list(features_list), max_features), None 

    if feat_sel_algo != 'NO_SEL':
        assert len(best_features) == max_features, print("Error, the number of selected features is not equal to the chosen number of features. ", len(best_features), max_features, best_features, feat_sel_algo)

    return best_features, best_model


def filter_dataset(X_train: np.ndarray, X_val: np.ndarray, best_features: Sequence, nb_features: int, feature_names: list): 
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
    selected_features = best_features[:nb_features] # select only i best features 
    
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_val = pd.DataFrame(X_val, columns=feature_names)

    # Use indices to filter the arrays
    X_train_filtered = X_train[selected_features]
    X_val_filtered = X_val[selected_features]

    # transform back X_train_filtered to np.ndarray
    X_train_filtered = X_train_filtered.to_numpy()
    X_val_filtered = X_val_filtered.to_numpy()


    return selected_features, X_train_filtered, X_val_filtered

def filter_dataset2(X: np.ndarray, best_features: Sequence, nb_features: int): 
    """
    Filters the dataset to retain only the features selected by the algorithms.

    Parameters:
    X (pd.DataFrame): Features array.
    best_features (Sequence): Sequence of best features, can be a list or a dictionary.
    nb_features (int): Number of top features to select.

    Returns:
    tuple: A tuple containing:
        - selected_features (list): List of selected feature names.
        - X_filtered (pd.DataFrame): Filtered data array.
        
    """

    if type(best_features) == dict: 
        best_features_dict = best_features
        best_features = list(best_features_dict.keys())
    elif type(best_features) == list: 
        pass 
    selected_features = best_features[:nb_features] # select only i best features 
    
    # Use indices to filter the arrays
    X_filtered = X[selected_features]

    assert len(selected_features) == nb_features, print(len(selected_features), nb_features, selected_features, best_features)

    return selected_features, X_filtered