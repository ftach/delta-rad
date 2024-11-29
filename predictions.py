'''Functions to predict outcome with filtered dataset after feature selection. '''
import pandas as pd 
import numpy as np 

from statsmodels.stats.contingency_tables import mcnemar
from sklearn.ensemble import RandomForestClassifier

import sklearn_utils as sku 

def train_rf(X_train_filtered, y_train): 
    param_grid = {'max_depth': range(1, 5, 4), 'n_estimators' : range(25, 50, 25)} # maximimum depth tuning
    scorer = 'roc_auc'
    estimator = RandomForestClassifier(random_state=42) 

    grid_rf = sku.hyper_parameters_search(estimator, X_train_filtered, y_train, param_grid, scorer=scorer, cv=5) 

    return grid_rf.best_estimator_

def train_model(pred_algo: str, X_train_filtered: pd.DataFrame, y_train: pd.DataFrame): 
    # TODO
    if pred_algo == 'RF': 
        print("Beginning training with Random Forest...")
        best_model = train_rf(X_train_filtered, y_train)
        print("Training with Random Forest ended.")

    return best_model 

def compute_metric(X_val: np.ndarray, y_val: np.ndarray):
    # TODO
    pass 
    return sens, spec, roc_auc, mispreds

def compute_pvalue(binary_preds1, binary_preds2, y_test): 
    # the algorithm needs to be trained again!! find the params and features in the json file. That avoids to save ALL the outcomes 
    # https://www.statology.org/mcnemars-test-python/ 

    # Create a contingency table
    contingency_table = [[0, 0], [0, 0]]
    for true, p1, p2 in zip(y_test, binary_preds1, binary_preds2):
        if p1 == true and p2 != true:
            contingency_table[0][1] += 1
        elif p1 != true and p2 == true:
            contingency_table[1][0] += 1

    # McNemar's test
    result = mcnemar(contingency_table, exact=True)

    return result.pvalue 
