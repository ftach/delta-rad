'''Functions to predict outcome with filtered dataset after feature selection. '''

import numpy as np 
import pandas as pd 

from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_curve, auc

import warnings
warnings.filterwarnings('ignore')
 
def def_results_dict(delta_rad_tables: list, feat_sel_algo_list: list, pred_algo_list: list, outcomes_list: list, max_features: int):
    '''
    Define the results dictionary. 

    Parameters:
    ----------------
    delta_rad_tables (list): The list of tables.
    feat_sel_algo_list (list): The list of feature selection algorithms.
    pred_algo_list (list): The list of prediction algorithms.
    outcomes_list (list): The list of outcomes.
    max_features (int): The maximum number of features to select.
    
    Returns:
    ----------------
    results (dict): The dictionary containing the results of the prediction algorithms.
    '''
    results = {}

    for table in delta_rad_tables:
        results[table] = {}
        for feat_sel_algo in feat_sel_algo_list:
            results[table][feat_sel_algo] = {'params': {}}
            results[table][feat_sel_algo].update({
                pred_algo: {
                    outcome: {
                        nb_features: {
                            'features': [],
                            'params': {},
                            'train_metrics': {
                                'auc': {
                                    'values': [],
                                    'conf_int': []
                                },
                                'brier_loss': {
                                    'values': [],
                                    'conf_int': []
                                }
                            },
                            'test_metrics': {
                                'auc': {
                                    'values': [],
                                    'conf_int': []
                                },
                                'sensitivity': {
                                    'values': [],
                                    'conf_int': []
                                },
                                'specificity': {
                                    'values': [],
                                    'conf_int': []
                                },
                                'brier_loss': {
                                    'values': [],
                                    'conf_int': []
                                },
                            }
                        }
                        for nb_features in range(1, 1 + max_features)
                    }
                    for outcome in outcomes_list
                }
                for pred_algo in pred_algo_list
            })

    return results


def save_results(results: dict, table: str, fs_algo: str, pred_algo: str, outcome: str, sel_features: list, gs_est: object, train_auc: float, train_brier_loss: float, test_auc: float, sensitivity: float, specificity: float, brier_loss: float, test_auc_ci: tuple, sensitivity_ci: tuple, specificity_ci: tuple, brier_loss_ci: tuple):
    '''Save the results of the prediction algorithms.

    Parameters:
    ----------------
    results (dict): The dictionary containing the results of the prediction algorithms.
    table (str): The name of the table.
    fs_algo (str): The name of the feature selection algorithm.
    pred_algo (str): The name of the prediction algorithm.
    outcome (str): The name of the outcome to predict.
    sel_features (list): The list of selected features.
    gs_est (object): The GridSearchCV object containing the best estimator.
    train_auc (float): The train AUC.
    train_brier_loss (float): The train Brier loss.
    test_auc (float): The test AUC.
    sensitivity (float): The sensitivity.
    specificity (float): The specificity.
    brier_loss (float): The Brier loss.
    test_auc_ci (tuple): The confidence interval for the test AUC.
    sensitivity_ci (tuple): The confidence interval for the sensitivity.
    specificity_ci (tuple): The confidence interval for the specificity.
    brier_loss_ci (tuple): The confidence interval for the Brier loss.

    Returns:
    ----------------
    results (dict): The updated results dictionary that will contain the results of the prediction algorithms
    '''

    # MODEL PARAMETERS
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['features'] = sel_features
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['params'] = gs_est.best_params_ #  params of best algo (based on cross validation search) trained again 

    # TRAIN SCORES
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['train_metrics']['auc']['values'].append(train_auc) 
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['train_metrics']['brier_loss']['values'].append(train_brier_loss)
    
    # TEST SCORES
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['test_metrics']['auc']['values'].append(test_auc)
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['test_metrics']['sensitivity']['values'].append(sensitivity)
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['test_metrics']['specificity']['values'].append(specificity)
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['test_metrics']['brier_loss']['values'].append(brier_loss)

    # CONFIDENCE INTERVALS
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['test_metrics']['auc']['conf_int'].append(test_auc_ci)
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['test_metrics']['sensitivity']['conf_int'].append(sensitivity_ci)
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['test_metrics']['specificity']['conf_int'].append(specificity_ci)
    results[table][fs_algo][pred_algo][outcome][len(sel_features)]['test_metrics']['brier_loss']['conf_int'].append(brier_loss_ci)

    return results

def make_predictions_with_roc(skfold, gridcvs, X_filtered, y, table, fs_algo, results, outcome, nb_features, sel_features, smote: bool = False):
    '''Make predictions with ROC curve. 
    
    Parameters:
    ----------------
    skfold (object): The StratifiedKFold object.
    gridcvs (dict): A dictionary containing the GridSearchCV objects for each classifier.
    X_filtered (pd.DataFrame): The filtered training data features.
    y (pd.DataFrame): The target labels for the training data.
    table (str): The name of the table.
    fs_algo (str): The name of the feature selection algorithm.
    results (dict): The dictionary containing the results of the prediction algorithms.
    outcome (str): The name of the outcome to predict.
    nb_features (int): The number of features selected.
    sel_features (list): The list of selected features.
    smote (bool): Whether to use SMOTE to balance the dataset.
    
    Returns:
    ----------------
    results (dict): The updated results dictionary that will contain the results of the prediction algorithms.
    fpr (list): The false positive rate.
    tpr (list): The true positive rate.
    '''

    for c, (outer_train_idx, outer_valid_idx) in enumerate(skfold.split(X_filtered, y)):
        for pred_algo, gs_est in sorted(gridcvs.items()):
            # print('outer fold %d/5 | tuning %-8s' % (c, pred_algo), end='')
            X_train = X_filtered.iloc[outer_train_idx]
            y_train = y.iloc[outer_train_idx]
            X_test = X_filtered.iloc[outer_valid_idx]
            y_test = y.iloc[outer_valid_idx]
            if smote: # use smote to balance the dataset
                sm = SMOTE(random_state=42, sampling_strategy='minority')
                X_train, y_train = sm.fit_resample(X_train, y_train) 
            # The inner loop for hyperparameter tuning
            gs_est.fit(X_train, y_train) # hyperparameter tuning
            optimal_threshold = compute_opt_threshold(gs_est, X_train, y_train) # compute optimal threshold

            # Computing the test metrics
            test_auc, sensitivity, specificity, fpr, tpr = compute_and_plot_test_metrics(gs_est, X_test, y_test, optimal_threshold)
        
            # plot r
            # save results 
            results[table][fs_algo][pred_algo][outcome][nb_features]['features'] = sel_features
            results[table][fs_algo][pred_algo][outcome][nb_features]['params'] = gs_est.best_params_ #  params of best algo (based on cross validation search) trained again 
            results[table][fs_algo][pred_algo][outcome][nb_features]['train_auc'] = gs_est.best_score_ # score of best algo (based on cross validation search) trained again 
            results[table][fs_algo][pred_algo][outcome][nb_features]['test_auc'].append(test_auc)
            results[table][fs_algo][pred_algo][outcome][nb_features]['sensitivity'].append(sensitivity)
            results[table][fs_algo][pred_algo][outcome][nb_features]['specificity'].append(specificity)

    return results, fpr, tpr # return what's needed for roc curve plotting: y_test, model, X_test, 


def compute_metric(X_val: np.ndarray, y_val: np.ndarray, model):
    y_pred = model.predict(X_val) # get predictions 

    y_val = y_val.astype('int64')

    # get sensitivity, specificity
    confmat = confusion_matrix(y_val, y_pred)
    try: 
        tn, fp, fn, tp = confmat.ravel()
    except ValueError:
        print("Confusion matrix is not well formatted.")
        return 0, 0, 0, []
    if tp+fn == 0: 
        print("Division by zero")
        return 0, 0, 0, []
    if tn+fp == 0:
        print("Division by zero")
        return 0, 0, 0, []
    else: 
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)

    # Get ROC AUC
    try:
        y_prob = model.predict_proba(X_val)[:, 1]
        x_axis, y_axis, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(x_axis, y_axis)
    except AttributeError: 
        roc_auc = 'N/A'
    # Get mispredictions
    mispreds = []
    for idx, (true, pred) in enumerate(zip(y_val, y_pred)):
        if true != pred: 
            mispreds.append(idx)

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

def compute_test_metrics(gs_est: object, X_test: pd.DataFrame, y_test: pd.DataFrame, optimal_threshold):
    '''
    Compute the test metrics and their confidence intervals for a given model.

    Parameters:
    gs_est (GridSearchCV): The GridSearchCV object containing the best estimator.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.DataFrame): The test data labels.
    optimal_threshold (float): The optimal threshold for the model.

    Returns:
    results (dict): The updated results dictionary that will contain the results of the prediction algorithms.
    '''
    outer_y_prob = gs_est.best_estimator_.predict_proba(X_test)[:, 1]
    # transform y_test to a numpy array
    y_test = y_test.to_numpy().flatten()
    assert y_test.shape == outer_y_prob.shape, "Shapes are not the same"
    idx = np.arange(y_test.shape[0])

    # Diagnostic checks
    unique_probs = np.unique(outer_y_prob)
    if len(unique_probs) == 1:
        print("Warning: All predictions are the same.")

    unique_classes = np.unique(y_test)
    if len(unique_classes) == 1:
        print("Warning: y_test contains only one class.")

    if np.isnan(outer_y_prob).any():
        print("Warning: outer_y_prob contains NaN values.")

    brier_loss_list = []
    test_auc_list = []
    sensitivity_list = []
    specificity_list = []

    for i in range(200): 
        pred_idx = np.random.choice(idx, size=idx.shape[0], replace=True)

        brier_loss = brier_score_loss(y_test[pred_idx], outer_y_prob[pred_idx]) 
        fpr, tpr, _ = roc_curve(y_test[pred_idx], outer_y_prob[pred_idx]) 
        test_auc = auc(fpr, tpr) 
        if np.isnan(test_auc) == False:
            test_auc_list.append(test_auc)


        # compute sensitivity and specificity based on y_pred 
        outer_y_pred = (outer_y_prob >= optimal_threshold).astype(int) # threshold obtained on train set 
        tn, fp, fn, tp = confusion_matrix(y_test, outer_y_pred).ravel()
        sensitivity = tp / (tp + fn) 
        specificity = tn / (tn + fp)  

        if np.isnan(sensitivity) == False:
            sensitivity_list.append(sensitivity)
        if np.isnan(specificity) == False:
            specificity_list.append(specificity)
        if np.isnan(brier_loss) == False:
            brier_loss_list.append(brier_loss)
            
    if len(test_auc_list) == 0: 
        raise ValueError("No AUC values were computed.")
    elif len(sensitivity_list) == 0:
        raise ValueError("No sensitivity values were computed.")
    elif len(specificity_list) == 0:
        raise ValueError("No specificity values were computed.")
    elif len(brier_loss_list) == 0:
        raise ValueError("No Brier loss values were computed.")
    else: 
        brier_loss = np.mean(brier_loss_list)
        brier_loss_ci = (np.percentile(brier_loss_list, 2.5), np.percentile(brier_loss_list, 97.5))
        
        test_auc = np.mean(test_auc_list)
        test_auc_ci = (np.percentile(test_auc_list, 2.5), np.percentile(test_auc_list, 97.5))
    
        sensitivity = np.mean(sensitivity_list)
        sensitivity_ci = (np.percentile(sensitivity_list, 2.5), np.percentile(sensitivity_list, 97.5))
    
        specificity = np.mean(specificity_list)
        specificity_ci = (np.percentile(specificity_list, 2.5), np.percentile(specificity_list, 97.5))

    return brier_loss, brier_loss_ci, test_auc, test_auc_ci, sensitivity, sensitivity_ci, specificity, specificity_ci

def compute_uncertainty_test_metrics(gs_est, X_test, y_test, optimal_threshold):
    '''
    Compute the test metrics for a given model using uncertainty thresholding.

    Parameters:
    gs_est (GridSearchCV): The GridSearchCV object containing the best estimator.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels.
    optimal_threshold (float): The optimal threshold for the model.

    Returns:
    float: The test AUC.
    float: The sensitivity.
    float: The specificity.
    '''

    # compute outer auc
    outer_y_prob = gs_est.best_estimator_.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, outer_y_prob)
    test_auc = auc(fpr, tpr)

    # compute sensitivity and specificity based on y_pred 
    outer_y_pred = (outer_y_prob >= optimal_threshold).astype(int) # threshold obtained on train set 
    tn, fp, fn, tp = confusion_matrix(y_test, outer_y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return test_auc, sensitivity, specificity

def compute_and_plot_test_metrics(gs_est, X_test, y_test, optimal_threshold):
    '''
    Compute the test metrics for a given model.

    Parameters:
    gs_est (GridSearchCV): The GridSearchCV object containing the best estimator.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels.
    optimal_threshold (float): The optimal threshold for the model.

    Returns:
    float: The test AUC.
    float: The sensitivity.
    float: The specificity.
    list: The false positive rate.
    list: The true positive rate.
    '''

    # compute outer auc
    outer_y_prob = gs_est.best_estimator_.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, outer_y_prob)
    test_auc = auc(fpr, tpr)

    # compute sensitivity and specificity based on y_pred 
    outer_y_pred = (outer_y_prob >= optimal_threshold).astype(int) # threshold obtained on train set 
    tn, fp, fn, tp = confusion_matrix(y_test, outer_y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return test_auc, sensitivity, specificity, fpr, tpr