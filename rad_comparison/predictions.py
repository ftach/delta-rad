'''Functions to predict outcome with filtered dataset after feature selection. '''

import pandas as pd 
import numpy as np 
from scipy.stats import entropy

from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import brier_score_loss

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from collections import Counter


import warnings
warnings.filterwarnings('ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import confusion_matrix, roc_curve, auc

import utils.sklearn_utils as sku 

SCORER = 'f1' # 'roc_auc' 

def init_for_prediction(results, table, fs_algo, best_feat_sel_model, pred_algo_list, nb_features, outcome):
    '''Initialize the parameters for the prediction algorithms. 
    
    Parameters:
    ----------------
    results (dict): The dictionary containing the results of the prediction algorithms.
    table (str): The name of the table.
    fs_algo (str): The name of the feature selection algorithm.
    best_feat_sel_model (object): The best feature selection model.
    pred_algo_list (list): The list of classifiers names for which to get the pipelines.
    nb_features (int): The number of features selected.
    sel_features (list): The list of selected features.
    outcome (str): The name of the outcome to predict.

    Returns:
    ----------------
    gridcvs (dict): A dictionary containing the GridSearchCV objects for each classifier.
    results (dict): The initialized results dictionary that will contain the results of the prediction algorithms.
    '''
    # INIT PRED ALGO RESULTS
    if best_feat_sel_model is not None:
        try: 
            results[table][fs_algo]['params'] = best_feat_sel_model.best_params_
        except AttributeError:
            results[table][fs_algo]['params'] = best_feat_sel_model.get_params()
    else: 
        results[table][fs_algo]['params'] = 'no_feature_selection'

    ###################### PREDICTION ####################################
    # INITIALIZATION 
    param_grids = get_param_grids(pred_algo_list) # Get parameter grids and pipelines
    pipes = get_pipelines(pred_algo_list) # Setting up the pipelines
    gridcvs = {} # Setting up multi GridSearchCV for each classifier
    for pgrid, est, pred_algo in zip(param_grids, # one iteration for each classifier
                                pipes,
                                pred_algo_list):
        # init results dict
        if not nb_features in results[table][fs_algo][pred_algo][outcome].keys():
            results[table][fs_algo][pred_algo][outcome][nb_features] = {}
            results[table][fs_algo][pred_algo][outcome][nb_features]['features'] = []
            results[table][fs_algo][pred_algo][outcome][nb_features]['params'] = []
            results[table][fs_algo][pred_algo][outcome][nb_features]['test_auc'] = [] 
            results[table][fs_algo][pred_algo][outcome][nb_features]['sensitivity'] = []
            results[table][fs_algo][pred_algo][outcome][nb_features]['specificity'] = []
            results[table][fs_algo][pred_algo][outcome][nb_features]['brier_loss'] = []
            # results[table][fs_algo][pred_algo][outcome][nb_features]['uncertain_preds'] = []

        # init gridsearch of each classifier
        gcv = GridSearchCV(estimator=est,
                           param_grid=pgrid,
                           scoring='roc_auc',
                           n_jobs=1,
                           cv=4, # inner folds
                           verbose=0,
                           refit=True) # refit the best model with the entire dataset
        gridcvs[pred_algo] = gcv
    
    return gridcvs, results

def make_predictions(skfold, gridcvs, X_filtered, y, table, fs_algo, results, outcome, nb_features, sel_features, smote: bool = False):
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
            test_auc, sensitivity, specificity, brier_loss = compute_test_metrics(gs_est, X_test, y_test, optimal_threshold)
            # print(' | Train AUC: %.2f%% Test AUC: %.2f%% sens: %.2f%% spec: %.2f%%' % (
            #     100 * gs_est.best_score_, 100 * test_auc, 100 * sensitivity, 100 * specificity))
                
            # save results 
            results[table][fs_algo][pred_algo][outcome][nb_features]['features'] = sel_features
            results[table][fs_algo][pred_algo][outcome][nb_features]['params'] = gs_est.best_params_ #  params of best algo (based on cross validation search) trained again 
            results[table][fs_algo][pred_algo][outcome][nb_features]['train_auc'] = gs_est.best_score_ # score of best algo (based on cross validation search) trained again 
            results[table][fs_algo][pred_algo][outcome][nb_features]['test_auc'].append(test_auc)
            results[table][fs_algo][pred_algo][outcome][nb_features]['sensitivity'].append(sensitivity)
            results[table][fs_algo][pred_algo][outcome][nb_features]['specificity'].append(specificity)
            results[table][fs_algo][pred_algo][outcome][nb_features]['brier_loss'].append(brier_loss)
            # results[table][fs_algo][pred_algo][outcome][nb_features]['uncertain_preds'].append(uncertain_preds)

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


def get_param_grids(pred_algo_list: list): 
    """
    Returns the parameter grids for each classifier in the list.
    
    Parameters:
    pred_algo_list (list): The list of classifiers names for which to get the parameter grids.

    Returns:
    dict: A dictionary containing the parameter grids for each classifier.
    """
    param_grids = []
    for pred_algo in pred_algo_list:
        if pred_algo == 'RF': 
            param_grid = [{'RF__max_depth': range(1, 5, 4), 'RF__n_estimators' : range(25, 50, 25)}]
        elif pred_algo == 'ADABOOST':
            param_grid = [{'ADABOOST__n_estimators' : range(25, 50, 25)}]
        elif pred_algo == 'LOGREGRIDGE':
            param_grid = [{'LOGREGRIDGE__penalty': ['l2'],
                        'LOGREGRIDGE__C': np.power(10., np.arange(-4, 4))}]
        elif pred_algo =='PSVM':
            param_grid = [{'PSVM__C' : list(np.arange(0.01, 0.11, 0.01)), 'PSVM__degree': range(2, 5, 1)}]
        elif pred_algo == 'KNN':
            param_grid = [{'KNN__n_neighbors': range(1, 10)}]
        elif pred_algo == 'BAGG':
            param_grid = [{'BAGG__n_estimators' : range(25, 1001, 25)}]
        elif pred_algo == 'MLP':
            param_grid = [{
                'MLP__alpha' : 10.0 ** -np.arange(2, 5), 
                'MLP__learning_rate_init': 10.0 ** -np.arange(2, 5)            }]
        elif pred_algo == 'QDA':
            param_grid = [{'QDA__reg_param': list(np.arange(0.01, 0.11, 0.01))}]
        param_grids.append(param_grid)

    return param_grids
    

def get_pipelines(pred_algo_list: list):
    '''Returns the pipelines for each classifier in the list. 
    
    Parameters:
    pred_algo_list (list): The list of classifiers names for which to get the pipelines.
    
    Returns:
    dict: A dictionary containing the pipelines for each classifier.
    '''
    pipelines = []
    for pred_algo in pred_algo_list:
        if pred_algo == 'RF':
            pipeline = Pipeline([('scaler', StandardScaler()), ('RF', RandomForestClassifier(random_state=42))])
        elif pred_algo == 'ADABOOST':
            pipeline = Pipeline([('scaler', StandardScaler()), ('ADABOOST', AdaBoostClassifier(random_state=42, algorithm='SAMME'))])
        elif pred_algo == 'LOGREGRIDGE':
            pipeline = Pipeline([('scaler', StandardScaler()), ('LOGREGRIDGE', LogisticRegression(multi_class='multinomial', solver='newton-cg', random_state=42))])
        elif pred_algo == 'PSVM':
            pipeline = Pipeline([('scaler', StandardScaler()), ('PSVM', SVC(kernel='poly', coef0=0, gamma=1.0, probability=True, random_state=42))])
        elif pred_algo == 'KNN':
            pipeline = Pipeline([('scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])
        elif pred_algo == 'BAGG':
            pipeline = Pipeline([('scaler', StandardScaler()), ('BAGG', BaggingClassifier(oob_score=True))])
        elif pred_algo == 'MLP':
            pipeline = Pipeline([('scaler', StandardScaler()), ('MLP', MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 100), solver='adam', learning_rate='invscaling'))])
        elif pred_algo == 'QDA':
            pipeline = Pipeline([('scaler', StandardScaler()), ('QDA', QuadraticDiscriminantAnalysis())])
        pipelines.append(pipeline)

    return pipelines
    

def train_rf(X_train_filtered, y_train): 
    """
    Trains a Random Forest classifier using grid search for hyperparameter tuning.

    Parameters:
    X_train_filtered (pd.DataFrame or np.ndarray): The training data features.
    y_train (pd.Series or np.ndarray): The target labels for the training data.

    Returns:
    RandomForestClassifier: The best estimator found by the grid search.
    """

    param_grid = {'max_depth': range(1, 5, 4), 'n_estimators' : range(25, 50, 25)} # maximimum depth and number of estimators tuning
    # param_grid = {'n_estimators' : range(25, 1001, 25)}

    estimator = RandomForestClassifier(random_state=42) 

    grid_rf = sku.hyper_parameters_search(estimator, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5) 

    return grid_rf.best_estimator_

def train_dt(X_train_filtered, y_train):
    """
    Trains a Decision Tree classifier using the provided training data and performs hyperparameter search.

    Parameters:
    X_train_filtered (pd.DataFrame or np.ndarray): The filtered training data features.
    y_train (pd.Series or np.ndarray): The training data labels.

    Returns:
    DecisionTreeClassifier: The best estimator found by the hyperparameter search.
    """

    param_grid = {'min_samples_split': range(2, 60, 5)} # minimum number of samples required to split an internal node
    estimator = DecisionTreeClassifier(random_state=42, criterion='gini') # we can the change the criterion to entropy 

    #HYPER PARAMETER SEARCH
    grid_dt = sku.hyper_parameters_search(estimator, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

    return grid_dt.best_estimator_

def train_adaboost(X_train_filtered, y_train):
    """
    Trains an AdaBoost classifier using the provided training data and performs hyperparameter tuning.

    Parameters:
    X_train_filtered (pd.DataFrame or np.ndarray): The filtered training feature set.
    y_train (pd.Series or np.ndarray): The training labels.

    Returns:
    AdaBoostClassifier: The best estimator found by the hyperparameter search.
    """

    #param_grid = {'n_estimators' : range(25, 50, 25)} # number of estimators tuning
    param_grid = {'n_estimators' : range(25, 1001, 25)}


    estimator = AdaBoostClassifier(random_state=42, algorithm='SAMME') 

    grid_adab = sku.hyper_parameters_search(estimator, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)
    
    return grid_adab.best_estimator_  

def train_lsvc(X_train_filtered, y_train):
    """
    Trains a Linear Support Vector Classifier (LinearSVC) using grid search for hyperparameter tuning.

    Parameters:
    X_train_filtered (array-like or sparse matrix): The training input samples.
    y_train (array-like): The target values (class labels) as integers or strings.

    Returns:
    sklearn.svm.LinearSVC: The best estimator found by the grid search.
    """

    param_grid = {'C' : list(np.arange(0.01, 0.11, 0.01))} # regularization strength tuning
    lsvm = LinearSVC(random_state=42, dual=False)

    grid_lsvm = sku.hyper_parameters_search(lsvm, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

    return grid_lsvm.best_estimator_

def train_psvm(X_train_filtered, y_train):
    """
    Trains a polynomial Support Vector Machine (SVM) using grid search for hyperparameter tuning.

    Parameters:
    X_train_filtered (array-like): The training input samples.
    y_train (array-like): The target values (class labels) as integers or strings.

    Returns:
    sklearn.svm.SVC: The best estimator found by the grid search.
    """

    param_grid = {'C' : list(np.arange(0.01, 0.11, 0.01)), 'degree': range(2, 5, 1)} # number of estimators tuning
    # param_grid = {'C' : list(np.arange(0.01, 0.11, 0.01)), 'degree': [2]}
    ksvm = SVC( kernel="poly", coef0=0, gamma=1.0, probability=True)
    grid_ksvm = sku.hyper_parameters_search(ksvm, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

    return grid_ksvm.best_estimator_

def train_knn(X_train_filtered, y_train):
    """
    Trains a K-Nearest Neighbors (KNN) classifier using grid search for hyperparameter tuning.

    Parameters:
    X_train_filtered (array-like or sparse matrix): The training input samples.
    y_train (array-like): The target values (class labels) as integers or strings.

    Returns:
    KNeighborsClassifier: The best KNN classifier found by grid search.
    """

    param_grid = {'n_neighbors': range(1, 10)} # number of neighbors tuning
    knn = KNeighborsClassifier()

    grid_knn = sku.hyper_parameters_search(knn, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

    return grid_knn.best_estimator_

def train_logreg(X_train_filtered, y_train, penalty = None):
    """
    Trains a logistic regression model using the provided training data and hyperparameter search.
    Parameters:
    X_train_filtered (array-like or sparse matrix): The input samples for training.
    y_train (array-like): The target values (class labels) for training.
    penalty (str, optional): The norm used in the penalization ('l1', 'l2', 'elasticnet', 'none'). Default is None.
    Returns:
    sklearn.linear_model.LogisticRegression: The best logistic regression model found by the hyperparameter search.
    """

    if penalty is None:
        param_grid = {'solver': ['newton-cg']} # C is the inverse of regularization stength, as in SVM 
    else:
        param_grid = {'C' : list(np.arange(0.01, 0.11, 0.01))}

    logreg = LogisticRegression(multi_class='multinomial', penalty=penalty, random_state=42, solver='newton-cg') 

    grid_logreg = sku.hyper_parameters_search(logreg, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

    return grid_logreg.best_estimator_

def train_bagg(X_train_filtered, y_train):
    """
    Trains a BaggingClassifier using the provided training data and performs hyperparameter tuning.

    Parameters:
    X_train_filtered (array-like or sparse matrix): The input samples for training.
    y_train (array-like): The target values (class labels) for training.

    Returns:
    BaggingClassifier: The best estimator found by the hyperparameter search.
    """

    #param_grid = {'n_estimators' : range(25, 50, 25)} # number of estimators tuning
    param_grid = {'n_estimators' : range(25, 1001, 25)}

    bagg = BaggingClassifier(oob_score=True)

    grid_bagging = sku.hyper_parameters_search(bagg, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

    return grid_bagging.best_estimator_

def train_mlp(X_train_filtered, y_train):
    param_grid = {
        'alpha' : 10.0 ** -np.arange(2, 5), 
        'learning_rate_init': 10.0 ** -np.arange(2, 5),
    }

    # param_grid = {
    #     'alpha' : [10.0 ** -3], 
    #     'learning_rate_init': [10.0 ** -3],
    # }

    mlp = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 100), solver='adam', learning_rate='invscaling')
    grid_mlp = sku.hyper_parameters_search(mlp, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

    return grid_mlp.best_estimator_

def train_lda(X_train_filtered, y_train):

    clf = LinearDiscriminantAnalysis()
    lda_model = clf.fit(X_train_filtered, y_train)

    return lda_model

def train_qda(X_train_filtered, y_train):

    param_grid = {
        'reg_param': list(np.arange(0.01, 0.11, 0.01))} # regularization strength tuning
    clf = QuadraticDiscriminantAnalysis()

    grid_qda = sku.hyper_parameters_search(clf, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

    return grid_qda.best_estimator_ 

def train_gaussianp(X_train_filtered, y_train):

    clf = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
    gpc_model = clf.fit(X_train_filtered, y_train)

    return gpc_model

def train_naiveb(X_train_filtered, y_train):
    
        clf = GaussianNB()
        nb_model = clf.fit(X_train_filtered, y_train)
    
        return nb_model


def train_model(pred_algo: str, X_train_filtered: pd.DataFrame, y_train: pd.DataFrame): 
    #print("Beginning training with {}...".format(pred_algo))

    if pred_algo == 'RF': 
        best_model = train_rf(X_train_filtered, y_train)
    elif pred_algo == 'ADABOOST':
        best_model = train_adaboost(X_train_filtered, y_train)
    elif pred_algo == 'LOGREG':
        best_model = train_logreg(X_train_filtered, y_train, penalty=None)
    elif pred_algo == 'LOGREGRIDGE':
        best_model = train_logreg(X_train_filtered, y_train, penalty='l2')
    elif pred_algo == 'DT':
        best_model = train_dt(X_train_filtered, y_train)
    elif pred_algo == 'LSVM':
        best_model = train_lsvc(X_train_filtered, y_train)
    elif pred_algo == 'PSVM':
        best_model = train_psvm(X_train_filtered, y_train)
    elif pred_algo == 'KNN':
        best_model = train_knn(X_train_filtered, y_train)
    elif pred_algo == 'BAGG':
        best_model = train_bagg(X_train_filtered, y_train)
    elif pred_algo == 'MLP':
        best_model = train_mlp(X_train_filtered, y_train)
    elif pred_algo == 'LDA':
        best_model = train_lda(X_train_filtered, y_train)
    elif pred_algo == 'QDA':
        best_model = train_qda(X_train_filtered, y_train)
    elif pred_algo == 'GaussianP':
        best_model = train_gaussianp(X_train_filtered, y_train)
    elif pred_algo == 'NaiveB':
        best_model = train_naiveb(X_train_filtered, y_train)
                                  
    #print("Training with {} ended.".format(pred_algo))
    return best_model 

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

def compute_opt_threshold(gs_est, X_train, y_train): 
    '''
    Compute the optimal threshold for a given model based on Youden's J statistic.

    Parameters:
    gs_est (GridSearchCV): The GridSearchCV object containing the best estimator.
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels.

    Returns:
    float: The optimal threshold for the model. 
    '''
    best_model = gs_est.best_estimator_  # Best model from inner CV
    y_prob = best_model.predict_proba(X_train)[:, 1]  # Inner validation set probabilities
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)

    # Determine the optimal threshold using Youden's J statistic
    J_scores = tpr - fpr
    optimal_idx = J_scores.argmax()
    optimal_threshold = thresholds[optimal_idx] 

    return optimal_threshold

def compute_test_metrics(gs_est, X_test, y_test, optimal_threshold):
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
    float: The Brier loss.
    int: number of uncertain predictions (entropy > 0.5) 
    '''

    # compute outer auc
    outer_y_prob = gs_est.best_estimator_.predict_proba(X_test)[:, 1]
    brier_loss = brier_score_loss(y_test, outer_y_prob)
    fpr, tpr, _ = roc_curve(y_test, outer_y_prob)
    test_auc = auc(fpr, tpr)

    # # uncertainty computation 
    # try: 
    #     rf_model = gs_est.best_estimator_.named_steps['classifier']
    #     tree_probs = np.array([tree.predict_proba(X_test) for tree in rf_model.estimators_])
    #     mean_probs = np.mean(tree_probs, axis=0)
    #     entropies = np.array([entropy(prob) for prob in mean_probs])
    #     uncertain_preds = np.sum(entropies > 0.5)
    # except AttributeError:
    #     uncertain_preds = None 

    # compute sensitivity and specificity based on y_pred 
    outer_y_pred = (outer_y_prob >= optimal_threshold).astype(int) # threshold obtained on train set 
    tn, fp, fn, tp = confusion_matrix(y_test, outer_y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return test_auc, sensitivity, specificity, brier_loss #, uncertain_preds

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