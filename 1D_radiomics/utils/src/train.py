'''Functions to train ML models. '''

import pandas as pd 
import numpy as np 

from sklearn.metrics import brier_score_loss, roc_curve, auc
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB


import warnings
warnings.filterwarnings('ignore')
from .sklearn_utils import * 

SCORER = 'roc_auc' 
def init_grids(pred_algo_list: list):
    '''Initialize the structures and parameters for the prediction algorithms. 
    
    Parameters:
    ----------------
    pred_algo_list (list): The list of classifiers names for which to get the pipelines.

    Returns:
    ----------------
    gridcvs (dict): A dictionary containing the GridSearchCV objects for each classifier.
    '''
    # INITIALIZATION 
    param_grids = get_param_grids(pred_algo_list) # Get parameter grids and pipelines
    pipes = get_pipelines(pred_algo_list) # Setting up the pipelines
    gridcvs = {} # Setting up multi GridSearchCV for each classifier
    for pgrid, est, pred_algo in zip(param_grids, # one iteration for each classifier
                                pipes,
                                pred_algo_list):

        # init gridsearch for cross validation (hyper-parameter tuning)
        gcv = GridSearchCV(estimator=est,
                           param_grid=pgrid,
                           scoring='roc_auc',
                           n_jobs=-1,
                           cv=4, # inner folds
                           verbose=0,
                           refit=True) # refit the best model with the entire dataset
        gridcvs[pred_algo] = gcv
    
    return gridcvs


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
            param_grid = [{'RF__max_depth': range(1, 5, 1), 'RF__n_estimators' : range(25, 1001, 25)}]
        elif pred_algo == 'ADABOOST':
            param_grid = [{'ADABOOST__n_estimators' : range(25, 1001, 25)}]
        elif pred_algo == 'LOGREGRIDGE':
            param_grid = [{'LOGREGRIDGE__penalty': ['l2'],
                        'LOGREGRIDGE__C': np.power(10., np.arange(-3, 4))}]
        elif pred_algo =='PSVM':
            param_grid = [{'PSVM__C' : np.power(10., np.arange(-3, 4)), 'PSVM__degree': range(2, 5, 1)}]
        elif pred_algo == 'KNN':
            param_grid = [{'KNN__n_neighbors': range(1, 10)}]
        elif pred_algo == 'BAGG':
            param_grid = [{'BAGG__n_estimators' : range(25, 1001, 25)}]
        elif pred_algo == 'MLP':
            param_grid = [{
                'MLP__alpha' : 10.0 ** -np.arange(2, 5), 
                'MLP__learning_rate_init': 10.0 ** -np.arange(2, 5)            }]
        elif pred_algo == 'QDA':
            param_grid = [{'QDA__reg_param': np.power(10., np.arange(-3, 4))}]
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
            pipeline = Pipeline([('RF', RandomForestClassifier(random_state=42))])
        elif pred_algo == 'ADABOOST':
            pipeline = Pipeline([('ADABOOST', AdaBoostClassifier(random_state=42, algorithm='SAMME'))])
        elif pred_algo == 'LOGREGRIDGE':
            pipeline = Pipeline([('LOGREGRIDGE', LogisticRegression(multi_class='multinomial', solver='newton-cg', random_state=42))])
        elif pred_algo == 'PSVM':
            pipeline = Pipeline([('PSVM', SVC(kernel='poly', coef0=0, gamma=1.0, probability=True, random_state=42))])
        elif pred_algo == 'KNN':
            pipeline = Pipeline([('KNN', KNeighborsClassifier())])
        elif pred_algo == 'BAGG':
            pipeline = Pipeline([('BAGG', BaggingClassifier(oob_score=True))])
        elif pred_algo == 'MLP':
            pipeline = Pipeline([('MLP', MLPClassifier(random_state=42, max_iter=2000, hidden_layer_sizes=(100, 100), solver='adam', learning_rate='invscaling'))])
        elif pred_algo == 'QDA':
            pipeline = Pipeline([('QDA', QuadraticDiscriminantAnalysis())])
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

    param_grid = {'max_depth': range(1, 5, 1), 'n_estimators' : range(25, 1001, 25)} # maximimum depth and number of estimators tuning

    estimator = RandomForestClassifier(random_state=42) 

    grid_rf = hyper_parameters_search(estimator, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5) 

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
    grid_dt = hyper_parameters_search(estimator, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

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

    param_grid = {'n_estimators' : range(25, 1001, 25)}

    estimator = AdaBoostClassifier(random_state=42, algorithm='SAMME') 

    grid_adab = hyper_parameters_search(estimator, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)
    
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

    grid_lsvm = hyper_parameters_search(lsvm, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

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

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'degree': range(2, 5, 1)} # number of estimators tuning
    ksvm = SVC( kernel="poly", coef0=0, gamma=1.0, probability=True)
    grid_ksvm = hyper_parameters_search(ksvm, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

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

    grid_knn = hyper_parameters_search(knn, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

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
        param_grid = {'solver': ['newton-cg']} 
    else:
        param_grid = {'C': np.power(10., np.arange(-3, 4))} # C is the inverse of regularization stength, as in SVM 

    logreg = LogisticRegression(multi_class='multinomial', penalty=penalty, random_state=42, solver='newton-cg') 

    grid_logreg = hyper_parameters_search(logreg, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

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

    param_grid = {'n_estimators' : range(25, 1001, 25)}

    bagg = BaggingClassifier(oob_score=True)

    grid_bagging = hyper_parameters_search(bagg, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

    return grid_bagging.best_estimator_

def train_mlp(X_train_filtered, y_train):
    param_grid = {
        'alpha' : 10.0 ** -np.arange(2, 5), 
        'learning_rate_init': 10.0 ** -np.arange(2, 5),
    }

    mlp = MLPClassifier(random_state=42, max_iter=2000, hidden_layer_sizes=(100, 100), solver='adam', learning_rate='invscaling')
    grid_mlp = hyper_parameters_search(mlp, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

    return grid_mlp.best_estimator_

def train_lda(X_train_filtered, y_train):

    clf = LinearDiscriminantAnalysis()
    lda_model = clf.fit(X_train_filtered, y_train)

    return lda_model

def train_qda(X_train_filtered, y_train):

    param_grid = {
        'reg_param': np.power(10., np.arange(-3, 4))} # regularization strength tuning
    clf = QuadraticDiscriminantAnalysis()

    grid_qda = hyper_parameters_search(clf, X_train_filtered, y_train, param_grid, scorer=SCORER, cv=5)

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

def compute_cv_opt_threshold(gs_est, X_train, y_train): 
    '''
    Compute the optimal threshold for a given model based on Youden's J statistic. Also reports the train AUC and Brier loss.
    Save the model too. 

    Parameters:
    gs_est (GridSearchCV): The GridSearchCV object containing the best estimator.
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels.

    Returns:
    float: The optimal threshold for the model. 
    float: The train AUC.
    float: The train Brier loss.
    '''
    
    best_model = gs_est.best_estimator_  # Best model from inner CV
    y_prob = best_model.predict_proba(X_train)[:, 1]  # Inner validation set probabilities
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    train_auc = auc(fpr, tpr)
    train_brier_loss = brier_score_loss(y_train, y_prob)

    # Determine the optimal threshold using Youden's J statistic
    J_scores = tpr - fpr
    optimal_idx = J_scores.argmax()
    optimal_threshold = thresholds[optimal_idx] 

    return optimal_threshold, train_auc, train_brier_loss

def compute_opt_threshold(best_model, X_train, y_train): 
    '''
    Compute the optimal threshold for a given model based on Youden's J statistic. Also reports the train AUC and Brier loss.
    Save the model too. 

    Parameters:
    best_model (object): The best estimator.
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels.

    Returns:
    float: The optimal threshold for the model. 
    float: The train AUC.
    float: The train Brier loss.
    '''
    
    y_prob = best_model.predict_proba(X_train)[:, 1]  # Inner validation set probabilities
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    train_auc = auc(fpr, tpr)
    train_brier_loss = brier_score_loss(y_train, y_prob)

    # Determine the optimal threshold using Youden's J statistic
    J_scores = tpr - fpr
    optimal_idx = J_scores.argmax()
    optimal_threshold = thresholds[optimal_idx] 

    return optimal_threshold, train_auc, train_brier_loss