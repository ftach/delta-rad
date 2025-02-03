'''Main script to run the feature selection and prediction algorithms comparison on the dataset. '''

import pandas as pd 
import json 
import os 
import time
import numpy as np 
import random 

import feature_selection as fsa 
import dataset 
import predictions as p 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc



np.random.seed(42)
random.seed(42)

def main(): 
    start_time = time.time()

    ###################### INITIALIZATION ##############################
    folder_path = '/home/tachennf/Documents/delta-rad/extracted_radiomics/'
    delta_rad_tables = ['rd_f1_f5_gtv.csv']
    feat_sel_algo_list = ['RF', 'ADABOOST', 'ANOVA_PERC', 'ANOVA_K_BEST', 'CHI2_PERC', 'CHI2_K_BEST', 'MI_PERC', 'MI_K_BEST', 'NO_SEL', 'RDM_SEL', 'LASSO']
    outcome_csv = 'outcomes.csv'
    results_file = 'results_ncv_test.json'
    dset_selection_method = 'random' 
    # pred_algo_list = ['DT', 'RF', 'ADABOOST', 'PSVM', 'KNN', 'LOGREG', 'LOGREGRIDGE', 'BAGG', 'MLP', 'LDA', 'QDA', 'NaiveB'] # 'LSVM', 
    pred_algo_list = ['RF', 'ADABOOST', 'LOGREG']
    MAX_FEATURES = 3
    outcomes_list = ['Récidive Locale'] # 'Récidive Méta', 

    for table in delta_rad_tables: 
        for outcome in outcomes_list: 
            # Load the dataset 
            X, y, features_list = dataset.get_xy(os.path.join(folder_path, table), os.path.join(folder_path, outcome_csv), outcome)

            # Get parameter grids and pipelines
            # TODO

            # Init classifiers 
            # clf1 = LogisticRegression(multi_class='multinomial',
            #                       solver='newton-cg',
            #                       random_state=42)
            # clf2 = KNeighborsClassifier(algorithm='ball_tree',
            #                             leaf_size=50)
            # clf3 = DecisionTreeClassifier(random_state=42)

            # Setting up the parameter grids 
            param_grids = p.get_param_grids(pred_algo_list)
            # param_grid1 = [{'clf1__penalty': ['l2'],
            #             'clf1__C': np.power(10., np.arange(-4, 4))}]
# 
            # param_grid2 = [{'clf2__n_neighbors': list(range(1, 10)),
            #                 'clf2__p': [1, 2]}]
# 
            # param_grid3 = [{'max_depth': list(range(1, 10)) + [None],
            #                 'criterion': ['gini', 'entropy']}]

            # Setting up the pipelines
            pipes = p.get_pipelines(pred_algo_list)

            # Setting up multi GridSearchCV for each classifier
            gridcvs = {}

            for pgrid, est, name in zip(param_grids,
                                        pipes,
                                        pred_algo_list):
                gcv = GridSearchCV(estimator=est,
                                   param_grid=pgrid,
                                   scoring='roc_auc',
                                   n_jobs=1,
                                   cv=4, # inner folds
                                   verbose=1,
                                   refit=True) # refit the best model with the entire dataset
                gridcvs[name] = gcv

                cv_scores = {name: [] for name in gridcvs.keys()}

            skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # outer folds

            # The outer loop for algorithm selection
            for c, (outer_train_idx, outer_valid_idx) in enumerate(skfold.split(X, y)):
                for name, gs_est in sorted(gridcvs.items()):
                    print('outer fold %d/5 | tuning %-8s' % (c, name), end='')
                    
                    # The inner loop for hyperparameter tuning
                    gs_est.fit(X.iloc[outer_train_idx], y.iloc[outer_train_idx])
                    
                    # Get optimal threshold from ROC analysis 
                    optimal_threshold = p.compute_opt_threshold(gs_est, X.iloc[outer_train_idx], y.iloc[outer_train_idx])
                    
                    # Compute test metrics 
                    test_auc, sensitivity, specificity = p.compute_test_metrics(gs_est, X.iloc[outer_valid_idx], y.iloc[outer_valid_idx], optimal_threshold)

                    # print inner auc and outter auc, sensitivity and specificity
                    print(' | inner AUC: %.2f%% outer AUC: %.2f%% sens: %.2f%% spec: %.2f%%' % (
                        100 * gs_est.best_score_, 100 * test_auc, 100 * sensitivity, 100 * specificity))
                    cv_scores[name].append(test_auc)

            # Looking at the results
            for name in cv_scores:
                print('%-8s | outer CV AUC: %.2f%% +\- %.3f' % (
                      name, 100 * np.mean(cv_scores[name]), 100 * np.std(cv_scores[name])))
            # get name of the best algorithm based on the best mean AUC
            best_algo_name = max(cv_scores, key=lambda key: np.mean(cv_scores[key]))
            print('Best algorithm: %s' % best_algo_name)
            best_algo = gridcvs[best_algo_name]

            # Fitting a model to the whole training set using the "best" algorithm
            best_algo.fit(X.iloc[outer_train_idx], y.iloc[outer_train_idx])
            train_auc = best_algo.best_score_
            test_auc, sensitivity, specificity = p.compute_test_metrics(best_algo, X.iloc[outer_valid_idx], y.iloc[outer_valid_idx], optimal_threshold)
        
                        
            print('Best Parameters: %s' % gridcvs[best_algo_name].best_params_)
            print('Training AUC when re-trained on whole train set: %.2f%%' % (train_auc))
            print('Test AUC: %.2f%%' % (test_auc))
            print('Sensitivity: %.2f%%' % (sensitivity))
            print('Specificity: %.2f%%' % (specificity))

if __name__ == "__main__": 
    main()