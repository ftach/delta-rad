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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    feat_sel_algo_list = ['ANOVA_PERC', 'ANOVA_K_BEST'] # 'RF', 'ADABOOST', , 'MI_PERC', 'MI_K_BEST', 'NO_SEL', 'RDM_SEL', 'LASSO'
    outcome_csv = 'outcomes.csv'
    results_file = 'results_ncv_test.json'
    dset_selection_method = 'random' 
    # pred_algo_list = ['DT', 'RF', 'ADABOOST', 'PSVM', 'KNN', 'LOGREG', 'LOGREGRIDGE', 'BAGG', 'MLP', 'LDA', 'QDA', 'NaiveB'] # 'LSVM', 
    pred_algo_list = ['RF', 'ADABOOST', 'LOGREG']
    MAX_FEATURES = 3
    outcomes_list = ['Récidive Locale'] # 'Récidive Méta', 
    results = {
        table: {
            feat_sel_algo: {
                pred_algo: {
                    outcome: {}
                    for outcome in outcomes_list
                }
                for pred_algo in pred_algo_list
            }
            for feat_sel_algo in feat_sel_algo_list
        }
        for table in delta_rad_tables
    }

    ###################### MAIN LOOP ##############################
    for table in delta_rad_tables:
        print('Training on table ', table)
        if not table in results.keys():
            results[table] = {} 
        for outcome in outcomes_list: 
            print("Training for outcome ", outcome)
            # Load the dataset 
            X, y, features_list = dataset.get_xy(os.path.join(folder_path, table), os.path.join(folder_path, outcome_csv), outcome)
            
            ###################### FEATURE SELECTION WITHOUT CVAL ##############################
            # TODO: move feature selection
            for fs_algo in feat_sel_algo_list:
                X_train = X.iloc[outer_train_idx]
                y_train = y.iloc[outer_train_idx]
                if 'CHI2' in fs_algo:
                    min_max_scaler = MinMaxScaler()
                    X_train = min_max_scaler.fit_transform(X_train)
                elif 'NZV' in fs_algo:
                    continue # no scaling for nzv
                else:     
                    znorm_scaler = StandardScaler()
                    X_train = znorm_scaler.fit_transform(X_train)
                best_features, best_feat_sel_model = fsa.get_best_features(X_train, y_train, fs_algo, features_list=features_list, max_features=MAX_FEATURES)
                if fs_algo != 'NO_SEL': # if no feature selection, we don't need to train the model for each number of features
                    for nb_features in range(1, MAX_FEATURES+1): # number of features selected
                        
                        sel_features, X_filtered = fsa.filter_dataset2(X, best_features, nb_features, features_list)
                        # TODO: save the selected features in the results dict 
                        
                        ###################### PREDICTION ####################################
                        param_grids = p.get_param_grids(pred_algo_list) # Get parameter grids and pipelines
                        pipes = p.get_pipelines(pred_algo_list) # Setting up the pipelines
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
                            # Perform feature selection on the training set here??? TODO 
                            for name, gs_est in sorted(gridcvs.items()):
                                print('outer fold %d/5 | tuning %-8s' % (c, name), end='')
                                # The inner loop for hyperparameter tuning
                                gs_est.fit(X_filtered.iloc[outer_train_idx], y.iloc[outer_train_idx])
                                optimal_threshold = p.compute_opt_threshold(gs_est, X_filtered.iloc[outer_train_idx], y.iloc[outer_train_idx])
                                test_auc, sensitivity, specificity = p.compute_test_metrics(gs_est, X_filtered.iloc[outer_valid_idx], y.iloc[outer_valid_idx], optimal_threshold)
                                print(' | inner AUC: %.2f%% outer AUC: %.2f%% sens: %.2f%% spec: %.2f%%' % (
                                    100 * gs_est.best_score_, 100 * test_auc, 100 * sensitivity, 100 * specificity))
                                cv_scores[name].append(test_auc)

                                # Looking at the results
                                for name in cv_scores:
                                    print('%-8s | outer CV AUC: %.2f%% +\- %.3f' % (
                                          name, 100 * np.mean(cv_scores[name]), 100 * np.std(cv_scores[name])))
                                
                                # # get name of the best algorithm based on the best mean AUC, DOES NOT ALREADY DOES IT WITH 'refit=True' option of gridsearchcv??? 
                                # best_algo_name = max(cv_scores, key=lambda key: np.mean(cv_scores[key]))
                                # print('Best algorithm: %s' % best_algo_name)
                                # best_algo = gridcvs[best_algo_name]
                                # # Fitting a model to the whole training set using the "best" algorithm
                                # best_algo.fit(X.iloc[outer_train_idx], y.iloc[outer_train_idx])
                                # train_auc = best_algo.best_score_
                                # test_auc, sensitivity, specificity = p.compute_test_metrics(best_algo, X.iloc[outer_valid_idx], y.iloc[outer_valid_idx], optimal_threshold)
                                                                         
                                print('Best Parameters: %s' % gs_est.best_params_)
                                print('Training AUC when re-trained on whole train set: %.2f%%' % (gs_est.best_score_))
                                print('Test AUC: %.2f%%' % (test_auc))
                                print('Sensitivity: %.2f%%' % (sensitivity))
                                print('Specificity: %.2f%%' % (specificity))

if __name__ == "__main__": 
    main()