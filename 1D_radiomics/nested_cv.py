'''Main script to run the feature selection and prediction algorithms comparison on the dataset. '''

import pandas as pd 
import json 
import os 
import time
import numpy as np 
import random 
import yaml 

import utils.feature_selection as fsa 
import utils.dataset as dataset
import utils.test as test
import utils.train as train

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

np.random.seed(42)
random.seed(42)
print("Seed example: ", random.randint(0, 100))

def main(param_file: str): 
    start_time = time.time()

    ###################### INITIALIZATION ##############################
    params = dataset.load_config(param_file)
    folder_path = params['paths']['data_folder_path']
    outcome_csv = params['paths']['outcome_csv_file']
    results_file = params['paths']['results_file']   
    delta_rad_tables = params['paths']['delta_rad_tables']

    outcomes_list = params['parameters']['outcomes_list']
    feat_sel_algo_list = params['parameters']['feat_sel_algo_list']
    pred_algo_list = params['parameters']['pred_algo_list']
    max_features = params['parameters']['max_features']
    smote = params['parameters']['smote']

    results = test.def_results_dict(delta_rad_tables, feat_sel_algo_list, pred_algo_list, outcomes_list, max_features)

    ###################### MAIN LOOP ##############################
    for table in delta_rad_tables:
        print('Training on table ', table)
        for outcome in outcomes_list: 
            print("Training for outcome ", outcome)
            # Load the dataset 
            X, y, features_list = dataset.get_xy(os.path.join(folder_path, table), os.path.join(folder_path, outcome_csv), outcome)
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y[outcome])
            
            ###################### FEATURE SELECTION WITHOUT CVAL ##############################
            for fs_algo in feat_sel_algo_list:
                if 'CHI2' in fs_algo:
                    min_max_scaler = MinMaxScaler()
                    X_train = min_max_scaler.fit_transform(X_train)
                else:     
                    znorm_scaler = StandardScaler()
                    X_train = znorm_scaler.fit_transform(X_train)
                best_features, best_feat_sel_model = fsa.get_best_features(X_train, y_train, fs_algo, features_list=features_list, max_features=max_features)
                print("Feature selection done for ", fs_algo)
                
                # PREDICTIONS
                print("Training...")
                for nb_features in range(1, max_features+1): # number of features selected
                    
                    sel_features, X_filtered = fsa.filter_dataset2(X, best_features, nb_features)
                    gridcvs, results = train.init_for_prediction(results, table, fs_algo, best_feat_sel_model, pred_algo_list, len(sel_features), outcome) # init pred algo
                    
                    # NESTED CV LOOP
                    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # outer folds
                    for c, (outer_train_idx, outer_valid_idx) in enumerate(skfold.split(X_filtered, y)):
                        for pred_algo, gs_est in sorted(gridcvs.items()):
                            X_train = X_filtered.iloc[outer_train_idx]
                            y_train = y.iloc[outer_train_idx]
                            X_test = X_filtered.iloc[outer_valid_idx]
                            y_test = y.iloc[outer_valid_idx]
                            
                            y_train = np.array(y_train) # convert to numpy array
                            y_train = y_train.reshape(-1, 1).ravel() # to avoid errors
                            y_test = np.array(y_test) # convert to numpy array
                            y_test = y_test.reshape(-1, 1).ravel()
                            
                            if smote: # use smote to balance the dataset
                                sm = SMOTE(random_state=42, sampling_strategy='minority')
                                X_train, y_train = sm.fit_resample(X_train, y_train) 
                            gs_est.fit(X_train, y_train) # work on grid search: hyperparameter tuning
                            optimal_threshold, train_auc, train_brier_loss = train.compute_opt_threshold(gs_est, X_train, y_train) # compute optimal threshold based on train set results
                            brier_loss, brier_loss_ci, test_auc, test_auc_ci, sensitivity, sensitivity_ci, specificity, specificity_ci = test.compute_test_metrics(gs_est, X_test, y_test, optimal_threshold)
                            results = test.save_results(results, table, fs_algo, pred_algo, outcome, sel_features, gs_est, train_auc, train_brier_loss, test_auc, sensitivity, specificity, brier_loss, test_auc_ci, sensitivity_ci, specificity_ci, brier_loss_ci)
                    print("Predictions done for ", len(sel_features), " features.")

    results_ser = dataset.convert_to_list(results)

    with open(results_file, 'w') as f: 
        json.dump(results_ser, f)                     
    print("Results saved in {} file.".format(results_file))    

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__": 
    main('/home/tachennf/Documents/delta-rad/1D_radiomics/nested_cv_settings.yaml')