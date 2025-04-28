'''Train the radiomic models on the full dataset. No test set is used.'''


import pandas as pd 
import json 
import os 
import time
import numpy as np 
import random 
import joblib 

import utils.src.feature_selection as fsa 
import utils.src.dataset as dataset
import utils.src.test as test
import utils.src.train as train

from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    models_folder = params['paths']['models_folder']
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

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

            X, y, features_list = dataset.get_xy(os.path.join(folder_path, table), os.path.join(folder_path, outcome_csv), outcome, smote=smote) # Load the dataset 
            y = np.array(y).reshape(-1, 1).ravel() 
            ###################### FEATURE SELECTION WITHOUT CVAL ##############################
            for fs_algo in feat_sel_algo_list:
                if 'CHI2' in fs_algo:
                    min_max_scaler = MinMaxScaler()
                    X = min_max_scaler.fit_transform(X)
                else:     
                    znorm_scaler = StandardScaler()
                    X = znorm_scaler.fit_transform(X)
                best_features, best_feat_sel_model = fsa.get_best_features(X, y, fs_algo, features_list=features_list, max_features=max_features)
                print("Feature selection done for ", fs_algo)
                # PREDICTIONS
                print("Training...")
                for nb_features in range(1, max_features+1): # number of features selected
                    sel_features, X_filtered = fsa.filter_dataset3(X, best_features, nb_features, features_list)
                    gridcvs = train.init_grids(pred_algo_list) # init pred algo and CV grid 
 
                    # CV LOOP
                    for pred_algo, gs_est in sorted(gridcvs.items()):
                        print("Training for ", pred_algo)   
                        gs_est.fit(X_filtered, y) # work on grid search: hyperparameter tuning
                        model_path = models_folder + table + "_" + fs_algo + "_" + pred_algo + "_" + str(nb_features) + ".joblib"
                        joblib.dump(gs_est.best_estimator_, model_path)
                        results = test.save_model_results(results, table, fs_algo, pred_algo, outcome, sel_features, best_feat_sel_model, gs_est) # save the best features in a file
                        print("Predictions done for ", len(sel_features), " features.")

    results_ser = dataset.convert_to_list(results)
 
    with open(results_file, 'w') as f: 
        json.dump(results_ser, f)                     
    print("Results saved in {} file.".format(results_file))    

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__": 
    main('/home/tachennf/Documents/delta-rad/1D_radiomics/full_training_settings.yaml')