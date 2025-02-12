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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


np.random.seed(42)
random.seed(42)

def main(): 
    start_time = time.time()

    ###################### INITIALIZATION ##############################
    folder_path = '/home/tachennf/Documents/delta-rad/extracted_radiomics/'
    delta_rad_tables = ['rd_f1_f5_gtv.csv'] # 'f3_gtv.csv', 'simu_gtv.csv', 'f1_gtv.csv', 'f5_gtv.csv', 'rd_simu_f1_gtv.csv', 'rd_simu_f3_gtv.csv', 'rd_simu_f5_gtv.csv', 'rd_f1_f3_gtv.csv', 
    feat_sel_algo_list = ['ANOVA_PERC', 'RDM_SEL', 'NO_SEL', 'RF']  # # , 'ADABOOST', , 'MI_PERC', 'MI_K_BEST', 'NO_SEL', 'RDM_SEL', 'LASSO'
    outcome_csv = 'outcomes.csv'
    smote = True
    results_file = 'json_results/results_ncv_smote.json'
    pred_algo_list = ['RF', 'ADABOOST', 'LOGREGRIDGE', 'PSVM', 'KNN',  'BAGG', 'MLP', 'QDA'] # 
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

            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y[outcome])
            
            ###################### FEATURE SELECTION WITHOUT CVAL ##############################
            for fs_algo in feat_sel_algo_list:
                if 'CHI2' in fs_algo:
                    min_max_scaler = MinMaxScaler()
                    X_train = min_max_scaler.fit_transform(X_train)
                else:     
                    znorm_scaler = StandardScaler()
                    X_train = znorm_scaler.fit_transform(X_train)
                best_features, best_feat_sel_model = fsa.get_best_features(X_train, y_train, fs_algo, features_list=features_list, max_features=MAX_FEATURES)
                print("Feature selection done for ", fs_algo)
                print("Training...")
                if fs_algo != 'NO_SEL': # if no feature selection, we don't need to train the model for each number of features 
                    for nb_features in range(1, MAX_FEATURES+1): # number of features selected
                        
                        sel_features, X_filtered = fsa.filter_dataset2(X, best_features, nb_features, features_list)
                        gridcvs, results = p.init_for_prediction(results, table, fs_algo, best_feat_sel_model, pred_algo_list, nb_features, outcome) # init pred algo
                        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # outer folds

                        # OUTER LOOP FOR ALGORITHM SELECTION 
                        results = p.make_predictions(skfold, gridcvs, X_filtered, y, table, fs_algo, results, outcome, nb_features, sel_features, smote)
                        print("Predictions done for ", nb_features, " features.")

                else: # no selection 
                    best_feat_sel_model = None 
                    sel_features, X_filtered = fsa.filter_dataset2(X, best_features, nb_features, features_list)
                    gridcvs, results = p.init_for_prediction(results, table, fs_algo, best_feat_sel_model, pred_algo_list, nb_features, outcome) # init pred algo  
                    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # outer folds
 
                    # OUTER LOOP FOR ALGORITHM SELECTION  
                    results = p.make_predictions(skfold, gridcvs, X_filtered, y, table, fs_algo, results, outcome, nb_features, sel_features, smote)
                    print("Predictions done for ", nb_features, " features.")

    results_ser = dataset.convert_to_list(results)

    with open(results_file, 'w') as f: 
        json.dump(results_ser, f)                     
    print("Results saved in {} file.".format(results_file))    

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__": 
    main()