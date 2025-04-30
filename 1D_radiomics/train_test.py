'''Main script to run the feature selection and prediction algorithms comparison on the dataset. '''

import json 
import os 
import time
import numpy as np 
import random 

import utils.src.feature_selection as fsa 
import utils.src.dataset as dataset
import utils.src.test as test
import utils.src.train as train

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


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
            X, y, features_list = dataset.get_xy(os.path.join(folder_path, table), os.path.join(folder_path, outcome_csv), outcome, smote=smote)
            
            # Normalization 
            for fs_algo in feat_sel_algo_list:
                if 'CHI2' in fs_algo:
                    min_max_scaler = MinMaxScaler()
                    X = min_max_scaler.fit_transform(X)
                else:     
                    znorm_scaler = StandardScaler()
                    X = znorm_scaler.fit_transform(X)
                
                # Split the dataset into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y[outcome])
                y_train = np.array(y_train).reshape(-1, 1).ravel() # convert to numpy array to avoid errors
                y_test = np.array(y_test).reshape(-1, 1).ravel() # convert to numpy array

                # Feature selection
                best_features, best_feat_sel_model = fsa.get_best_features(X_train, y_train, fs_algo, features_list=features_list, max_features=max_features)
                print("Feature selection done for ", fs_algo)
                
                # PREDICTIONS
                print("Training...")
                for nb_features in range(1, max_features+1): # number of features selected
                    
                    sel_features, X_train = fsa.filter_dataset3(X_train, best_features, nb_features, features_list)
                    # check if there are NaN values in the training set dataframe
                    assert not X_train.isna().any().any(), print(X_train)
                    X_test = fsa.filter_dataset3(X_test, best_features, nb_features, features_list)[1] # filter the test set with the same features as the train set
                    for pred_algo in pred_algo_list:
                        best_model = train.train_model(pred_algo, X_train, y_train) # train the model                       
                        optimal_threshold, train_auc, train_brier_loss = train.compute_opt_threshold(best_model, X_train, y_train) # compute optimal threshold based on train set results
                        brier_loss, brier_loss_ci, test_auc, test_auc_ci, sensitivity, sensitivity_ci, specificity, specificity_ci = test.compute_test_metrics(best_model, X_test, y_test, optimal_threshold) # bootstraping
                        results = test.save_model_results(results, table, fs_algo, pred_algo, outcome, sel_features, best_feat_sel_model, best_model) # save the best features in a file
                        results = test.save_results(results, table, fs_algo, pred_algo, outcome, sel_features, train_auc, train_brier_loss, test_auc, sensitivity, specificity, brier_loss, test_auc_ci, sensitivity_ci, specificity_ci, brier_loss_ci)
                    print("Predictions done for ", len(sel_features), " features.")

    results_ser = dataset.convert_to_list(results)

    with open(results_file, 'w') as f: 
        json.dump(results_ser, f)                     
    print("Results saved in {} file.".format(results_file))    

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__": 

    main('/home/tachennf/Documents/delta-rad/1D_radiomics/train_test.yaml')
