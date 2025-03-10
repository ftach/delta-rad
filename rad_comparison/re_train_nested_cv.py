'''Main script to run the feature selection and prediction algorithms comparison on the dataset. '''

import pandas as pd 
import json 
import os 
import time
import numpy as np 
import random 
import matplotlib.pyplot as plt

import feature_selection as fsa 
import dataset 
import predictions as p 

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


np.random.seed(42)
random.seed(42)

def retrain(top_results: dict, n_top_results: int): 
    start_time = time.time()

    ###################### INITIALIZATION ##############################
    folder_path = '/home/tachennf/Documents/delta-rad/extracted_radiomics/'
    delta_rad_tables = ['simu_gie_gtv.csv', 'rd_simu_onemth_gtv.csv', 'simu_onemth_gtv.csv'] # ['rd_f1_f5_gtv.csv'] # 'f3_gtv.csv', 'simu_gtv.csv', 'f1_gtv.csv', 'f5_gtv.csv', 'rd_simu_f1_gtv.csv', 'rd_simu_f3_gtv.csv', 'rd_simu_f5_gtv.csv', 'rd_f1_f3_gtv.csv', 
    nice_tables = ['SIMU', 'DELTA', '1 MONTH']
    feat_sel_algo_list = ['ANOVA_PERC', 'RDM_SEL', 'NO_SEL', 'RF']  # # , 'ADABOOST', , 'MI_PERC', 'MI_K_BEST', 'NO_SEL', 'RDM_SEL', 'LASSO'
    outcome_csv = 'outcomes.csv'
    smote = False
    results_file = 'json_results/results_ncv_simu_gie_retrain.json'
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
    cmap = plt.get_cmap('tab10')  # You can choose other colormaps like 'viridis', 'plasma', etc.
    colors = cmap(np.linspace(0, 1, n_top_results))

    plt.figure(figsize=(12, 4))
    plt.plot([0, 1], [0, 1], color=(0.6, 0.6, 0.6), linestyle='--')
    plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':',
             color='black', label='Perfect classifier')
    k = 0
    j = 0
    for table in top_results.keys():
        print('Training again on table ', table)
        if not table in results.keys():
            results[table] = {} 
        for outcome in outcomes_list: 
            print("Training again for outcome ", outcome)
            # Load the dataset 
            for i in range(len(top_results[table][outcome])): 
                result = top_results[table][outcome][i]
                test_auc = result[0]
                pred_algo = result[1]
                fs_algo = result[2]
                features = result[3]
                X, y, features_list = dataset.get_xy(os.path.join(folder_path, table), os.path.join(folder_path, outcome_csv), outcome)
                X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y[outcome])
                znorm_scaler = StandardScaler()
                X_train = znorm_scaler.fit_transform(X_train)

                best_features, best_feat_sel_model = fsa.get_best_features(X_train, y_train, fs_algo, features_list=features_list, max_features=MAX_FEATURES)


                sel_features, X_filtered = fsa.filter_dataset2(X, best_features, len(features), features_list)
                gridcvs, results = p.init_for_prediction(results, table, fs_algo, best_feat_sel_model, pred_algo_list, len(features), outcome) # init pred algo
                skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # outer folds

                # OUTER LOOP FOR ALGORITHM SELECTION 
                results, fpr, tpr = p.make_predictions_with_roc(skfold, gridcvs, X_filtered, y, table, fs_algo, results, outcome, len(features), sel_features, smote)
                print("Predictions done for ", len(features), " features.")

                # SAVE ROC CURVES FOR BEST ALGORITHMS
                plt.plot(fpr, tpr, lw=2, label='%s %s %s| AUC = %0.3f' %
                         (nice_tables[k], pred_algo, fs_algo, test_auc), color=colors[j])
                j += 1
        k += 1
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR (1-ESP)')
    plt.ylabel('SEN')
    plt.title(f'ROC-AUC Curve for {outcome}')
    plt.legend(loc="lower right")
    plt.show()

    results_ser = dataset.convert_to_list(results)

    with open(results_file, 'w') as f: 
        json.dump(results_ser, f)                     
    print("Results saved in {} file.".format(results_file))    

    print("--- %s seconds ---" % (time.time() - start_time))
