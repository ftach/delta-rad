'''Main script to run the feature selection and prediction algorithms comparison on the dataset. '''

import pandas as pd 
import json 
import os 
import time

import feature_selection as fsa 
import dataset 
import predictions as p 



def main(): 
    start_time = time.time()

    ###################### INITIALIZATION ##############################
    folder_path = '/home/tachennf/Documents/delta-rad/extracted_radiomics/'
    delta_rad_tables = ['f1_gtv.csv', 'f5_gtv.csv', 'rd_f1_f5_gtv.csv', 'rd_f1_f3_gtv.csv', 'rd_f1_f4_gtv.csv'] # X data csv names if (p != 'outcomes.csv') & (p != 'simu_gtv.csv')
    feat_sel_algo_list = ['RF', 'ADABOOST', 'ANOVA_PERC', 'ANOVA_K_BEST', 'CHI2_PERC', 'CHI2_K_BEST', 'MI_PERC', 'MI_K_BEST', 'NO_SEL', 'RDM_SEL', 'LASSO']
    outcome_csv = 'outcomes.csv'
    results_file = 'result_random_run.json'
    dset_selection_method = 'random' # 'fixed' 

    #                 , 'PCA_7', 'PCA_8', 'PCA_9'] 'NZV_01', 'NZV_01', 
    #feat_sel_algo_list = ['RDM_SEL']
    #pred_algo_list = ['LOGREGRIDGE']
    pred_algo_list = ['DT', 'RF', 'ADABOOST', 'PSVM', 'KNN', 'LOGREG', 'LOGREGRIDGE', 'BAGG', 'MLP', 'LDA', 'QDA', 'NaiveB'] # 'LSVM', 
    MAX_FEATURES = 3
    outcomes_list = ['Récidive Locale'] # 'Récidive Méta', 'Décès'
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

    ##################### COMPUTATIONS #################################
    for table in delta_rad_tables: # each radiomic table is analyzed separately
        print('Training on table ', table)
        if not table in results.keys():
            results[table] = {}
        for outcome in outcomes_list: # each outcome is analyzed separately
            print("Training for outcome ", outcome)
            X_train, X_val, y_train, y_val, features_list = dataset.get_dataset(os.path.join(folder_path, table), os.path.join(folder_path, outcome_csv), selection_method=dset_selection_method, outcome=outcome, test_ratio=0.3)
            # X_train and X_val are not normalized!! 

            ##################### FEATURE SELECTION ############################
            for feat_sel_algo in feat_sel_algo_list: # we compare different feature selection algorithms
                if 'CHI2' in feat_sel_algo:
                    X_train, X_val = dataset.min_max_scaling(X_train, X_val) # min max scaling for chi2
                elif 'NZV' in feat_sel_algo:
                    continue # no scaling for nzv
                else:     
                    X_train, X_val = dataset.znorm_scaling(X_train, X_val) # standard scaling for other algorithms 

                # elif 'PCA' not in feat_sel_algo:
                best_features, best_feat_sel_model = fsa.get_best_features(X_train, y_train, feat_sel_algo, features_list=features_list, max_features=MAX_FEATURES)

                # save feature selection model parameters 
                if best_feat_sel_model is None:
                    results[table][feat_sel_algo]['params'] = 'None'
                else:
                    try: 
                        results[table][feat_sel_algo]['params'] = best_feat_sel_model.best_params_
                    except AttributeError:
                        results[table][feat_sel_algo]['params'] = best_feat_sel_model.get_params() 

                ##################### PREDICTION ALGORITHMS ############################
                if feat_sel_algo != 'NO_SEL': # if no feature selection, we don't need to train the model for each number of features
                    for nb_features in range(1, MAX_FEATURES+1): # number of features selected
                        sel_features, X_train_filtered, X_val_filtered = fsa.filter_dataset(X_train, X_val, best_features, nb_features, features_list)


                        for pred_algo in pred_algo_list:
                            if not nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys():
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features] = {}
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['features'] = []
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params'] = []
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['mispreds'] = []

                            if X_train_filtered.shape[1] > 0:
                                best_model = p.train_model(pred_algo, X_train_filtered, y_train) # train with selected features in cross validation 

                                sens, spec, roc_auc, mispreds = p.compute_metric(X_val_filtered, y_val, best_model) # use best algo to make predictions # compute roc auc # later compare the auc for each model using a boxplot (to show the distribution of auc for different number of selected features)

                                # save results in a dict
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['features'] = sel_features 
                                if best_model is None:
                                    results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params'] = 'None'
                                else: 
                                    try: 
                                        results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params'] = best_model.best_params_
                                    except AttributeError:
                                        results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params'] = best_model.get_params()
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['roc_auc'] = roc_auc
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['specificity'] = spec
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['sensitivity'] = sens
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['mispreds'] = mispreds

                            else:
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['features'] = sel_features 
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params'] = 'None'
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['roc_auc'] = 'None'
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['specificity'] = 'None'
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['sensitivity'] = 'None'
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['mispreds'] = 'None'

                else:
                    sel_features, X_train_filtered, X_val_filtered = fsa.filter_dataset(X_train, X_val, best_features, nb_features, features_list)

                    for pred_algo in pred_algo_list:
                        if not nb_features in results[table][feat_sel_algo][pred_algo][outcome].keys():
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features] = {}
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['features'] = []
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params'] = []
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['mispreds'] = []

                        if X_train_filtered.shape[1] > 0:
                            best_model = p.train_model(pred_algo, X_train_filtered, y_train) # train with selected features in cross validation 

                            sens, spec, roc_auc, mispreds = p.compute_metric(X_val_filtered, y_val, best_model) # use best algo to make predictions # compute roc auc # later compare the auc for each model using a boxplot (to show the distribution of auc for different number of selected features)

                            # save results in a dict
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['features'] = sel_features 
                            if best_model is None:
                                results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params'] = 'None'
                            else: 
                                try: 
                                    results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params'] = best_model.best_params_
                                except AttributeError:
                                    results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params'] = best_model.get_params()
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['roc_auc'] = roc_auc
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['specificity'] = spec
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['sensitivity'] = sens
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['mispreds'] = mispreds

                        else:
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['features'] = sel_features 
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['params'] = 'None'
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['roc_auc'] = 'None'
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['specificity'] = 'None'
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['sensitivity'] = 'None'
                            results[table][feat_sel_algo][pred_algo][outcome][nb_features]['mispreds'] = 'None'
            
            # save results in a json file, at each iteration to avoid errors
            results_ser = dataset.convert_to_list(results)

            with open(results_file, 'w') as f: 
                json.dump(results_ser, f)                     
            print("Results saved in {} file.".format(results_file))    

        print("--- %s seconds ---" % (time.time() - start_time))
 

# display patient for which prediction got wrong (mispreds)
# compute p value between models for the ones we want to compare

# FORMAT TO SAVE DATA>s
# {
#   "f1_gtv.csv": {
#     "feat_sel_algo1": {
#       "params": [...] # extracted from model.best_params_
#       "pred_algo1": {
#         
#         "outcome": {
#            "nb_features": 
#           {"features": [feat1, feat2, ...], "params": [] # extracted from model.best_params_ , "roc_auc": 0.85, "specificity": 0.8, "sensitivity": 0.78},
#           {"features": 2, "params": [], "roc_auc": 0.87, "specificity": 0.82, "sensitivity": 0.8}
#         }
#       },
#       "pred_algo2": {
#         "results": [
#           {"features": 1, "roc_auc": 0.81, "specificity": 0.75, "sensitivity": 0.7}
#         ]
#       }
#     }
#   }
# }

if __name__ == '__main__': 
    main()

