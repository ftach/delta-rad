import pandas as pd 

import os 
import feature_selection as fsa 
import dataset 
import predictions as p 



def main(): 
    ###################### INITIALIZATION ##############################
    folder_path = '/mnt/c/Users/tachenne/delta-rad/extracted_radiomics'
    delta_rad_tables = [p for p in os.listdir(folder_path) if (p != 'outcomes.csv') & (p != 'simu_gtv.csv')] # X data csv names 

    #feat_sel_algo_list = ['RF', 'ADABOOST', 'NZV_01', 'NZV_01', 'ANOVA_PERC', 'ANOVA_K_BEST', 'CHI2_PERC', 'CHI2_K_BEST', 'MI_PERC', 'MI_K_BEST', 'NO_SEL', 'RDM_SEL',
    #                'LASSO', 'PCA_7', 'PCA_8', 'PCA_9']
    feat_sel_algo_list = ['RF']
    pred_algo_list = ['RF']
    MAX_FEATURES = 5
    outcomes_list = ['Récidive Locale', 'Récidive Méta', 'Décès']
    ##################### COMPUTATIONS #################################
    for table in delta_rad_tables:
        for outcome in outcomes_list: 
            X_train, X_val, y_train, y_val, features_list = dataset.get_dataset(os.path.join(folder_path, table), os.path.join(folder_path, 'outcomes.csv'), selection_method='fixed', outcome=outcome, test_ratio=0.3)
            # X_train and X_val are already normalized
            for feat_sel_algo in feat_sel_algo_list:
                if 'PCA' not in feat_sel_algo:
                    best_features = fsa.get_best_features(X_train, y_train, feat_sel_algo, features_list=features_list, max_features=MAX_FEATURES)
                    print(best_features, sep='\n')
                    for nb_features in range(1, len(best_features)+1): # number of features selected
                        sel_features, X_train_filtered, X_val_filtered = fsa.filter_dataset(X_train, X_val, best_features, nb_features, features_list)
                        for pred_algo in pred_algo_list:
                            best_model = p.train_model(pred_algo, X_train_filtered, y_train) # train with selected features in cross validation 
                            sens, spec, roc_auc, mispreds = p.compute_metric(X_val, y_val) # use best algo to make predictions # compute roc auc # later compare the auc for each model using a boxplot (to show the distribution of auc for different number of selected features)

                else: 
                    break 
                    # sel_features, X_train_filtered, X_val_filtered = filter_dataset(X_train, X_val, best_features, nb_features)
                    # for pred_algo in pred_algo_list:
                        # best_model = train_model(pred_algo, X_train_filtered, y_train) # train with selected features in cross validation 
                        # sens, spec, roc_auc, mispreds = compute_metric(X_val, y_val) # use best algo to make predictions # compute roc auc # later compare the auc for each model using a boxplot (to show the distribution of auc for different number of selected features)
    return best_features

# make plots 
# display patient for which prediction got wrong (mispreds)
# compute p value between models for the ones we want to compare

# FORMAT TO SAVE DATA>s
# {
#   "f1_gtv.csv": {
#     "feat_sel_algo1": {
#       "params": [...] # extracted from model.best_params_
#       "pred_algo1": {
#         
#         "results": [
#           {"features": [feat1, feat2, ...], "params": [] # extracted from model.best_params_ , "roc_auc": 0.85, "specificity": 0.8, "sensitivity": 0.78},
#           {"features": 2, "params": [], "roc_auc": 0.87, "specificity": 0.82, "sensitivity": 0.8}
#         ]
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
