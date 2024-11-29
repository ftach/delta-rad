'''Functions to select features. '''
from sklearn.ensemble import RandomForestClassifier
import sklearn_utils as sku 
import pandas as pd 
import numpy as np 
import predictions as p 

def select_best_features(importances, feat_names, n_features):
    # Combine importances and feature names into a list of tuples
    feature_importance = list(zip(feat_names, importances))
    
    # Sort the list of tuples based on importances in descending order
    sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    # Select top n_features from both input and sorted lists
    top_features = {k: v for k, v in sorted_features[:n_features] if v > 0}

    return top_features

def rf_feat_sel(znorm_scaled_x_train, y_train, features_list, max_features: int = 5): 
    # TODO: check that znorm_scaled_x_train, y_train are df !! 
    print("Beginning feature selection with Random Forest...")

    best_model = p.train_rf(znorm_scaled_x_train, y_train)

    selected_features = select_best_features(best_model.feature_importances_, features_list, n_features=max_features)
    
    print("Feature selection with Random Forest ended.")

    return selected_features 


# TODO: Add other algorithms 



def get_best_features(X_train: np.ndarray, y_train: np.ndarray, feat_sel_algo: str, features_list: list, max_features: int = 5): 
    if feat_sel_algo == 'RF': 
        best_features = rf_feat_sel(X_train, y_train, features_list, max_features)
    # TODO: Add other algorithms 
    return best_features


def filter_dataset(X_train: np.ndarray, X_val: np.ndarray, best_features_dict: dict, nb_features: int, feature_names: list): 
    best_features = list(best_features_dict.keys())
    selected_features = best_features[:nb_features+1] # select only i best features 

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_val = pd.DataFrame(X_val, columns=feature_names)

    # Use indices to filter the arrays
    X_train_filtered = X_train[selected_features]
    X_val_filtered = X_val[selected_features]

    # transform back X_train_filtered to np.ndarray
    X_train_filtered = X_train_filtered.to_numpy()
    X_val_filtered = X_val_filtered.to_numpy()


    return selected_features, X_train_filtered, X_val_filtered
