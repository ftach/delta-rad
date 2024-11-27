import pandas as pd 
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_highly_corr_features(correlation_matrix, threshold=0.9):
    # Find pairs of highly correlated features
    highly_correlated_pairs = [
        (column1, column2, correlation_matrix.loc[column1, column2])
        for column1 in correlation_matrix.columns
        for column2 in correlation_matrix.columns
        if (column1 != column2) and (abs(correlation_matrix.loc[column1, column2]) > threshold)
    ]

    print("There are {} highly correlated features (correlation superior to {}): ".format(len(highly_correlated_pairs), threshold))

    # Display highly correlated pairs
    for column1, column2, corr_value in highly_correlated_pairs:
        print(f"{column1} and {column2} have a correlation of {corr_value:.2f}")

    return highly_correlated_pairs

def remove_highly_corr_features(highly_correlated_pairs, original_df):    
    # Remove one feature from each highly correlated pair
    to_drop = set()
    for column1, column2, _ in highly_correlated_pairs:
        if column1 not in to_drop and column2 not in to_drop:
            to_drop.add(column2)  # Keep column1, drop column2

    print("{} features dropped. ".format(len(to_drop)))

    # Drop features
    reduced_df = original_df.drop(columns=to_drop)
    print(f"Reduced DataFrame has {reduced_df.shape[1]} features.")

    return reduced_df 

def get_dataset(rad_csv_path: str, outcome_csv_path: str, y_val: pd.DataFrame, selection: str = 'random'): 
    # TODO: get dataset from y_val
    
    rad_df = pd.read_csv(rad_csv_path, index_col=0) # header=None, names=feature_list
    rad_df = rad_df.dropna() # delete nan values 
    correlation_matrix = rad_df.corr(method='pearson') 

    X = remove_highly_corr_features(get_highly_corr_features(correlation_matrix), rad_df) #  drop features whom collinearity > 0.9 
    
    outcome_df = pd.read_csv(outcome_csv_path, index_col=0)
    y = outcome_df.loc[outcome_df.index.isin(X.index)] 
    
    # obtain X_train, X_val, y_train, y_val 
    if selection == 'random': 
        y_train, y_val = get_random_test_patient(y) # if we want to have the same x_val and y_val dataset for the whole study!! 
        # TODO: get X_train, y_train 
    else: 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize X_train and x_val 
    znorm_scaler = StandardScaler()
    znorm_scaled_x_train = znorm_scaler.fit_transform(X_train)
    znorm_scaled_x_val = znorm_scaler.fit_transform(X_val)
    
    return znorm_scaled_x_train, znorm_scaled_x_val, y_train, y_val 

def get_random_test_patient(outcome_df: pd.DataFrame, sample_features: list = ['Récidive Locale', 'Récidive Méta', 'Décès'], forbidden_patients: list = [57, 32, 74, 82, 84, 85, 56, 63], test_ratio: float = 0.3): 
    forbidden_patients = ['Patient ' + str(x) for x in forbidden_patients]
    authorized_df = outcome_df.drop(forbidden_patients)

    y_val = pd.DataFrame(columns=outcome_df.columns)
    for feat in sample_features: 
        outcome_selected_df = authorized_df[authorized_df[feat] == 1] # patients in outcome_df for which value at feature 'feat' == 1
        to_sample = round(len(outcome_selected_df[outcome_selected_df[feat] == 1])*test_ratio) - len(y_val[y_val[feat] == 1]) # number of patients in outcome_df for which value at feature 'feat' == 1 - number of patients in filtered_df for which value at feature 'feat' == 1
        print(to_sample)
        sampled_rows = outcome_selected_df.sample(n=to_sample, random_state=42)
        y_val = pd.concat([y_val, sampled_rows]) # append sampled rows to filtered df 

    print(len(outcome_df))
    # Drop filtered_df from outcome_df
    y_train = outcome_df.drop(y_val.index)
    print(y_val.duplicated().sum())  # Should return 0

    print(len(y_train), len(y_val), len(outcome_df))

    # Validate size consistency
    assert len(y_train) + len(y_val) == len(outcome_df), "Mismatch in DataFrame sizes"

    return y_train, y_val  

def get_best_features(X_train: pd.DataFrame, feat_sel_algo: str): 
    pass 

def filter_dataset(X_train: pd.DataFrame, X_val: pd.DataFrame, best_features: list, nb_features: int): 
    # selected_features = best_features[:nb_features+1] # select only i features from best_features 
    # X_train_filtered = znorm_scaled_x_train[selected_features.keys()] 
    # X_val_filtered = znorm_scaled_x_val[selected_features.keys()]
    # return sel_features, X_train_filtered, X_val_filtered 
    pass 

def train_model(pred_algo: str, X_train_filtered: pd.DataFrame, y_train: pd.DataFrame): 
    # if pred_algo == RF: 
        # train RF 
    pass 

def compute_metric(X_val: pd.DataFrame, y_val: pd.DataFrame):
    pass 
    # return sens, spec, roc_auc, mispreds

def compute_pvalue(binary_preds1, binary_preds2, y_test): 
    # the algorithm needs to be trained again!! find the params and features in the json file. That avoids to save ALL the outcomes 
    # https://www.statology.org/mcnemars-test-python/ 

    # Create a contingency table
    contingency_table = [[0, 0], [0, 0]]
    for true, p1, p2 in zip(y_test, binary_preds1, binary_preds2):
        if p1 == true and p2 != true:
            contingency_table[0][1] += 1
        elif p1 != true and p2 == true:
            contingency_table[1][0] += 1

    # McNemar's test
    result = mcnemar(contingency_table, exact=True)

    return result.pvalue 

def main(): 
    pass
    # TODO: Get y_val first!!! it should not move if we want to compare
    # for each table in delta_rad_tables
        # TODO: X_train, X_val, y_train, y_val = get_dataset(rad_csv_path, outcome_csv_path, y_val)
        # for feat_sel_algo in feat_sel_algo_list:
            # best_features = get_best_features(X_train, feat_sel_algo)
            # for nb_features in range(1, max(best_features)+1) # number of features selected
                # sel_features, X_train_filtered, X_val_filtered = filter_dataset(X_train, X_val, best_features, nb_features)
                # for pred_algo in pred_algo_list:
                    # best_model = train_model(pred_algo, X_train_filtered, y_train) # train with selected features in cross validation 
                    # sens, spec, roc_auc, mispreds = compute_metric(X_val, y_val) # use best algo to make predictions # compute roc auc # later compare the auc for each model using a boxplot (to show the distribution of auc for different number of selected features)
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
