'''Functions to create the dataset before feature selection. '''

import pandas as pd 
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
    # for column1, column2, corr_value in highly_correlated_pairs:
    #     print(f"{column1} and {column2} have a correlation of {corr_value:.2f}")

    return highly_correlated_pairs

def remove_highly_corr_features(highly_correlated_pairs, original_df):    
    # Remove one feature from each highly correlated pair
    to_drop = set()
    for column1, column2, _ in highly_correlated_pairs:
        if column1 not in to_drop and column2 not in to_drop:
            to_drop.add(column2)  # Keep column1, drop column2

    # Drop features
    reduced_df = original_df.drop(columns=to_drop)
    print("{} features dropped. Reduced DataFrame has {} features.".format(len(to_drop), reduced_df.shape[1]))

    return reduced_df 

def get_dataset(rad_csv_path: str, outcome_csv_path: str, selection_method: str = 'fixed', outcome: str = 'Décès', sample_features: list = ['Récidive Locale', 'Récidive Méta', 'Décès'], forbidden_patients: list = [57, 32, 56, 63], test_ratio: float = 0.3): # also 74, 82, 84, 85 are forbidden
    X = pd.read_csv(rad_csv_path, index_col=0) # header=None, names=feature_list
    X = X.dropna() # delete nan values 

    correlation_matrix = X.corr(method='pearson') 
    X = remove_highly_corr_features(get_highly_corr_features(correlation_matrix), X) #  drop features whom collinearity > 0.9 
    
    features_list = X.columns 

    outcome_df = pd.read_csv(outcome_csv_path, index_col=0)
    y = outcome_df.loc[outcome_df.index.isin(X.index)] 

    if selection_method == 'fixed': 
        y_val = get_random_test_patient(y, sample_features, forbidden_patients, test_ratio)
        y_train = y.drop(y_val.index)

        X_train = X.drop(y_val.index)
        X_val = X.loc[X.index.isin(y_val.index)] 

        assert set(y_train.index).isdisjoint(set(y_val.index)), "y_train and y_val have common indices"
        assert set(X_train.index).isdisjoint(set(X_val.index)), "X_train and X_val have common indices"
        assert len(X) == len(X_train) + len(X_val), "X array not of good size"
        assert len(y) == len(y_train) + len(y_val), "y array not of good size"

    else: 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize X_train and x_val 
    znorm_scaler = StandardScaler()
    znorm_scaled_x_train = znorm_scaler.fit_transform(X_train)
    znorm_scaled_x_val = znorm_scaler.fit_transform(X_val)

    # keep only one outcome 
    y_train = y_train.loc[:, [outcome]]
    y_val = y_val.loc[:, [outcome]]

    # reorganize y 
    y_train = y_train.values.reshape(-1, 1).ravel() # to avoid errors
    y_val = y_val.values.reshape(-1, 1).ravel() # to avoid errors
    
    return znorm_scaled_x_train, znorm_scaled_x_val, y_train, y_val, features_list

def get_random_test_patient(outcome_df: pd.DataFrame, sample_features: list = ['Récidive Locale', 'Récidive Méta', 'Décès'], forbidden_patients: list = [57, 32, 74, 82, 84, 85, 56, 63], test_ratio: float = 0.3): 
    forbidden_patients = ['Patient ' + str(x) for x in forbidden_patients]
    checked_forbidden_patients = []
    for fp in forbidden_patients: 
        if fp in outcome_df.index: 
            checked_forbidden_patients.append(fp)

    authorized_df = outcome_df.drop(checked_forbidden_patients)
    
    y_val = pd.DataFrame(columns=outcome_df.columns)
    while len(y_val) != round(len(outcome_df)*test_ratio):
        y_val = pd.DataFrame(columns=outcome_df.columns)
        for feat in sample_features: 
            outcome_selected_df = authorized_df[authorized_df[feat] == 1]
            
            # Get patient IDs already sampled
            existing_patient_ids = set(y_val.index) if not y_val.empty else set()

            to_sample = round(len(outcome_selected_df[outcome_selected_df[feat] == 1])*test_ratio) - len(y_val[y_val[feat] == 1])
            
            if to_sample > 0: 
                sampled_rows = outcome_selected_df[outcome_selected_df.index.isin(y_val.index) == False].sample(n=to_sample)
                outcome_selected_df = outcome_selected_df.drop(sampled_rows.index)
                
                # Add sampled rows to y_val only if they are not already there
                new_samples = sampled_rows[sampled_rows.index.isin(existing_patient_ids) == False]
                y_val = pd.concat([y_val, new_samples])
    return y_val