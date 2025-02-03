'''Functions to create the dataset before feature selection. '''

import pandas as pd 
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import ConstantKernel


def get_highly_corr_features(correlation_matrix: pd.DataFrame, threshold: float = 0.9):
    """
    Identify pairs of highly correlated features from a correlation matrix.
    Args:
        correlation_matrix (pd.DataFrame): A pandas DataFrame representing the correlation matrix.
        threshold (float, optional): The correlation threshold above which features are considered highly correlated. Default is 0.9.
    Returns:
        list of tuples: A list of tuples where each tuple contains two feature names and their correlation value.
                        Example: [(feature1, feature2, correlation_value), ...]
    """

    # Find pairs of highly correlated features
    highly_correlated_pairs = [
        (column1, column2, correlation_matrix.loc[column1, column2])
        for column1 in correlation_matrix.columns
        for column2 in correlation_matrix.columns
        if (column1 != column2) and (abs(correlation_matrix.loc[column1, column2]) > threshold)
    ]

    # print("There are {} highly correlated features (correlation superior to {}): ".format(len(highly_correlated_pairs), threshold))

    # Display highly correlated pairs
    # for column1, column2, corr_value in highly_correlated_pairs:
    #     print(f"{column1} and {column2} have a correlation of {corr_value:.2f}")

    return highly_correlated_pairs

def remove_highly_corr_features(highly_correlated_pairs, original_df: pd.DataFrame):    
    """
    Remove highly correlated features from a DataFrame.
    This function takes a list of highly correlated feature pairs and removes one feature from each pair
    to reduce multicollinearity in the dataset.
    Parameters:
    highly_correlated_pairs (list of tuples): A list of tuples where each tuple contains two feature names
                                              and their correlation value (column1, column2, correlation).
    original_df (pandas.DataFrame): The original DataFrame from which highly correlated features need to be removed.
    Returns:
    pandas.DataFrame: A DataFrame with the highly correlated features removed.

    """

    # Remove one feature from each highly correlated pair
    to_drop = set()
    for column1, column2, _ in highly_correlated_pairs:
        if column1 not in to_drop and column2 not in to_drop:
            to_drop.add(column2)  # Keep column1, drop column2

    # Drop features
    reduced_df = original_df.drop(columns=to_drop)
    # print("{} features dropped. Reduced DataFrame has {} features.".format(len(to_drop), reduced_df.shape[1]))

    return reduced_df 

def get_dataset(rad_csv_path: str, outcome_csv_path: str, selection_method: str = 'fixed', outcome: str = 'Décès', sample_features: list = ['Récidive Locale', 'Récidive Méta', 'Décès'], forbidden_patients: list = [57, 32, 56, 63], test_ratio: float = 0.3): # also 74, 82, 84, 85 are forbidden
    """
    Load and preprocess dataset for training and validation.

    Parameters:
    rad_csv_path (str): Path to the radiomics CSV file.
    outcome_csv_path (str): Path to the outcome CSV file.
    selection_method (str, optional): Method to select the validation set. Default is 'fixed' (always same validation patients). Other possible values are 'random' (randomly selected patients).
    outcome (str, optional): The outcome feature to predict. Default is 'Décès'.
    sample_features (list, optional): List of sample features to consider. Default is ['Récidive Locale', 'Récidive Méta', 'Décès'].
    forbidden_patients (list, optional): List of patient IDs to exclude from the dataset. Default is [57, 32, 56, 63].
    test_ratio (float, optional): Ratio of the dataset to use for validation. Default is 0.3.

    Returns:
    tuple: A tuple containing:
        - X_train (pd.DataFrame): Training features.
        - X_val (pd.DataFrame): Validation features.
        - y_train (np.ndarray): Training labels.
        - y_val (np.ndarray): Validation labels.
        - features_list (pd.Index): List of feature names.
    """

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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y[outcome])

    # # save y_val and y_train
    y_val.to_csv('y_val.csv')
    y_train.to_csv('y_train.csv')

    # keep only one outcome 
    y_train = y_train.loc[:, [outcome]]
    y_val = y_val.loc[:, [outcome]]

    # reorganize y 
    y_train = y_train.values.reshape(-1, 1).ravel() # to avoid errors
    y_val = y_val.values.reshape(-1, 1).ravel() # to avoid errors
    
    return X_train, X_val, y_train, y_val, features_list

def get_random_test_patient(outcome_df: pd.DataFrame, sample_features: list = ['Récidive Locale', 'Récidive Méta', 'Décès'], forbidden_patients: list = [57, 32, 74, 82, 84, 85, 56, 63], test_ratio: float = 0.3): 
    """
    Get a random sample of patients for validation, keeping the same ratio of each outcome feature.
    Parameters:
    outcome_df (pd.DataFrame): DataFrame containing the outcome features.
    sample_features (list, optional): List of sample features to consider. Default is ['Récidive Locale', 'Récidive Méta', 'Décès'].
    forbidden_patients (list, optional): List of patient IDs to exclude from the dataset. Default is [57, 32, 74, 82, 84, 85, 56, 63].
    test_ratio (float, optional): Ratio of the dataset to use for validation. Default is 0.3.
    
    Returns:
    y_val (pd.DataFrame): A DataFrame containing the validation patients.
    """

    # remove patients that we can't use as validation (don't have all the RT fractions)
    forbidden_patients = ['Patient ' + str(x) for x in forbidden_patients]
    checked_forbidden_patients = []
    for fp in forbidden_patients: 
        if fp in outcome_df.index: 
            checked_forbidden_patients.append(fp)
    authorized_df = outcome_df.drop(checked_forbidden_patients)
    
    # get patients outcomes for validation 
    y_val = pd.DataFrame(columns=outcome_df.columns)
    while len(y_val) != round(len(outcome_df)*test_ratio):
        y_val = pd.DataFrame(columns=outcome_df.columns)
        for feat in sample_features: 
            outcome_selected_df = authorized_df[authorized_df[feat] == 1] # patients for which the selected outcome is 1
            
            existing_patient_ids = set(y_val.index) if not y_val.empty else set() # Get patient IDs already sampled

            to_sample = round(len(outcome_selected_df[outcome_selected_df[feat] == 1])*test_ratio) - len(y_val[y_val[feat] == 1]) # nbr of patients to sample for the selected outcome
            
            # Sample patients for the selected outcome
            if to_sample > 0: 
                sampled_rows = outcome_selected_df[outcome_selected_df.index.isin(y_val.index) == False].sample(n=to_sample)
                outcome_selected_df = outcome_selected_df.drop(sampled_rows.index) # remove patients already sampled
                
                # Add sampled rows to y_val only if they are not already there
                new_samples = sampled_rows[sampled_rows.index.isin(existing_patient_ids) == False]
                y_val = pd.concat([y_val, new_samples])

    return y_val

def min_max_scaling(X_train, X_val): 
    """
    Scales the training and validation data to 0-1 range.

    Parameters:
    X_train (array-like): Training data to be scaled.
    X_val (array-like): Validation data to be scaled.

    Returns:
    tuple: A tuple containing the scaled training data and scaled validation data.
    """

    min_max_scaler = MinMaxScaler()
    min_max_scaled_x_train = min_max_scaler.fit_transform(X_train)
    min_max_scaled_x_val = min_max_scaler.fit_transform(X_val)
    return min_max_scaled_x_train, min_max_scaled_x_val

def znorm_scaling(X_train, X_val):
    """
    Scales the training and validation data to have zero mean and unit variance.
    
    Parameters:
    X_train (array-like): Training data to be scaled.
    X_val (array-like): Validation data to be scaled.

    Returns:
    tuple: A tuple containing the scaled training data and scaled validation data.
    """
    znorm_scaler = StandardScaler()
    znorm_scaled_x_train = znorm_scaler.fit_transform(X_train)
    znorm_scaled_x_val = znorm_scaler.fit_transform(X_val)

    return znorm_scaled_x_train, znorm_scaled_x_val

def convert_to_list(obj):
    """
    Convert various types of objects to a list or a more serializable format.

    Parameters:
    obj (any): The object to be converted. This can be a callable, numpy array, dictionary, or pandas Index.

    Returns:
    any: The converted object. If the input is a callable, its name is returned as a string. 
         If the input is a numpy array or pandas Index, it is converted to a list. 
         If the input is a dictionary, its values are recursively converted. 
         Otherwise, the original object is returned.
    """
    try:
        if callable(obj):
            return obj.__name__
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_list(v) for k,v in obj.items()}
        elif isinstance(obj, pd.Index):
            return obj.tolist()
        if isinstance(obj, ConstantKernel):
            return {"type": "ConstantKernel", "constant_value": obj.constant_value}
        else:
            return obj
    except Exception as e:
        print(obj)
        print(e)

