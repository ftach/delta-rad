'''Functions to get statistics from a radiomics map. '''

import pingouin as pg 
import pandas as pd 
import os 
import numpy as np 
import matplotlib.pyplot as plt

import utils.get_map as gm

def compute_params(fractions: list, patients: list, enabled_features: list, mask_type: str = 'gtv') -> None: 
    '''Compute intensity parameters for each feature map and save in a csv file.
    
    Parameters:
    ----------
        fractions: list, list of fractions to compute feature maps for;
        patients: list, list of patients to compute feature maps for;
        enabled_features: list, list of enabled features;
        mask_type: str, type of mask to use for the feature maps; 
        
        Returns: None 
    '''
    # for each radiomics feature map, compute the intensity parameters, store them in a csv with patient ID as index 
    for feature in enabled_features: 
        for fraction in fractions: 
            # create df with patient ID as index
            stored_params_df = pd.DataFrame(index=patients, columns=['mean', 'std', 'min', 'max', 'cv', 'skewness', 'kurtosis'])
            for p in patients:
                rad_params = gm.compute_feature_map_params('Data/' + p + '/rad_maps/' + mask_type + '/' + fraction + '/' + feature + '.npy')
                if rad_params is not None:
                    stored_params_df.loc[p] = rad_params
            if not os.path.exists('Data/intensity_params/' + fraction + '/'):
                os.makedirs('Data/intensity_params/' + fraction + '/')

            # remove empty rows before saving
            stored_params_df = stored_params_df.dropna()

            stored_params_df.to_csv('Data/intensity_params/' + fraction + '/' + feature + '_params.csv')

def compare_params(outcomes: list, outcomes_df: pd.DataFrame, fractions: list, enabled_features: list) -> None:
    '''Compare intensity parameters between patients of different outcome-group with Mann-Whitney U test.

    Parameters:
    ----------
        outcomes: list, list of outcomes to compare;
        outcomes_df: pandas.DataFrame, dataframe with outcomes;
        fractions: list, list of fractions to compare;
        enabled_features: list, list of enabled features;
        
    Returns:
    '''

    outcomes_df = outcomes_df[outcomes]    # keep only columns of interest
    outcomes_df = outcomes_df.dropna()     # remove rows with NaN values
 
    for o in outcomes: 
        for ttt in fractions: 
            for f in enabled_features: 
                df = pd.read_csv('Data/intensity_params/' + ttt + '/' + f + '_params.csv', index_col=0) # load csv file 
                intensity_params = df.columns
                for i in intensity_params: 
                    x1, x2 = separate_groups(df, outcomes_df, o, i)  # separate groups of patients based on outcome and intensity parameter 
                    # normality = gs.assess_normality(x1, x2) # assess normality
                    result, pval = compare_groups(x1, x2) # compare between patients using pingouin
                    if result: 
                        print('Significant difference between groups for ' + i + ' in ' + o + ' patients for ' + f + ' in ' + ttt + ' fraction.')
                        print('Mean: {} and std: {} for group 1: '.format(np.mean(x1), np.std(x1)))
                        print('Mean: {} and std: {} for group 2: '.format(np.mean(x2), np.std(x2)))
                        print('P-value: ', pval)
                        plot_comparison(o, outcomes_df, fractions, f, i) # plot comparison between groups

def plot_comparison(outcome: str, outcomes_df: pd.DataFrame, fractions: list, feature: str, intensity_param: str) -> None:
    '''Plot intensity parameter for a given feature map.

    Parameters:
    ----------
        outcomes: list, list of outcomes to compare;
        outcomes_df: pandas.DataFrame, dataframe with outcomes;
        fractions: list, list of fractions to compare;
        feature: str, name of feature to be plot;
        intensity_param: str, name of intensity parameter to be plot;
        
    Returns:
    '''

    for ttt in fractions: 
        df = pd.read_csv('Data/intensity_params/' + ttt + '/' + feature + '_params.csv', index_col=0) # load csv file 
        x1, x2 = separate_groups(df, outcomes_df, outcome, intensity_param)  # separate groups of patients based on outcome and intensity parameter
        plt.figure()
        plt.scatter(np.zeros_like(x1), x1, label=r'${outcome} = 1$', color='blue', marker='x')
        plt.scatter(np.zeros_like(x2), x2, label=r'${outcome} = 0$', color='red', marker='x')
        plt.ylabel('Value of ' + intensity_param)
        plt.legend()
        plt.show()



def compute_delta_params(fractions: list, patients: list, enabled_features: list, mask_type: str = 'gtv') -> None:
    '''Compute statistics for each delta feature map and save in a csv file.

    Parameters:
    ----------
        fractions: list, list of the 2 fractions to compute delta feature maps for;
        patients: list, list of patients to compute feature maps for;
        enabled_features: list, list of enabled features;
        mask_type: str, type of mask to use for the feature maps;
        
    Returns:
    '''
    for feature in enabled_features: 
        stored_params_df = pd.DataFrame(index=patients, columns=['mean', 'std', 'min', 'max', 'cv', 'skewness', 'kurtosis']) # create df with patient ID as index

        for p in patients:
            if os.path.exists('Data/' + p + '/rad_maps/' + mask_type + '/delta/' + fractions[0] + '_' + fractions[1] + '/' + feature + '.npy'):
                rad_params = gm.compute_feature_map_params('Data/' + p + '/rad_maps/' + mask_type + '/delta/' + fractions[0] + '_' + fractions[1] + '/' + feature + '.npy')
            
            # print("Computed delta params for ", p)
            if rad_params is not None:
                stored_params_df.loc[p] = rad_params
        if not os.path.exists('Data/intensity_params/' + fractions[0] + '_' + fractions[1] + '/'):
            os.makedirs('Data/intensity_params/' + fractions[0] + '_' + fractions[1] + '/')

        # remove empty rows before saving
        stored_params_df = stored_params_df.dropna()

        stored_params_df.to_csv('Data/intensity_params/' + fractions[0] + '_' + fractions[1] + '/' + feature + '_params.csv') # save to csv

def separate_groups(df: pd.DataFrame, outcomes_df: pd.DataFrame, outcome: str, intensity_param: str) -> tuple:
    '''Separate data into two groups based on the outcome selected.

    Parameters
    ----------
    df: pandas.DataFrame, dataframe with data;
    outcomes_df: pandas.DataFrame, dataframe with outcomes;
    outcome: str, name of the outcome selected;
    intensity_param: str, name of the intensity parameter to compare;

    Returns
    -------
    x1: numpy.array, data for group 1;
    x2: numpy.array, data for group 2;
    '''
    index1 = outcomes_df.index[outcomes_df[outcome] == 1]
    index1 = [p.replace(' ', '') for p in index1]
    index2 = outcomes_df.index[outcomes_df[outcome] == 0]
    index2 = [p.replace(' ', '') for p in index2]

    # check that index1 are contained in df
    index1 = [i for i in index1 if i in df.index]
    index2 = [i for i in index2 if i in df.index]

    # separate data
    x1 = df.loc[index1, intensity_param].values
    x2 = df.loc[index2, intensity_param].values

    return x1, x2 

def assess_normality(x1: np.ndarray, x2: np.ndarray) -> bool:
    '''Assess normality of two sets of data for student tests.

    Parameters
    ----------
    x1: numpy.array, data for group 1;
    x2: numpy.array, data for group 2;

    Returns
    -------
    normality: bool, True if both groups are normal;
    '''
    n1 = pg.normality(x1)
    n2 = pg.normality(x2)
    if bool(n1['normal'].values) and bool(n2['normal'].values):
        return True
    else:
        return False

def compare_groups(x1: np.ndarray, x2: np.ndarray) -> tuple:
    '''Compare two groups of data using a Mann-Whitney U test.
    The Mann–Whitney U test (also called Wilcoxon rank-sum test) is a non-parametric test of the null hypothesis 
    that it is equally likely that a randomly selected value from one sample will be less than or greater than a randomly selected value from a second sample. 

    Parameters
    ----------
    x1: numpy.array, data for group 1;
    x2: numpy.array, data for group 2;

    Returns
    -------
    result: bool, True if the two groups are significantly different;
    '''
    result = pg.mwu(x1, x2, alternative='two-sided')
    p_val = float(result['p-val'].values)
    if p_val < 0.05:
        return True, p_val
    else:
        return False, p_val