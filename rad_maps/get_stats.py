'''Functions to get statistics from a radiomics map. '''

import pingouin as pg 


def separate_groups(df, outcomes_df, outcome, intensity_param):
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

def assess_normality(x1, x2):
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

def compare_groups(x1, x2):
    '''Compare two groups of data using a Mann-Whitney U test.
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