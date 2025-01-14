'''Script to compare feature map parameters between patients.'''

import os 
import numpy as np
import get_map as gm 
import get_stats as gs
import pandas as pd 

def compute_feature_maps(fractions, patients, params_path, enabled_features):
    '''Compute feature maps for all given patients and fractions.
    Args:
        fractions: list, list of fractions to compute feature maps for;
        patients: list, list of patients to compute feature maps for;
        params_path: str, path to the parameters file;
        enabled_features: list, list of enabled features;
        
    Returns:    
    '''
    
    errors = []
    for f in fractions:
        for p in patients:
            image_path = 'Data/' + p + '/img_dir/' + p + '_mridian_' + f + '.nii'
            mask_path = 'Data/' + p + '/mask_dir/' + p + '_mridian_' + f + '_gtv.nii' 
            try: 
                gm.generate_feature_map(image_path, mask_path, params_path, 'Data/' + p + '/rad_maps/' + f + '/', enabled_features)
                assert os.path.exists('Data/' + p + '/rad_maps/' + f + '/'), 'Feature map not created'
            except ValueError: 
                print('Error in ' + p + ' ' + f)
                errors.append(p + ' ' + f)
                continue
    print(errors)


def compute_params(fractions, patients, enabled_features): 
    '''Compute statistics for each feature map and save in a csv file.
    Args:
        fractions: list, list of fractions to compute feature maps for;
        patients: list, list of patients to compute feature maps for;
        enabled_features: list, list of enabled features;
        
        Returns:
    '''
    # for each radiomics feature map, compute the intensity parameters, store them in a csv with patient ID as index 
    for feature in enabled_features: 
        for fraction in fractions: 
            # create df with patient ID as index
            stored_params_df = pd.DataFrame(index=patients, columns=['mean', 'std', 'min', 'max', 'cv', 'skewness', 'kurtosis'])
            for p in patients:
                rad_params = gm.compute_feature_map_params('Data/' + p + '/rad_maps/' + fraction + '/' + feature + '.nrrd')
                if rad_params is not None:
                    stored_params_df.loc[p] = rad_params
            if not os.path.exists('Data/intensity_params/' + fraction + '/'):
                os.makedirs('Data/intensity_params/' + fraction + '/')

            # remove empty rows before saving
            stored_params_df = stored_params_df.dropna()

            stored_params_df.to_csv('Data/intensity_params/' + fraction + '/' + feature + '_params.csv')

def main(): 
    # COMPUTE FEATURE MAPS
    params = 'params.yaml' 

    fractions = ['ttt_3'] # , 'ttt_1'
    # get list of folders in Data/ if the name of the folder begins by Patient 
    patients = os.listdir('Data/')
    patients = [p for p in patients if p.startswith('Patient')]
    patients_to_remove = ['Patient' + str(n) for n in [57, 32, 74, 82, 84, 85, 56, 63]]
    patients_filtered = [p for p in patients if patients not in patients_to_remove]
    enabled_features = [ 'original_firstorder_Skewness' ] # 'original_firstorder_Skewness', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum'

    # compute_feature_maps(fractions, patients_filtered, params, enabled_features)

    # compute_params(fractions, patients_filtered, enabled_features)

    # TODO: compare statistics between patients using scipy or pengouin 
    # load outcome table 
    outcomes = ['Décès' ] # 'Récidive Locale'
    outcomes_df = pd.read_csv('/home/tachennf/Documents/delta-rad/extracted_radiomics/outcomes.csv', index_col=0)
    outcomes_df = outcomes_df[outcomes]    # keep only columns of interest
    outcomes_df = outcomes_df.dropna()     # remove rows with NaN values
 
    for o in outcomes: 
        for ttt in fractions: 
            for f in enabled_features: 
                df = pd.read_csv('Data/intensity_params/' + ttt + '/' + f + '_params.csv', index_col=0) # load csv file 
                intensity_params = df.columns
                for i in intensity_params: 
                    x1, x2 = gs.separate_groups(df, outcomes_df, o, i)  # separate groups of patients based on outcome and intensity parameter 
                    # normality = gs.assess_normality(x1, x2) # assess normality
                    result, pval = gs.compare_groups(x1, x2) # compare between patients using pingouin
                    if result: 
                        print('Significant difference between groups for ' + i + ' in ' + o + ' patients for ' + f + ' in ' + ttt + ' fraction. P-value: ', pval)


if __name__ == '__main__':
    main()