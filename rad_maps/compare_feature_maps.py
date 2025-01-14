'''Script to compare feature map parameters between patients.'''

import os 
import numpy as np
import get_map as gm 
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
                try: 
                    rad_params = gm.compute_feature_map_params('Data/' + p + '/rad_maps/' + fraction + '/' + feature + '.nrrd')
                    stored_params_df.loc[p] = rad_params
                except RuntimeError:
                    print('Feature map not found for ' + p)
                    continue
            stored_params_df.to_csv('Data/' + feature + '_params.csv')

def main(): 
    # COMPUTE FEATURE MAPS
    params = 'params.yaml' 

    fractions = ['ttt_1', 'ttt_3']

    patients = os.listdir('Data/')
    patients_to_remove = ['Patient' + str(n) for n in [57, 32, 74, 82, 84, 85, 56, 63]]
    patients_filtered = [p for p in patients if patients not in patients_to_remove]
    enabled_features = ['original_firstorder_Skewness', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum'] 
    
    # compute_feature_maps(fractions, patients_filtered, params, enabled_features)

    # compute_params(fractions, patients_filtered, enabled_features)

    # TODO: compare statistics between patients using scipy or pengouin 

if __name__ == '__main__':
    main()