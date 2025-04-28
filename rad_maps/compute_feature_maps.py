'''Script to compute radiomic feature maps.'''

import os 
import utils.get_map as gm 
import utils.get_stats as gs

def compute_feature_maps(patients: list, fractions: list, input_path: str, output_path: str, params_path: str, enabled_features: list, mask_type: str) -> None:
    '''Compute the feature maps for the given patients and fractions.
    
    Parameters:
    patients: list of str, list of patients to compute the feature maps for.
    fractions: list of str, list of fractions to compute the feature maps for.
    input_path: str, path to the input data.
    output_path: str, path to the output data.
    params_path: str, path to the parameters file.
    enabled_features: list of str, list of features to compute.
    mask_type: str, type of mask to use. Options are 'ptv' and 'gtv'.

    Returns:
    None
    '''

    # Compute features maps if they were not computed yet
    for p in patients: 
        for f in fractions:
            rad_map_path = output_path + p + '/' + mask_type + '/' + f + '/'
            if os.path.exists(rad_map_path) == False: # if maps were not computed yet
                os.makedirs(rad_map_path)
                image_path = input_path + p + '/img_dir/' + p + '_mridian_' + f + '.nii'
                if os.path.exists(image_path) == False: # if fraction is missing 
                    raise ValueError('Image not found for ' + p + ' ' + f) 
                mask_path = input_path + p + '/mask_dir/' + p + '_mridian_' + f + '_' + mask_type + '.nii'
                if os.path.exists(mask_path) == False: # mask is missing 
                    raise ValueError('Mask not found for ' + p + ' ' + f)
                computed_features = gm.generate_feature_map(image_path, mask_path, params_path, rad_map_path, enabled_features)
                print(f'Computed {len(computed_features)} feature maps for {p}.')
            
            else: 
                continue

def main(mask_type='ptv'): 
    params = 'params.yaml' 

    fractions = ['ttt_1', 'ttt_5'] 
    folder_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/registered_data/'
    output_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/rad_maps/'
    patients_filtered = ['Patient48', 'Patient76', 'Patient75', 'Patient72', 'Patient59', 'Patient46', 'Patient34', 'Patient36', 'Patient31', 'Patient12', 'Patient20', 'Patient22', 'Patient26', 'Patient39', 'Patient40']
    enabled_features = ['original_gldm_GrayLevelNonUniformity', 'original_glrlm_RunLengthNonUniformity', 'original_glszm_ZoneEntropy', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_gldm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized']
  
    compute_feature_maps(patients_filtered, fractions, folder_path, output_path, params, enabled_features, mask_type) 
    # gs.compute_params(fractions, patients_filtered, enabled_features, mask_type='ptv') # compute intensity parameters for each feature map and save in a csv file
    print(f'Computed feature maps for {len(patients_filtered)} patients and {len(fractions)} fractions.')

if __name__ == '__main__':
    main(mask_type='ptv_5px')
    # features from Gladis analysis:     enabled_features = ['original_gldm_GrayLevelNonUniformity', 'original_glrlm_RunLengthNonUniformity', 'original_glszm_ZoneEntropy', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_gldm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized']

    # Récidives # all have F1 and F5!  but not PTV!!!
    # Patient 48
    # Patient 76
    # Patient 75
    # Patient 72
    # Patient 59
    # Patient 46
    # Patient 34
    # Patient 36
    # Patient 31
    # Patient 35: error registration 
    # Patient 80: error registration

    # Non récidives # all have F1 and F5!  but not PTV!!!
    # Patient 12
    # Patient 20
    # Patient 22
    # Patient 26
    # Patient 39
    # Patient 40




