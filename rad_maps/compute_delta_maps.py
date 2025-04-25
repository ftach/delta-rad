'''Script to compute the delta maps for the features of the patients in the Data/ folder. Also applies clustering and analyze heterogeneity of the features.'''

import os 

import sys
sys.path.append('/home/tachennf/Documents/delta-rad/rad_maps/utils')

import utils.get_map as gm 
import utils.get_stats as gs

def compute_feature_maps_same_mask(patients: list, fractions: list, input_path: str, output_path: str, params_path: str, enabled_features: list, mask_type: str) -> None:
    '''Compute the feature maps for the given patients and fractions using the same mask for all the fractions (the first).
    
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
        mask_path = input_path + p + '/mask_dir/' + p + '_mridian_' + fractions[0] + '_' + mask_type + '.nii'
        if os.path.exists(mask_path) == False: # mask is missing 
            raise ValueError('Mask not found for ' + p + ' ' + f)
        for f in fractions:
            rad_map_path = output_path + p + '/' + mask_type + '/' + f + '/'
            if os.path.exists(rad_map_path) == False: # if maps were not computed yet
                os.makedirs(rad_map_path)
                image_path = input_path + p + '/img_dir/' + p + '_mridian_' + f + '.nii'
                if os.path.exists(image_path) == False: # if fraction is missing 
                    raise ValueError('Image not found for ' + p + ' ' + f) 

                computed_features = gm.generate_feature_map(image_path, mask_path, params_path, rad_map_path, enabled_features)
                print(f'Computed {len(computed_features)} feature maps for {p}.')
            
            else: 
                continue

def compute_delta_map(patients: list, fractions: list, original_data_folder: str, rad_maps_folder: str, enabled_features: list, mask_type: str) -> None:
    '''Compute the feature maps for the given patients and fractions.
    
    Parameters:
    patients: list of str, list of patients to compute the feature maps for.
    fractions: list, fractions to compute the delta feature maps for.
    rad_maps_folder: str, path to the input data.
    output_path: str, path to the output data.
    params_path: str, path to the parameters file.
    enabled_features: list of str, list of features to compute.
    mask_type: str, type of mask to use. Options are 'ptv' and 'gtv'.

    Returns:
    None
    '''
    delta_fraction = fractions[0] + '_' + fractions[1] # delta fraction name
    # Compute delta features maps if they were not computed yet
    for p in patients: 
        print(f'Computing delta feature maps for {p}...')
        output_folder = rad_maps_folder + p + '/' + mask_type + '/' + delta_fraction + '/'
        if os.path.exists(output_folder) == False: # if maps were not computed yet
            os.mkdir(output_folder)
            for feature in enabled_features:

                map1_path = rad_maps_folder + p + '/' + mask_type + '/' + fractions[0] + '/' + feature + '.nrrd'
                if os.path.exists(map1_path) == False: # if fraction is missing 
                    raise ValueError('Image not found for ' + p + ' ' + fractions[0]) 
                
                map2_path = rad_maps_folder + p + '/' + mask_type + '/' + fractions[1] + '/' + feature + '.nrrd'
                if os.path.exists(map2_path) == False: # mask is missing 
                    raise ValueError('Image not found for ' + p + ' ' + fractions[1])
                
                mask_path = original_data_folder + p + '/mask_dir/' + p + '_mridian_' + fractions[0] + '_' + mask_type + '.nii'
                if os.path.exists(mask_path) == False: # mask is missing
                    raise ValueError('Mask not found for ' + p + ' ' + fractions[0])
                
                gm.generate_delta_map([map1_path, map2_path], output_folder, feature)

            # print(f'Computed delta feature maps for {p}.')
        else: 
            continue

def main(mask_type: str = 'ptv') -> None:    

    params_path = 'params.yaml' 
    fractions = ['ttt_1', 'ttt_5'] 

    # get list of folders in Data/ if the name of the folder begins by Patient 
    folder_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/registered_data/'
    rad_maps_folder = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/rad_maps/'
    enabled_features = ['original_gldm_GrayLevelNonUniformity', 'original_glrlm_RunLengthNonUniformity', 'original_glszm_ZoneEntropy', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_gldm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized']
    patients_filtered = ['Patient48', 'Patient76', 'Patient75', 'Patient72', 'Patient59', 'Patient46', 'Patient34', 'Patient36', 'Patient31', 'Patient12', 'Patient20', 'Patient22', 'Patient26', 'Patient39', 'Patient40']
    # 59, 36, 12, 36, 40 have different feature map shapes 

    # Compute features maps if they were not computed yet
    compute_feature_maps_same_mask(patients_filtered, fractions, folder_path, rad_maps_folder, params_path, enabled_features, mask_type)

    compute_delta_map(patients_filtered, fractions, folder_path, rad_maps_folder, enabled_features, mask_type) 

if __name__ == '__main__':
    main(mask_type='ptv_5px')