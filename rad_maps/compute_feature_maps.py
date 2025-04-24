'''Script to compute radiomic feature maps.'''

import os 
import utils.get_map as gm 
import utils.get_stats as gs

def main(mask_type='ptv'): 
    params = 'params.yaml' 

    fractions = ['ttt_1', 'ttt_5'] 
    # get list of folders in Data/ if the name of the folder begins by Patient 
    folder_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/registered_data/'
    output_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/rad_maps/'
    patients_filtered = ['Patient48', 'Patient76', 'Patient75', 'Patient72', 'Patient59', 'Patient46', 'Patient34', 'Patient36', 'Patient31', 'Patient12', 'Patient20', 'Patient22', 'Patient26', 'Patient39', 'Patient40']
    
    #patients_list = [p for p in os.listdir(folder_path) if p.startswith('Patient')]
    # patients_to_remove = ['Patient32', 'Patient56', 'Patient57', 'Patient66', 'Patient14', 'Patient27', 'Patient80','Patient 85', 'Patient79', 'Patient54', 'Patient86', 'Patient20', 'Patient64', 'Patient61', 'Patient71'] # 54, 61, 64, 66, 71, 79, 86 don't have F5
    # patients_filtered = [p for p in patients_list if p not in patients_to_remove]

    enabled_features = ['original_gldm_GrayLevelNonUniformity', 'original_glrlm_RunLengthNonUniformity', 'original_glszm_ZoneEntropy', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_gldm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized']
    #enabled_features = ['all'] # ['original_firstorder_Kurtosis', 'original_gldm_DependenceEntropy'] # 'original_glcm_Imc1', # TODO/ deal size issue witg glcm_Imc1 features
    # enabled_features = [f.replace('.nrrd', '') for f in os.listdir('/home/tachennf/Documents/delta-rad/rad_maps/Data/Patient76/rad_maps/ptv/ttt_1/') if f.endswith('.nrrd')] # list of feature maps to compare
    
    # COMPUTE SIMPLE FEATURE MAPS AND ANALYZE THEIR PARAMETERS 
    computed_features = gm.compute_feature_maps(folder_path, output_path, fractions, patients_filtered, params, enabled_features, 'ptv') 
    # gs.compute_params(fractions, patients_filtered, enabled_features, mask_type='ptv') # compute intensity parameters for each feature map and save in a csv file
    print(f'Computed {len(computed_features)} feature maps for {len(patients_filtered)} patients and {len(fractions)} fractions.')

if __name__ == '__main__':
    main(mask_type='ptv')
    # features from Gladis analysis:     enabled_features = ['original_gldm_GrayLevelNonUniformity', 'original_glrlm_RunLengthNonUniformity', 'original_glszm_ZoneEntropy', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_gldm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized']

    # Récidives # all have F1 and F5!  but not PTV!!!
    # Patient 48
    # Patient 80: error registration
    # Patient 76
    # Patient 75
    # Patient 72
    # Patient 59
    # Patient 46
    # Patient 34
    # Patient 35: error registration 
    # Patient 36
    # Patient 31

    # Non récidives # all have F1 and F5!  but not PTV!!!
    # Patient 12
    # Patient 20
    # Patient 22
    # Patient 26
    # Patient 39
    # Patient 40




