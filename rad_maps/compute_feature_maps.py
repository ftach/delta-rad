'''Script to compute radiomic feature maps.'''

import os 
import utils.get_map as gm 

def main(): 
    params = 'params.yaml' 

    fractions = ['ttt_1', 'ttt_5'] 
    # get list of folders in Data/ if the name of the folder begins by Patient 
    folder_path = '/home/tachennf/Documents/delta-rad/rad_maps/Data/'
    patients_list = [p for p in os.listdir(folder_path) if p.startswith('Patient')]

    patients_to_remove = ['Patient32', 'Patient56', 'Patient57', 'Patient66', 'Patient14', 'Patient27', 'Patient80','Patient 85', 'Patient79', 'Patient54', 'Patient86', 'Patient20', 'Patient64', 'Patient61', 'Patient71'] # 54, 61, 64, 66, 71, 79, 86 don't have F5
    patients_filtered = [p for p in patients_list if p not in patients_to_remove]

    # enabled_features = ['original_firstorder_Entropy', 'original_gldm_DependenceEntropy', 'original_glrlm_GrayLevelNonUniformity']
    enabled_features = ['original_firstorder_Kurtosis', 'original_gldm_DependenceEntropy'] # 'original_glcm_Imc1', # TODO/ deal size issue witg glcm_Imc1 features

    # COMPUTE SIMPLE FEATURE MAPS AND ANALYZE THEIR PARAMETERS 
    gm.compute_feature_maps(fractions, patients_filtered, params, enabled_features) 

if __name__ == '__main__':
    main()