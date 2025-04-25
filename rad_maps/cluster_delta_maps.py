'''Script to cluster the delta maps previously computed.'''

import os 
from utils.clustering import gen_clustered_map

def cluster_delta_map(patients: list, delta_fraction: str, original_data_folder: str, rad_maps_folder: str, enabled_features: list, mask_type: str) -> None:
    # load delta map 
    for p in patients: 
        print(f'Computing delta feature maps for {p}...')
        output_folder = rad_maps_folder + p + '/' + mask_type + '/' + delta_fraction + '/clustered_map/'
        # if os.path.exists(output_folder) == False:
        #     os.mkdir(output_folder)
        # if len(os.listdir(output_folder)) == 0: # if maps were not computed yet
        for feature in enabled_features:
            delta_map_path = rad_maps_folder + p + '/' + mask_type + '/' + delta_fraction + '/' + feature + '.nrrd'
            if os.path.exists(delta_map_path) == False: # if map is missing
                raise ValueError('Delta map not found for ' + p + ' ' + delta_fraction)
            mask_path = original_data_folder + p + '/mask_dir/' + p + '_mridian_' + delta_fraction[:5] + '_' + mask_type + '.nii'
            if os.path.exists(mask_path) == False: # mask is missing
                raise ValueError('Mask not found for ' + p + ' ' + delta_fraction.split('_')[0])
            gen_clustered_map(delta_map_path, mask_path, output_folder, feature, k=4, method='otsu') # generate clustered map
        # else: 
        #     print(f'Clustered maps for {p} and {delta_fraction} already computed.')

def main():
    delta_fraction = 'ttt_1_ttt_5'
    mask_type = 'ptv_5px'
    # get list of folders in Data/ if the name of the folder begins by Patient 
    folder_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/registered_data/'
    rad_maps_folder = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/rad_maps/'
    enabled_features = ['original_gldm_GrayLevelNonUniformity', 'original_glrlm_RunLengthNonUniformity', 'original_glszm_ZoneEntropy', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelNonUniformityNormalized']
    patients_filtered = ['Patient48', 'Patient76', 'Patient75', 'Patient72', 'Patient59', 'Patient46', 'Patient34', 'Patient36', 'Patient31', 'Patient12', 'Patient20', 'Patient22', 'Patient26', 'Patient39', 'Patient40']
    
    cluster_delta_map(patients_filtered, delta_fraction, folder_path, rad_maps_folder, enabled_features, mask_type)
if __name__ == '__main__':
    main()