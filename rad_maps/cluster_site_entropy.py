'''Script to compute the entropy of the dissimilarity map between two fractions.'''

from cluster_delta_maps import cluster_delta_map
from utils.clustering import *
import os
from utils.get_stats import *

# compute dissimilarity map between the two fractions
# compute entropy of the dissimilarity map 

def compute_dissimilarity_map(patients: list, fractions: list, rad_maps_folder: str, enabled_features: list, mask_type: str) -> None:
    for p in patients: 
        print(f'Computing dissimilarity map for {p}...')
        output_folder = rad_maps_folder + p + '/' + mask_type + '/' + fractions[0]+ '_' + fractions[1] + '/dissimilarity_map/'
        if os.path.exists(output_folder) == False:
            os.makedirs(output_folder)
        for feature in enabled_features:
            delta_map_path_1 = rad_maps_folder + p + '/' + mask_type + '/' + fractions[0] + '/' + feature + '.nrrd'
            delta_map_path_2 = rad_maps_folder + p + '/' + mask_type + '/' + fractions[1] + '/' + feature + '.nrrd'
            if os.path.exists(delta_map_path_1) == False:
                raise ValueError('Delta map not found for ' + p + ' ' + fractions[0])
            if os.path.exists(delta_map_path_2) == False:
                raise ValueError('Delta map not found for ' + p + ' ' + fractions[1])
            dissimilarity_map(delta_map_path_1, delta_map_path_2, output_folder, feature)

def compute_cluster_site_entropies(patients: list, fractions: list, rad_maps_folder: str, feature: str, mask_type: str) -> None:
    entropies1 = []
    entropies2 = []
    for patient in patients:
        cluster_img_path1 = rad_maps_folder + patient + '/' + mask_type + '/' + fractions[0] + '/clustered_map/' + feature + '.nrrd' 
        if os.path.exists(cluster_img_path1) == False:
            raise ValueError('Clustered map not found for ' + patient + ' ' + fractions[0])
        cluster_img_path2 = rad_maps_folder + patient + '/' + mask_type + '/' + fractions[1] + '/clustered_map/' + feature + '.nrrd'
        if os.path.exists(cluster_img_path2) == False:
            raise ValueError('Clustered map not found for ' + patient + ' ' + fractions[1])
        entropies = cluster_site_entropy(cluster_img_path1, cluster_img_path2)
        entropies1.append(entropies[0])
        entropies2.append(entropies[1])

    return entropies1, entropies2


def main(): 
    fractions = ['ttt_1', 'ttt_5']
    mask_type = 'ptv_5px'
    # get list of folders in Data/ if the name of the folder begins by Patient 
    folder_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/registered_data/'
    rad_maps_folder = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/rad_maps/'
    enabled_features = ['original_gldm_GrayLevelNonUniformity', 'original_glrlm_RunLengthNonUniformity', 'original_glszm_ZoneEntropy', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelNonUniformityNormalized']
    # patients_filtered = ['Patient48', 'Patient76', 'Patient75', 'Patient72', 'Patient59', 'Patient46', 'Patient34', 'Patient36', 'Patient31', 'Patient12', 'Patient20', 'Patient22', 'Patient26', 'Patient39', 'Patient40']
    recidivist = ['Patient48', 'Patient76', 'Patient75', 'Patient72', 'Patient59']
    non_recidivist = ['Patient22', 'Patient12', 'Patient20', 'Patient26', 'Patient39']
    # cluster_delta_map(patients_filtered, fractions[0], rad_maps_folder, enabled_features, mask_type, method='gmm', k=5) # Cluster radiomic map with fixed k 
    # cluster_delta_map(patients_filtered, fractions[1], rad_maps_folder, enabled_features, mask_type, method='gmm', k=5) # Cluster radiomic map with fixed k 

    # compute_dissimilarity_map(patients_filtered, fractions, rad_maps_folder, enabled_features, mask_type) # Compute dissimilarity map between the two fractions
    for feature in enabled_features:
        print(feature)
        entropies_recidivist = compute_cluster_site_entropies(recidivist, fractions, rad_maps_folder, feature, mask_type) # Compute cluster site entropy between the two fractions
        entropies_non_recidivist = compute_cluster_site_entropies(non_recidivist, fractions, rad_maps_folder, feature, mask_type) # Compute cluster site entropy between the two fractions
        # print(f'Recidivist patients: {np.mean(entropies_recidivist[0])}, {np.mean(entropies_recidivist[1])}')
        # print(f'Non-recidivist patients: {np.mean(entropies_non_recidivist[0])}, {np.mean(entropies_non_recidivist[1])}', end='\n\n')
        result, pval = compare_groups(entropies_recidivist[1], entropies_non_recidivist[1])
        if result == True: 
            print(f'Significant difference between recidivist and non-recidivist patients for {feature}.')
            print(f'Mean recidivist: {np.mean(entropies_recidivist[1])}, std: {np.std(entropies_recidivist[1])}')
            print(f'Mean non-recidivist: {np.mean(entropies_non_recidivist[1])}, std: {np.std(entropies_non_recidivist[1])}', end='\n\n')

    # compute entropies of recidivist patients 
    # compute entropies of non-recidivist patients
    # compare mean value between recidivist and non-recidivist patients
if __name__ == '__main__':
    main()