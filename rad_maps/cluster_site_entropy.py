'''Script to compute the entropy of the dissimilarity map between two fractions.'''

from cluster_delta_maps import cluster_delta_map
from utils.clustering import *
import os
import re 
from utils.get_stats import *
from compute_delta_maps import compute_feature_maps_same_mask, compute_delta_map
from preprocessing.create_masks import create_dilated_ptv_mask
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
            print(f'Clustered map not found for {patient} {fractions[0]}')
            continue
        cluster_img_path2 = rad_maps_folder + patient + '/' + mask_type + '/' + fractions[1] + '/clustered_map/' + feature + '.nrrd'
        if os.path.exists(cluster_img_path2) == False:
            print(f'Clustered map not found for {patient} {fractions[1]}')
            continue
        entropies = cluster_site_entropy(cluster_img_path1, cluster_img_path2)
        entropies1.append(entropies[0])
        entropies2.append(entropies[1])

    return entropies1, entropies2

def main(): 
    params_path = 'params.yaml'
    fractions = ['ttt_1', 'ttt_5']
    mask_type = 'ptv_5px'
    # get list of folders in Data/ if the name of the folder begins by Patient 
    folder_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/registered_data/'
    rad_maps_folder = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/rad_maps/'
    enabled_features = ['original_gldm_GrayLevelNonUniformity', 'original_glrlm_RunLengthNonUniformity', 'original_glszm_ZoneEntropy', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelNonUniformityNormalized']
    
    outcome_df = pd.read_csv('/home/tachennf/Documents/delta-rad/data/ICM_0.35T/extracted_radiomics/nine_months_metastatic_status.csv', index_col=0)
    recidivist_patients = outcome_df[outcome_df['nine_months_metastatic_status'] == 1].index.tolist() #  one_year_local_control_status
    recidivist = [x.replace(" ", "") for x in recidivist_patients]
    non_recidivist_patients = outcome_df[outcome_df['nine_months_metastatic_status'] == 0].index.tolist() # one_year_local_control_status
    non_recidivist = [x.replace(" ", "") for x in non_recidivist_patients]


    error_patients = ['Patient11', 'Patient18', 'Patient85', 'Patient79', 'Patient54', 'Patient86', 'Patient66', 'Patient32', 'Patient64', 'Patient61', 'Patient80'] 
    recidivist = [x for x in recidivist if x not in error_patients]
    non_recidivist = [x for x in non_recidivist if x not in error_patients]

    # METASTATIC STATUS
    recidivist = ['Patient81', 'Patient76', 'Patient01', 'Patient29', 'Patient10', 'Patient04', 'Patient08', 'Patient51', 'Patient06', 'Patient09', 'Patient15', 'Patient21', 'Patient49', 'Patient25', 
     'Patient19', 'Patient23', 'Patient24', 'Patient28', 'Patient33', 'Patient37', 'Patient41', 'Patient42', 'Patient58', 'Patient46', 'Patient63', 'Patient56', 'Patient59', 'Patient72', 
     'Patient78', 'Patient70', 'Patient65', 'Patient67', 'Patient68', 'Patient74', 'Patient84', 'Patient50', 'Patient47']
    non_recidivist = ['Patient73', 'Patient16', 'Patient82', 'Patient07', 'Patient03', 'Patient05', 'Patient02', 'Patient12', 'Patient14', 'Patient13', 'Patient17', 'Patient20', 'Patient22', 
                    'Patient26', 'Patient31', 'Patient38', 'Patient40', 'Patient30', 'Patient36', 'Patient71', 'Patient45', 'Patient34', 'Patient35', 'Patient48', 'Patient39', 'Patient57',
                    'Patient43', 'Patient44', 'Patient60', 'Patient69', 'Patient53', 'Patient55', 'Patient62', 'Patient75', 'Patient77', 'Patient83', 'Patient52']
    # LOCAL CONTROL 
    # recidivist = ['Patient76', 'Patient05', 'Patient02', 'Patient09', 'Patient31', 'Patient45', 'Patient48', 'Patient59', 'Patient72', 'Patient75']
    # non_recidivist = ['Patient73', 'Patient81', 'Patient16', 'Patient01', 'Patient29', 'Patient10', 'Patient82', 'Patient07', 'Patient03', 'Patient04', 'Patient08', 'Patient12', 'Patient51', 
    #                   'Patient14', 'Patient06', 'Patient15', 'Patient13', 'Patient17', 'Patient21', 'Patient49', 'Patient25', 'Patient19', 'Patient23', 'Patient20', 'Patient22', 'Patient24', 
    #                   'Patient26', 'Patient28', 'Patient38', 'Patient33', 'Patient40', 'Patient30', 'Patient36', 'Patient71', 'Patient34', 'Patient35', 'Patient37', 'Patient39', 'Patient57', 
    #                   'Patient41', 'Patient42', 'Patient43', 'Patient44', 'Patient58', 'Patient60', 'Patient46', 'Patient63', 'Patient69', 'Patient53', 'Patient55', 'Patient56', 'Patient62', 
    #                   'Patient78', 'Patient70', 'Patient65', 'Patient67', 'Patient68', 'Patient74', 'Patient84', 'Patient77', 'Patient83', 'Patient52', 'Patient50', 'Patient47']
    
    print("Number of recidivist patients: ", len(recidivist))
    print("Number of non-recidivist patients: ", len(non_recidivist))

    patients = recidivist + non_recidivist
    patients_filtered = [x for x in patients if x not in error_patients]
    
    # print("Creating dilated PTV masks")
    # create_dilated_ptv_mask(patients_filtered)
    
    # print("Computing feature maps")
    # compute_feature_maps_same_mask(patients_filtered, fractions, folder_path, rad_maps_folder, params_path, enabled_features, mask_type)

    # cluster_delta_map(patients_filtered, fractions[0], rad_maps_folder, enabled_features, mask_type, method='gmm', k=5) # Cluster radiomic map with fixed k 
    # cluster_delta_map(patients_filtered, fractions[1], rad_maps_folder, enabled_features, mask_type, method='gmm', k=5) # Cluster radiomic map with fixed k 
# # 
    # compute_dissimilarity_map(patients_filtered, fractions, rad_maps_folder, enabled_features, mask_type) # Compute dissimilarity map between the two fractions
    for feature in enabled_features:
        entropies_recidivist = compute_cluster_site_entropies(recidivist, fractions, rad_maps_folder, feature, mask_type) # Compute cluster site entropy between the two fractions
        entropies_non_recidivist = compute_cluster_site_entropies(non_recidivist, fractions, rad_maps_folder, feature, mask_type) # Compute cluster site entropy between the two fractions
        print(f'Recidivist patients: {np.mean(entropies_recidivist[1])}, {np.std(entropies_recidivist[1])}')
        print(f'Non-recidivist patients: {np.mean(entropies_non_recidivist[1])}, {np.std(entropies_non_recidivist[1])}', end='\n\n')
        result, pval = compare_groups(entropies_recidivist[0], entropies_non_recidivist[0])
        if result == True: 
            print(f'Significant difference between recidivist and non-recidivist patients for {feature}.')
            print(f'Mean recidivist: {np.mean(entropies_recidivist[1])}, std: {np.std(entropies_recidivist[1])}')
            print(f'Mean non-recidivist: {np.mean(entropies_non_recidivist[1])}, std: {np.std(entropies_non_recidivist[1])}', end='\n\n')


if __name__ == '__main__':
    main()