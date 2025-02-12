'''Script to compare feature map parameters between patients.'''

import os 
import numpy as np
import get_map as gm 
import get_stats as gs
import clustering as cl 
import pandas as pd 
import shutil

def compute_feature_maps(fractions, patients, params_path, enabled_features):
    '''Compute feature maps for all given patients and fractions. Save them as .nrrd files. 
    Computations are made on fractions MRI that are registered to simulation MRI. Mask is GTV from simulation MRI. 
    Parameters:
    ----------
        fractions: list, list of fractions to compute feature maps for;
        patients: list, list of patients to compute feature maps for;
        params_path: str, path to the parameters file;
        enabled_features: list, list of enabled features;
        
    Returns:    
    '''
    
    for f in fractions:
        for p in patients:
            # if image == simu: modifications TODO 
            image_path = 'Data/' + p + '/img_dir/registered_' + p + '_mridian_' + f + '.nii'
            if os.path.exists(image_path) == False: # if fraction is missing 
                print('Image not found for ' + p + ' ' + f)
                continue 
            mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_mridian_gtv_oriented.nii' # standard simu GTV path
            if os.path.exists(mask_path) == False:                
                mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_MRIdian_gtv_oriented.nii' # other way to write GTV path
                if os.path.exists(mask_path) == False: # means that simu GTV does not exists 
                    print('Mask not found for ' + p, 'Use fraction 1 GTV instead. ')
                    mask_path = 'Data/' + p + '/mask_dir/' + p + '_mridian_ttt_1_gtv_oriented.nii' # use fraction 1 GTV otherwise (fractions were registered to F1 in this case)
             
            gm.generate_feature_map(image_path, mask_path, params_path, 'Data/' + p + '/rad_maps/' + f + '/', enabled_features)
            assert os.path.exists('Data/' + p + '/rad_maps/' + f + '/'), 'Feature map not created'
            

def compute_delta_maps(fractions, patients, enabled_features):
    '''Compute delta feature maps for all given patients and fractions. Save them as .nrrd file. 
    Parameters:
    ----------
        fractions: list, list of the 2 fractions to compute delta feature maps for;
        patients: list, list of patients to compute feature maps for;
        enabled_features: list, list of enabled features;
        
        Returns: None 

    '''
    for p in patients: 
        print(p)
        for f in enabled_features:
            print(f)
            mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_mridian_gtv_oriented.nii' # standard simu GTV path
            if os.path.exists(mask_path) == False:    
                mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_MRIdian_gtv_oriented.nii' # other way to write GTV path
                if os.path.exists(mask_path) == False: # means that simu GTV does not exists 
                    print('Mask not found for ' + p, 'Use fraction 1 GTV instead. ')
                    mask_path = 'Data/' + p + '/mask_dir/' + p + '_mridian_ttt_1_gtv_oriented.nii' # use fraction 1 GTV otherwise (fractions were registered to F1 in this case)
            
            gm.generate_delta_map2(mask_path=mask_path,  
                map_paths=['Data/' + p + '/rad_maps/' + fractions[0] + '/' + f + '.nrrd', 'Data/' + p + '/rad_maps/' + fractions[1] + '/' + f + '.nrrd'], 
                                  store_path='Data/' + p + '/rad_maps/delta/' + fractions[0] + '_' + fractions[1] + '/', feature_name=f)
        print('Delta maps computed for ', p)
    

def compute_clustered_delta_maps(fractions, patients, enabled_features, k):
    '''Compute clustered delta feature maps for all given patients and fractions. Save them as .nrrd file. 
    
    Parameters:
    ----------
    
    fractions: list, list of the 2 fractions to compute delta feature maps for;
    patients: list, list of patients to compute feature maps for;
    enabled_features: list, list of enabled features;
    k: int, number of clusters;
    
    Returns: None 
    
    '''
    for p in patients: 
        for f in enabled_features:
            mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_mridian_gtv_oriented.nii' # standard simu GTV path
            if os.path.exists(mask_path) == False:    
                mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_MRIdian_gtv_oriented.nii' # other way to write GTV path
                if os.path.exists(mask_path) == False: # means that simu GTV does not exists 
                    print('Mask not found for ' + p, 'Use fraction 1 GTV instead. ')
                    mask_path = 'Data/' + p + '/mask_dir/' + p + '_mridian_ttt_1_gtv_oriented.nii' # use fraction 1 GTV otherwise (fractions were registered to F1 in this case)
            
            cl.gen_clustered_map(delta_map_path='Data/' + p + '/rad_maps/delta/' + fractions[0] + '_' + fractions[1] + '/' + f + '.nrrd', 
                                    mask_path=mask_path, 
                                    store_path='Data/' + p + '/rad_maps/clustered_delta/' + fractions[0] + '_' + fractions[1] + '/', feature_name=f, k=k)
                
            print('Clustered delta maps computed for ', p)
            
def compute_params(fractions, patients, enabled_features): 
    '''Compute intensity parameters for each feature map and save in a csv file.
    Parameters:
    ----------
        fractions: list, list of fractions to compute feature maps for;
        patients: list, list of patients to compute feature maps for;
        enabled_features: list, list of enabled features;
        
        Returns: None 
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

def compare_params(outcomes, outcomes_df, fractions, enabled_features):
    '''Compare intensity parameters between patients of different outcome-group with Mann-Whitney U test.
    Parameters:
    ----------
        outcomes: list, list of outcomes to compare;
        outcomes_df: pandas.DataFrame, dataframe with outcomes;
        fractions: list, list of fractions to compare;
        enabled_features: list, list of enabled features;
        
        Returns:
    '''

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

def compute_delta_params(fractions, patients, enabled_features):
    '''Compute statistics for each delta feature map and save in a csv file.
    Parameters:
    ----------
        fractions: list, list of the 2 fractions to compute delta feature maps for;
        patients: list, list of patients to compute feature maps for;
        enabled_features: list, list of enabled features;
        
        Returns:
    '''
    # for each radiomics feature map, compute the intensity parameters, store them in a csv with patient ID as index 
    for feature in enabled_features: 
        stored_params_df = pd.DataFrame(index=patients, columns=['mean', 'std', 'min', 'max', 'cv', 'skewness', 'kurtosis']) # create df with patient ID as index

        for p in patients:
            rad_params = gm.compute_feature_map_params('Data/' + p + '/rad_maps/delta/' + fractions[0] + '_' + fractions[1] + '/' + feature + '.nrrd')
            if rad_params is not None:
                stored_params_df.loc[p] = rad_params
        if not os.path.exists('Data/intensity_params/' + fractions[0] + '_' + fractions[1] + '/'):
            os.makedirs('Data/intensity_params/' + fractions[0] + '_' + fractions[1] + '/')

        # remove empty rows before saving
        stored_params_df = stored_params_df.dropna()

        stored_params_df.to_csv('Data/intensity_params/' + fractions[0] + '_' + fractions[1] + '/' + feature + '_params.csv')

def main(): 
    # COMPUTE FEATURE MAPS
    params = 'params.yaml' 

    fractions = ['ttt_1', 'ttt_5'] # 
    # get list of folders in Data/ if the name of the folder begins by Patient 
    folder_path = '/home/tachennf/Documents/delta-rad/rad_maps/Data/'
    patients_list = os.listdir(folder_path)

    # patients_to_remove = ['Patient' + str(n) for n in [57, 32, 74, 82, 84, 85, 56, 63]]
    patients_to_remove = ['Patient32', 'Patient56', 'Patient57', 'Patient66', 'Patient14', 'Patient27', 'Patient80','Patient 85']
    patients_filtered = [p for p in patients_list if patients_list not in patients_to_remove]
    patients_filtered = ['Patient76'] # TODO: remove this line after first test 
    # enabled_features = ['original_firstorder_Entropy', 'original_gldm_DependenceEntropy', 'original_glrlm_GrayLevelNonUniformity']
    enabled_features = ['original_firstorder_Kurtosis', 'original_gldm_DependenceEntropy'] # 'original_glcm_Imc1', # TODO/ deal size issue witg glcm_Imc1 features

    # COMPUTE SIMPLE FEATURE MAPS AND ANALYZE THEIR PARAMETERS 
    # compute_feature_maps(fractions, patients_filtered, params, enabled_features) 
    # compute_params(fractions, patients_filtered, enabled_features) 

    # COMPUTE DELTA FEATURES MAPS 
    # TODO: modify .yaml file to have same pre-processing as Gladis 
    # compute_feature_maps(fractions, patients_filtered, params, enabled_features) # optional if already computed 
    
    # compute_delta_maps(fractions, patients_filtered, enabled_features) # optional if already computed
    compute_clustered_delta_maps(fractions, patients_filtered, enabled_features, 3) # optional if already computed
    # #gm.disp_map('Data/Patient77/rad_maps/delta/ttt_1_ttt_3/original_firstorder_Kurtosis.nrrd', 2)
    # compute_delta_params(fractions, patients_filtered, enabled_features)

if __name__ == '__main__':
    main()