'''Functions to get a radiomics map from a nii image. '''

import nibabel as nib 
import matplotlib.pyplot as plt
import numpy as np
from radiomics import featureextractor
import SimpleITK as sitk
import time
import six 
import os 

import utils.clustering as cl 

def compute_feature_maps(fractions: list, patients: list, params_path: str, enabled_features: str, mask_type: str = 'gtv') -> list:
    '''Compute feature maps for all given patients and fractions. Save them as .nrrd files. 
    Computations are made on fractions MRI that are registered to simulation MRI. Mask is GTV or PTV from simulation MRI. 

    Parameters:
    ----------
        fractions: list, list of fractions to compute feature maps for
        patients: list, list of patients to compute feature maps for
        params_path: str, path to the parameters file
        enabled_features: list, list of enabled features
        mask_type: str, type of mask to use. Options are 'gtv' or 'ptv'
        
    Returns:    
    ----------
        computed_features: list, list of computed features
    '''
    errors = []
    for f in fractions:
        for p in patients:
            # if image == simu: modifications TODO 
            image_path = 'Data/' + p + '/img_dir/registered_' + p + '_mridian_' + f + '.nii'
            if os.path.exists(image_path) == False: # if fraction is missing 
                image_path = 'Data/' + p + '/img_dir/' + p + '_mridian_' + f + '_oriented.nii' # if there was no simu, we don't have registered in front of the name
                if os.path.exists(image_path) == False:
                    #print('Image not found for ' + p + ' ' + f)
                    continue 
            mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_mridian_' + mask_type + '_oriented.nii' # standard simu mask path
            if os.path.exists(mask_path) == False:                
                mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_MRIdian_' + mask_type + '_oriented.nii' # other way to write mask path
                if os.path.exists(mask_path) == False: # means that simu mask does not exists 
                    #print('Mask not found for ' + p, 'Use fraction 1 mask instead. ')
                    mask_path = 'Data/' + p + '/mask_dir/' + p + '_mridian_ttt_1_' + mask_type + '_oriented.nii' # use fraction 1 mask otherwise (fractions were registered to F1 in this case)
            try: 
                computed_features = generate_feature_map(image_path, mask_path, params_path, 'Data/' + p + '/rad_maps/'  + mask_type + '/' + f + '/', enabled_features)
                assert os.path.exists('Data/' + p + '/rad_maps/' + mask_type + '/' + f + '/'), 'Feature map not created'
            except ValueError: 
                #print('Feature map not created for ', p, ' ', f)
                errors.append(p)
                continue
    #print('Feature maps not computed for ', errors)

    return computed_features
            

def compute_delta_maps(fractions: list, patients: list, enabled_features: list, mask_type: str = 'gtv') -> None:
    '''Compute delta feature maps for all given patients and fractions. Save them as .nrrd file. 
    Parameters:
    ----------
        fractions: list, list of the 2 fractions to compute delta feature maps for;
        patients: list, list of patients to compute feature maps for;
        enabled_features: list, list of enabled features;
        mask_type: str, type of mask to use. Options are 'gtv' or 'ptv';
        
        Returns: None 

    '''
    for p in patients: 
        for f in enabled_features:
            mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_mridian_' + mask_type + '_oriented.nii' # standard simu mask path
            if os.path.exists(mask_path) == False:    
                mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_MRIdian_' + mask_type + '_oriented.nii' # other way to write mask path
                if os.path.exists(mask_path) == False: # means that simu mask does not exists 
                    #print('Mask not found for ' + p, 'Use fraction 1 mask instead. ')
                    mask_path = 'Data/' + p + '/mask_dir/' + p + '_mridian_ttt_1_' + mask_type + '_oriented.nii' # use fraction 1 mask otherwise (fractions were registered to F1 in this case)
            
            try: 
                generate_delta_map2(mask_path=mask_path,  
                map_paths=['Data/' + p + '/rad_maps/' + mask_type + '/' + fractions[0] + '/' + f + '.nrrd', 'Data/' + p + '/rad_maps/' + mask_type + '/' + fractions[1] + '/' + f + '.nrrd'], 
                                  store_path='Data/' + p + '/rad_maps/' + mask_type + '/delta/' + fractions[0] + '_' + fractions[1] + '/', feature_name=f)
            except ValueError:
                #print('Delta maps NOT computed for ', p, ' ', f)
                continue
        #print('Delta maps computed for ', p)
    

def compute_clustered_delta_maps(fractions: list, patients: list, enabled_features: list, k: int, mask_type: str = 'gtv') -> None:
    '''Compute clustered delta feature maps for all given patients and fractions. Save them as .nrrd file. 
    
    Parameters:
    ----------
    
    fractions: list, list of the 2 fractions to compute delta feature maps for;
    patients: list, list of patients to compute feature maps for;
    enabled_features: list, list of enabled features;
    k: int, number of clusters;
    mask_type: str, type of mask to use. Options are 'gtv' or 'ptv';
    
    Returns: None 
    
    '''
    for p in patients: 
        for f in enabled_features:
            mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_mridian_' + mask_type + '_oriented.nii' # standard simu mask path
            if os.path.exists(mask_path) == False:    
                mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_MRIdian_' + mask_type + '_oriented.nii' # other way to write mask path
                if os.path.exists(mask_path) == False: # means that simu mask does not exists 
                    # print('Mask not found for ' + p, 'Use fraction 1 mask instead. ')
                    mask_path = 'Data/' + p + '/mask_dir/' + p + '_mridian_ttt_1_' + mask_type + '_oriented.nii' # use fraction 1 mask otherwise (fractions were registered to F1 in this case)
            
            try: 
                cl.gen_clustered_map(delta_map_path='Data/' + p + '/rad_maps/' + mask_type + '/delta/' + fractions[0] + '_' + fractions[1] + '/' + f + '.nrrd', 
                                    mask_path=mask_path, store_path='Data/' + p + '/rad_maps/' + mask_type + '/clustered_delta/' + fractions[0] + '_' + fractions[1] + '/', feature_name=f, k=k)
            except ValueError: 
                #print('Clustered {} delta maps NOT computed for {}'.format(f, p))
                continue
            except RuntimeError:
                #print('No delta maps was found for {}. NOT computed clustering'.format(f))
                continue
                
            # print('Clustered {} delta maps computed for {}'.format(f, p))

def generate_feature_map(img_path: str, roi_path: str, parameter_path: str, store_path: str, enabled_features: list) -> None:
    """Generate specific feature map based on kernel Radius.

    Parameters
    ----------

    img_path: str, candidate image path;
    roi_path: str, candidate ROI path;
    parameter_path: str, .yaml parameter path;
    store_path: str, directory where to store the feature maps;
    enabled_features: list, list of enabled features. Compute all features if None;

    Returns
    -------
    """
    start_time = time.time()
    extractor = featureextractor.RadiomicsFeatureExtractor(parameter_path, store_path)

    # compute features
    result = extractor.execute(img_path, roi_path, voxelBased=True)

    # save maps
    if os.path.exists(store_path) is False:
        os.makedirs(store_path)
    for key, val in six.iteritems(result):
        if 'all' in enabled_features: 
            enabled_features = [f for f in result.keys() if f.startswith('original')] # we save all the features
        if isinstance(val, sitk.Image) and key in enabled_features:
            sitk.WriteImage(val, os.path.join(store_path, key + '.nrrd'), True)

    print('Elapsed time: {} s'.format(time.time() - start_time))
    return enabled_features

def generate_delta_map(mask_paths: list, map_paths: list, feature_name: list, store_path: str) -> None:
    """
        Generate delta-radiomics feature map with two different masks. 

    Parameters
    ----------
    mask_paths: list, mask paths to .nii files, length=2;
    map_paths: list, map paths to .nrrd files, length=2;
    feature_name: str, name of the feature computed;
    store_path: str, directory where to store the delta map;

    Returns
    -------
    """

    # load mask1 and mask2 to get original image shape
    full_size_mask1 = nib.load(mask_paths[0]).get_fdata()
    full_size_mask2 = nib.load(mask_paths[1]).get_fdata()
    
    # Load maps computed with pyradiomics (mini because cropped)
    mini_map1_img = sitk.ReadImage(map_paths[0])
    mini_map1_array = sitk.GetArrayFromImage(mini_map1_img)
    mini_map1 = np.transpose(mini_map1_array, (2, 1, 0))

    mini_map2_img = sitk.ReadImage(map_paths[1])
    mini_map2_array = sitk.GetArrayFromImage(mini_map2_img)
    mini_map2 = np.transpose(mini_map2_array, (2, 1, 0)) # z, y, x -> x, y, z

    # Retrieve original image shape
    map1 = full_size_mask1.copy()
    map2 = full_size_mask2.copy()
    map1[np.where(full_size_mask1 == 1)] = mini_map1[np.where(mini_map1 != 0)]
    map2[np.where(full_size_mask2 == 1)] = mini_map2[np.where(mini_map2 != 0)]

    # check if the two maps have the same shape, otherwise use padding
    if map1.shape != map2.shape: 
        map1, map2 = pad_img(map1, map2)

    full_size_delta_map = (map1 - map2) / map2 # compute delta map 

    full_size_delta_map[np.abs(full_size_delta_map) == 1] = np.nan # we set to nan valus that are on the border of the delta-rad map 

    # SAVE DELTA MAPs
    if os.path.exists(store_path) is False:
        os.makedirs(store_path) 

    # Save numpy array to conserve nan for statistical analysis
    np.save(os.path.join(store_path, feature_name + '.npy'), full_size_delta_map)

    # Save as nrrd to visualize the map in 3D    
    nii_delta_map = full_size_delta_map.copy()
    nii_delta_map[np.isnan(nii_delta_map)] = 0
    nii_delta_map[np.isinf(nii_delta_map)] = 0
    nii_delta_map = np.transpose(nii_delta_map, (2, 1, 0))  # (x, y, z) → (z, y, x)

    nii_delta_map_sitk = sitk.GetImageFromArray(nii_delta_map)
    nii_delta_map_sitk.SetSpacing(mini_map1_img.GetSpacing())  # Set voxel spacing
    nii_delta_map_sitk.SetDirection(mini_map1_img.GetDirection())  # Set orientation
    nii_delta_map_sitk.SetOrigin(mini_map1_img.GetOrigin())  # Set origin

    sitk.WriteImage(nii_delta_map_sitk, os.path.join(store_path, feature_name + '.nrrd'), True)

    # Save delta map mask 
    if full_size_mask1.shape != full_size_mask2.shape:
        full_size_mask1, full_size_mask2 = pad_img(full_size_mask1, full_size_mask2)
    full_size_mask = np.logical_and(full_size_mask1, full_size_mask2).astype(np.uint8)
    np.save(os.path.join(store_path, feature_name + '_mask.npy'), full_size_mask) # save the mask as npy to keep nan and inf values

def generate_delta_map2(mask_path: str, map_paths: list, feature_name: list, store_path: str) -> None:
    """
        Generate delta-radiomics feature map with one same GTV. 

    Parameters
    ----------
    mask_path: str, mask path to .nii file, length=2;
    map_paths: list, map paths to .nrrd files, length=2;
    feature_name: str, name of the feature computed;
    store_path: str, directory where to store the delta map;

    Returns
    -------
    """
    # load mask to get original image shape
    full_size_mask = nib.load(mask_path).get_fdata()
    
    # Load maps computed with pyradiomics (mini because cropped)
    mini_map1_img = sitk.ReadImage(map_paths[0])
    mini_map1_array = sitk.GetArrayFromImage(mini_map1_img)
    mini_map1 = np.transpose(mini_map1_array, (2, 1, 0))

    mini_map2_img = sitk.ReadImage(map_paths[1])
    mini_map2_array = sitk.GetArrayFromImage(mini_map2_img)
    mini_map2 = np.transpose(mini_map2_array, (2, 1, 0)) # z, y, x -> x, y, z*
    
    # Retrieve original image shape
    map1 = full_size_mask.copy()
    map2 = full_size_mask.copy()
    map1[np.where(full_size_mask == 1)] = mini_map1[np.where(mini_map1 != 0)]
    map2[np.where(full_size_mask == 1)] = mini_map2[np.where(mini_map2 != 0)]

    # # normalize the maps to 0 - 1 range 
    map1 = (map1 - np.nanmin(map1)) / (np.nanmax(map1) - np.nanmin(map1))
    map2 = (map2 - np.nanmin(map2)) / (np.nanmax(map2) - np.nanmin(map2))

    # check if the two maps have the same shape, otherwise use padding
    if map1.shape != map2.shape: 
        map1, map2 = pad_img(map1, map2)

    map2 = map2 + 1e-5 # add a small value to avoid division by zero

    full_size_delta_map = (map1 - map2) / map2 # compute delta map 

    full_size_delta_map = (full_size_delta_map - np.nanmin(full_size_delta_map)) / (np.nanmax(full_size_delta_map) - np.nanmin(full_size_delta_map)) # normalize

    full_size_delta_map[np.abs(full_size_delta_map) == 1] = np.nan # we set to nan valus that are on the border of the delta-rad map 

    # SAVE DELTA MAPs
    if os.path.exists(store_path) is False:
        os.makedirs(store_path) 

    # Save numpy array to conserve nan for statistical analysis
    np.save(os.path.join(store_path, feature_name + '.npy'), full_size_delta_map)

    # Save as nrrd to visualize the map in 3D    
    nii_delta_map = full_size_delta_map.copy()
    nii_delta_map[np.isnan(nii_delta_map)] = 0
    nii_delta_map[np.isinf(nii_delta_map)] = 0
    nii_delta_map = np.transpose(nii_delta_map, (2, 1, 0))  # (x, y, z) → (z, y, x)

    nii_delta_map_sitk = sitk.GetImageFromArray(nii_delta_map)
    nii_delta_map_sitk.SetSpacing(mini_map1_img.GetSpacing())  # Set voxel spacing
    nii_delta_map_sitk.SetDirection(mini_map1_img.GetDirection())  # Set orientation
    nii_delta_map_sitk.SetOrigin(mini_map1_img.GetOrigin())  # Set origin

    sitk.WriteImage(nii_delta_map_sitk, os.path.join(store_path, feature_name + '.nrrd'), True)


def pad_img(X1: np.ndarray, X2: np.ndarray) -> np.ndarray: 
    '''Pad the images to the biggest size of both 

    Parameters
    ----------
    X1: numpy array, image 1;
    X2: numpy array, image 2;

    Returns
    -------
    X1, X2: numpy arrays, padded images;
    '''
    max_shape = np.maximum(X1.shape, X2.shape)
    if X1.shape[0] < max_shape[0]:
        X1_padded = np.zeros(max_shape)
        X1_padded[:X1.shape[0], :X1.shape[1], :X1.shape[2]] = X1
        X1 = X1_padded
    if X2.shape[0] < max_shape[0]:
        X2_padded = np.zeros(max_shape)
        X2_padded[:X2.shape[0], :X2.shape[1], :X2.shape[2]] = X2
        X2 = X2_padded
    if X1.shape[1] < max_shape[1]:
        X1_padded = np.zeros(max_shape)
        X1_padded[:X1.shape[0], :X1.shape[1], :X1.shape[2]] = X1
        X1 = X1_padded
    if X2.shape[1] < max_shape[1]:
        X2_padded = np.zeros(max_shape)
        X2_padded[:X2.shape[0], :X2.shape[1], :X2.shape[2]] = X2
        X2 = X2_padded
    if X1.shape[2] < max_shape[2]:
        X1_padded = np.zeros(max_shape)
        X1_padded[:X1.shape[0], :X1.shape[1], :X1.shape[2]] = X1
        X1 = X1_padded
    if X2.shape[2] < max_shape[2]:
        X2_padded = np.zeros(max_shape)
        X2_padded[:X2.shape[0], :X2.shape[1], :X2.shape[2]] = X2
        X2 = X2_padded

    return X1, X2


def disp_map(map_path: str, slice_num: int) -> None:
    """
        Display the map.
    Parameters
    ----------
    map_path: str, map path to .nrrd file;
    slice_num: int, slice number;
    Returns
    -------
    """
    feature_map = sitk.ReadImage(map_path)
    feature_map = sitk.GetArrayFromImage(feature_map)
    plt.imshow(feature_map[:, :, slice_num], cmap='inferno')
    plt.colorbar()
    plt.show()

def compute_feature_map_params(feature_map_path: str) -> tuple:
    """
        Compute intensity parameters of a given feature map.
    Parameters
    ----------
    feature_map_path: str, feature map path to .npy file;

    Returns
    mean, std, min, max, coefficient of variation, skewness, kurtosis
    -------
    """
    try: 
        feature_map = np.load(feature_map_path)
    except RuntimeError:
        print('Feature map not found. File path: ' + feature_map_path)
        return None
    
    mean = np.nanmean(feature_map) # we use the maps with nan so that the 0 values are not taken into account
    std = np.nanstd(feature_map)
    max_val = np.nanmax(feature_map)
    min_val = np.nanmin(feature_map)
    cv = std / mean
    skewness = np.nanmean(((feature_map - mean) / std) ** 3)
    kurtosis = np.nanmean(((feature_map - mean) / std) ** 4)

    return mean, std, min_val, max_val, cv, skewness, kurtosis