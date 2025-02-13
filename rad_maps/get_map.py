'''Functions to get a radiomics map from a nii image. '''

import nibabel as nib 
import matplotlib.pyplot as plt
import numpy as np
from radiomics import featureextractor
import SimpleITK as sitk
import time
import six 
import os 
from skimage.morphology import ball, erosion

def generate_feature_map(img_path, roi_path, parameter_path, store_path, enabled_features):
    """
        Generate specific feature map based on kernel Radius.
    Parameters
    ----------
    img_path: str, candidate image path;
    roi_path: str, candidate ROI path;
    parameter_path: str, .yaml parameter path;
    store_path: str, directory where to store the feature maps;
    enabled_features: list, list of enabled features;
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
        if isinstance(val, sitk.Image) and key in enabled_features:
            sitk.WriteImage(val, os.path.join(store_path, key + '.nrrd'), True)
    print('Elapsed time: {} s'.format(time.time() - start_time))

def generate_delta_map(mask_paths, map_paths, feature_name, store_path):
    """
        Generate delta-radiomics feature map with two different GTVs. 

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

def generate_delta_map2(mask_path, map_paths, feature_name, store_path):
    """
        Generate delta-radiomics feature map with one same GTV. 

    Parameters
    ----------
    mask_paths: list, mask paths to .nii files, length=2;
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


def pad_img(X1, X2): 
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


def disp_map(map_path, slice_num):
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

def compute_feature_map_params(feature_map_path):
    """
        Compute intensity parameters of a given feature map.
    Parameters
    ----------
    feature_map_path: str, feature map path to .nrrd file;

    Returns
    mean, std, min, max, coefficient of variation, skewness, kurtosis
    -------
    """
    try: 
        feature_map = sitk.ReadImage(feature_map_path)
        feature_map = sitk.GetArrayFromImage(feature_map)
    except RuntimeError:
        print('Feature map not found. File path: ' + feature_map_path)
        return None
    
    mean = np.mean(feature_map)
    std = np.std(feature_map)
    max_val = np.max(feature_map)
    min_val = np.min(feature_map)
    cv = std / mean
    skewness = np.mean(((feature_map - mean) / std) ** 3)
    kurtosis = np.mean(((feature_map - mean) / std) ** 4)

    return mean, std, min_val, max_val, cv, skewness, kurtosis