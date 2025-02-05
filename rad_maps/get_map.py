'''Functions to get a radiomics map from a nii image. '''

import nibabel as nib 
import matplotlib.pyplot as plt
import numpy as np
from radiomics import featureextractor
import SimpleITK as sitk
import time
import six 
import os 

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
    # extractor.disableAllImageTypes()
    # extractor.disableAllFeatures()

    # compute features
    result = extractor.execute(img_path, roi_path, voxelBased=True)

    # save maps
    if os.path.exists(store_path) is False:
        os.makedirs(store_path)
    for key, val in six.iteritems(result):
        if isinstance(val, sitk.Image) and key in enabled_features:
            sitk.WriteImage(val, os.path.join(store_path, key + '.nrrd'), True)
    print('Elapsed time: {} s'.format(time.time() - start_time))

def generate_delta_map(map_paths, feature_name, store_path):
    """
        Generate delta-radiomics feature map.
    Parameters
    ----------
    map_paths: list, map paths to .nrrd files, length=2;
    feature_name: str, name of the feature computed;
    store_path: str, directory where to store the delta map;
    Returns
    -------
    """
    map1 = sitk.GetArrayFromImage(sitk.ReadImage(map_paths[0]))
    map2 = sitk.GetArrayFromImage(sitk.ReadImage(map_paths[1]))

    # TODO: check if the two maps have the same shape, otherwise use padding
    if map1.shape != map2.shape: 
        map1, map2 = pad_img(map1, map2)

    delta_map = (map1 - map2) / map2 # compute delta map 

    delta_map[np.abs(delta_map) == 1] = np.nan # we set to nan valus that are on the border of the delta-rad map 

    if os.path.exists(store_path) is False:
        os.makedirs(store_path) 

    # SAVE NUMPY ARRAY DELTA MAP to conserve nan for statistical analysis
    np.save(os.path.join(store_path, feature_name + '.npy'), delta_map)

    # SAVE NII DELTA MAP   
    nii_delta_map = delta_map.copy()
    nii_delta_map[np.isnan(nii_delta_map)] = 0
    nii_delta_map[np.isinf(nii_delta_map)] = 0

    assert np.any(np.isnan(nii_delta_map)) == False, "Error, NaN values in the delta map"    # check for nan values in the delta map array

    nii_delta_map = sitk.GetImageFromArray(nii_delta_map)
    sitk.WriteImage(nii_delta_map, os.path.join(store_path, feature_name + '.nrrd'), True)

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