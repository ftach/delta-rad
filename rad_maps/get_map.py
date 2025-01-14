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

def generate_delta_map(map_paths, fraction_names, store_path):
    """
        Generate delta-radiomics feature map.
    Parameters
    ----------
    map_paths: list, map paths to .nrrd files, length=2;
    fraction_names: list, list of fraction names, length=2;
    store_path: str, directory where to store the delta map;
    Returns
    -------
    """
    map1 = sitk.GetArrayFromImage(sitk.ReadImage(map_paths[0]))
    map2 = sitk.GetArrayFromImage(sitk.ReadImage(map_paths[1]))

    # TODO: check if the two maps have the same shape, otherwise use padding
    if map1.shape != map2.shape: 
        target_shape = (
            max(map1.shape[0], map2.shape[0]),  # Max height
            max(map1.shape[1], map2.shape[1]),  # Max width
            max(map1.shape[2], map2.shape[2])   # Max depth
        )
        map1 = pad_to_shape(map1, target_shape)
        map2 = pad_to_shape(map2, target_shape)

    map2[map2==0] = 1e-12 # add a little value to the voxels of map2 equal to zero to avoid division by zero

    delta_map = (map1 - map2) / map2 # compute delta map 

    assert np.any(np.isnan(delta_map)) == False, "Error, NaN values in the delta map"    # check for nan values in the delta map array

    delta_map = sitk.GetImageFromArray(delta_map)

    if os.path.exists(store_path) is False:
        os.makedirs(store_path) 
    sitk.WriteImage(delta_map, os.path.join(store_path, fraction_names[0] + '_' + fraction_names[1] + '.nrrd'), True)

def pad_to_shape(image, target_shape):
    '''Pad one image to a given shape. 
    Parameters
    image: np.ndarray;
    target_shape: tuple, length=3, x, y, z dimensions for padding
    '''

    # Calculate the padding for each dimension
    pad_width = [
        ((target_shape[i] - image.shape[i]) // 2,  # Padding before
         (target_shape[i] - image.shape[i] + 1) // 2)  # Padding after
        for i in range(len(target_shape))
    ]

    # Apply padding
    return np.pad(image, pad_width, mode='constant', constant_values=0)


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