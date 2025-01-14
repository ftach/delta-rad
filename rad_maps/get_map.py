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