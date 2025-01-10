'''Functions to get a radiomics map from a nii image. '''

import nibabel as nib 
import matplotlib.pyplot as plt
import numpy as np
from radiomics import featureextractor
import SimpleITK as sitk
import time
import six 
import os 

def generate_feature_map(img_path, roi_path, parameter_path, store_path):
    """
        Generate specific feature map based on kernel Radius.
    Parameters
    ----------
    img_path: str, candidate image path;
    roi_path: str, candidate ROI path;
    parameter_path: str, .yaml parameter path;
    store_path: str;
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
    for key, val in six.iteritems(result):
        if isinstance(val, sitk.Image):
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

