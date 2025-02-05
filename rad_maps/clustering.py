'''Functions to cluster radiomics maps. '''

import numpy as np
from sklearn.cluster import KMeans
import SimpleITK as sitk
import os 

def cluster_map(map, k=3): 
    # test with 2D map first 
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(map.flatten().reshape(-1, 1))
    clustered_map = np.reshape(kmeans.labels_, map.shape)

    return clustered_map

def gen_clustered_map(delta_map_path, mask_path, store_path, feature_name, k=3): 
    delta_map = sitk.ReadImage(delta_map_path) # load delta map nrrd (without nan and inf values) 
    delta_map = sitk.GetArrayFromImage(delta_map) # get delta map array
    clustered_map = cluster_map(delta_map, k)
    clustered_map[clustered_map == 0] = -1 # change the cluster 0 to -1 - decrease 
    clustered_map[clustered_map == 1] = 0 # change the cluster 1 to 0 - no big change
    clustered_map[clustered_map == 2] = 1 # change the cluster 2 to 1 - increase
    mask = np.load(mask_path) # load mask 
    clustered_map = clustered_map.astype(float)
    clustered_map[mask == 0] = np.nan # mask the clustered map 

    if os.path.exists(store_path) == False: 
        os.makedirs(store_path)

    np.save(os.path.join(store_path, feature_name + '.npy'), clustered_map) # save the clustered map as npy to keep nan and inf values
    # save as nrrd to visualize the map
    nii_clustered_map = clustered_map.copy()
    nii_clustered_map[np.isnan(nii_clustered_map)] = 0
    nii_clustered_map[np.isinf(nii_clustered_map)] = 0

    nii_clustered_map = sitk.GetImageFromArray(nii_clustered_map)
    sitk.WriteImage(nii_clustered_map, os.path.join(store_path, feature_name + '.nrrd'), True)
