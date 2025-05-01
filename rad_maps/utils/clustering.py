'''Functions to cluster radiomics maps. '''

import numpy as np
from sklearn.cluster import KMeans
from skimage.filters import threshold_multiotsu
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

import SimpleITK as sitk
import os 

def kmeans_cluster_map(map: np.ndarray, k: int = 3) -> np.ndarray: 
    '''Function to cluster a radiomics map based on KMeans algorithm. 
    Parameters
    ----------
    map: np.array, radiomics map to cluster (2D or 3D);
    
    Returns
    -------
    clustered_map: np.array, clustered map (2D or 3D);
    '''
    kmeans = KMeans(n_clusters=k, random_state=0, init="k-means++").fit(map.flatten().reshape(-1, 1))
    clustered_map = np.reshape(kmeans.labels_, map.shape)

    return clustered_map

def otsu_cluster_map(map: np.ndarray, k: int = 3):
    '''Function to cluster a radiomics map based on Otsu algorithm. 
    Parameters
    ----------
    map: np.array, radiomics map to cluster (2D or 3D);
    
    Returns
    -------
    clustered_map: np.array, clustered map (2D or 3D);
    '''
    thresholds = threshold_multiotsu(map, classes=k, nbins=512)
    clustered_map = np.digitize(map, bins=thresholds)

    return clustered_map

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def gmm_cluster_map_cv(map: np.ndarray) -> np.ndarray:
    '''Function to cluster a radiomics map based on Gaussian Mixture Model.  K is chosen based on BIC score.

    Parameters
    ----------
    map: np.array, radiomics map to cluster (2D or 3D);
    
    Returns
    -------
    clustered_map: np.array, clustered map (2D or 3D);
    '''
    param_grid = {
        "n_components": range(2, 10),
    }
    grid_search = GridSearchCV(GaussianMixture(covariance_type='full', random_state=0), param_grid=param_grid, scoring=gmm_bic_score)
    grid_search.fit(map.flatten().reshape(-1, 1))
    clustered_map = np.reshape(grid_search.predict(map.flatten().reshape(-1, 1)), map.shape)

    return clustered_map

def gmm_cluster_map(map: np.ndarray, k: int) -> np.ndarray:
    '''Function to cluster a radiomics map based on Gaussian Mixture Model. 

    Parameters
    ----------
    map: np.array, radiomics map to cluster (2D or 3D);
    
    Returns
    -------
    clustered_map: np.array, clustered map (2D or 3D);
    '''
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0).fit(map.flatten().reshape(-1, 1))
    clustered_map = np.reshape(gmm.predict(map.flatten().reshape(-1, 1)), map.shape)

    return clustered_map

def gen_clustered_map(rad_map_path: str, store_path: str, feature_name: str, k: int = None, method: str = 'gmm', replace_value: bool = True) -> None: 
    '''Function to generate a clustered map based on a radiomic map. The clustered map is saved as nrrd file.

    Parameters
    ----------
    rad_map_path: str, path to the delta radiomic map;
    store_path: str, path to store the clustered map;
    feature_name: str, feature name;
    k: int, number of clusters;

    Returns
    -------
    None
    '''

    rad_map_img = sitk.ReadImage(rad_map_path) # load delta map nrrd (without nan and inf values) 
    rad_map = sitk.GetArrayFromImage(rad_map_img)
    rad_map = np.transpose(rad_map, (2, 1, 0)) # change the axis to have the same orientation as the mask

    if k is None: # if k is not specified, use GMM with cross validation to find the best k
        clustered_map = gmm_cluster_map_cv(rad_map)
        
    elif np.unique(rad_map).shape[0] > k: # 'Delta map has only one unique value. Clustering is not possible.'
        if method == 'gmm':
            clustered_map = gmm_cluster_map(rad_map, k)
        elif method == 'kmeans':
            clustered_map = kmeans_cluster_map(rad_map, k)
        elif method == 'otsu':
            try:
                clustered_map = otsu_cluster_map(rad_map, k)
            except ValueError as e:
                print(f"Error: {e}. Using GMM instead.")
                clustered_map = gmm_cluster_map(rad_map, k)
                
        else:
            raise ValueError('Method not supported. Choose from gmm, kmeans, otsu.')
    clustered_map[clustered_map == 0] = -1 # change the cluster 0 to -1 - decrease 

    clustered_map[rad_map == 0] = 0 # set the values outside the mask to nan
            
    modified_clustered_map = clustered_map.copy()
    if replace_value: 
        # assign each subregion value to the mean value of the subregion rad map 
        for i in range(len(np.unique(clustered_map))): 
            subregion = rad_map[clustered_map == i]
            if subregion.size > 0: # check if the subregion is not empty
                modified_clustered_map[clustered_map == i] = np.mean(subregion)

    clustered_map = modified_clustered_map.astype(float)

    if os.path.exists(store_path) == False: 
        os.makedirs(store_path)
    
    # save as nrrd to visualize the map
    nii_clustered_map = clustered_map.copy()
    nii_clustered_map = np.transpose(nii_clustered_map, (2, 1, 0))  # (x, y, z) → (z, x, y)
    nii_clustered_map = sitk.GetImageFromArray(nii_clustered_map)
    
    nii_clustered_map.SetSpacing(rad_map_img.GetSpacing())  # Set voxel spacing
    nii_clustered_map.SetDirection(rad_map_img.GetDirection())  # Set orientation
    nii_clustered_map.SetOrigin(rad_map_img.GetOrigin())  # Set origin
    
    sitk.WriteImage(nii_clustered_map, os.path.join(store_path, feature_name + '.nrrd'), True)

def dissimilarity_map(map1_path: str, map2_path: str, store_path: str, feature_name: str) -> None:
    '''Function to compute the dissimilarity map as the euclidean distance between two maps. Saved as nrrd file.
    Parameters
    ----------
    map1_path: str, path to the first map;
    map2_path: str, path to the second map;
    
    Returns
    -------
    dissimilarity_map: np.array, dissimilarity map;
    '''
    map1_img = sitk.ReadImage(map1_path) # load delta map nrrd (without nan and inf values)
    map1_array = sitk.GetArrayFromImage(map1_img)
    map1_array = np.transpose(map1_array, (2, 1, 0)) # change the axis to have the same orientation as the mask

    map2_img = sitk.ReadImage(map2_path) # load delta map nrrd (without nan and inf values)
    map2_array = sitk.GetArrayFromImage(map2_img)
    map2_array = np.transpose(map2_array, (2, 1, 0)) # change the axis to have the same orientation as the mask

    # compute euclidean distance between two maps
    dissimilarity_map = (map1_array - map2_array) ** 2

    # save as nrrd to visualize the map
    dissimilarity_map = dissimilarity_map.astype(float)
    
    # save as nrrd to visualize the map
    dissimilarity_map = np.transpose(dissimilarity_map, (2, 1, 0))  # (x, y, z) → (z, x, y)
    dissimilarity_map = sitk.GetImageFromArray(dissimilarity_map)

    dissimilarity_map.SetSpacing(map1_img.GetSpacing())  # Set voxel spacing
    dissimilarity_map.SetDirection(map1_img.GetDirection())  # Set orientation
    dissimilarity_map.SetOrigin(map1_img.GetOrigin())  # Set origin
    
    sitk.WriteImage(dissimilarity_map, os.path.join(store_path, feature_name + '_dissimilarity.nrrd'), True)

def cluster_site_entropy(cluster_img_path1: str, cluster_img_path2: str) -> float: 
    '''Computes the cluster site entropy for a given patient and two fractions. '''

    # Load the clustered maps
    clustered_img1 = sitk.ReadImage(cluster_img_path1) # load delta map nrrd (without nan and inf values)
    clustered_map1 = sitk.GetArrayFromImage(clustered_img1)
    clustered_map1 = np.transpose(clustered_map1, (2, 1, 0)) # change the axis to have the same orientation as the mask

    clustered_img2 = sitk.ReadImage(cluster_img_path2) # load delta map nrrd (without nan and inf values)
    clustered_map2 = sitk.GetArrayFromImage(clustered_img2)
    clustered_map2 = np.transpose(clustered_map2, (2, 1, 0)) # change the axis to have the same orientation as the mask
    
    cluster_site_entropy = 0 
    dissimilarity_map = (clustered_map1 - clustered_map2) ** 2

        # subregion1 = clustered_map1[clustered_map1 == k]
        # subregion2 = clustered_map2[clustered_map2 == k]
        # if (subregion1.size > 0) and (subregion2.size > 0):

            # # dissimilarity value 
            # dissimilarity = np.sqrt(np.sum((subregion1 - subregion2) ** 2))
            # print(f'Dissimilarity for subregion {k}: {dissimilarity}')

            # compute the entropy of the subregion pair 
    dissimilarity_map[dissimilarity_map == 0] = 1e-6 # avoid log(0)
    cluster_site_entropy = np.sum(dissimilarity_map * np.log2(dissimilarity_map))
    cluster_site_entropy2 = np.sqrt(np.sum(dissimilarity_map)) * np.log2(np.sqrt(np.sum(dissimilarity_map)))


    return cluster_site_entropy, cluster_site_entropy2
