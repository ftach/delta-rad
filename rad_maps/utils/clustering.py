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
        "n_components": range(2, 5),
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

def gen_clustered_map(delta_map_path: str, mask_path: str, store_path: str, feature_name: str, k: int = 3, method: str = 'gmm') -> None: 
    '''Function to generate a clustered map based on a delta radiomic map. The clustered map is saved as npy(with nan) and nrrd (without nan).

    Parameters
    ----------
    delta_map_path: str, path to the delta radiomic map;
    mask_path: str, path to the mask;
    store_path: str, path to store the clustered map;
    feature_name: str, feature name;
    k: int, number of clusters;

    Returns
    -------
    None
    '''

    delta_map_img = sitk.ReadImage(delta_map_path) # load delta map nrrd (without nan and inf values) 
    delta_map = sitk.GetArrayFromImage(delta_map_img)
    delta_map = np.transpose(delta_map, (2, 1, 0)) # change the axis to have the same orientation as the mask
    
    if np.unique(delta_map).shape[0] > k: # 'Delta map has only one unique value. Clustering is not possible.'
        if method == 'gmm':
            clustered_map = gmm_cluster_map_cv(delta_map)
        elif method == 'kmeans':
            clustered_map = kmeans_cluster_map(delta_map, k)
        elif method == 'otsu':
            try:
                clustered_map = otsu_cluster_map(delta_map, k)
            except ValueError as e:
                print(f"Error: {e}. Using GMM instead.")
                clustered_map = gmm_cluster_map(delta_map, k)
                
        else:
            raise ValueError('Method not supported. Choose from gmm, kmeans, otsu.')
    clustered_map[clustered_map == 0] = -1 # change the cluster 0 to -1 - decrease 

    clustered_map[delta_map == 0] = 0 # set the values outside the mask to nan
    
    clustered_map = clustered_map.astype(float)

    if os.path.exists(store_path) == False: 
        os.makedirs(store_path)
    
    # save as nrrd to visualize the map
    nii_clustered_map = clustered_map.copy()
    nii_clustered_map = np.transpose(nii_clustered_map, (2, 1, 0))  # (x, y, z) → (z, x, y)
    nii_clustered_map = sitk.GetImageFromArray(nii_clustered_map)
    
    nii_clustered_map.SetSpacing(delta_map_img.GetSpacing())  # Set voxel spacing
    nii_clustered_map.SetDirection(delta_map_img.GetDirection())  # Set orientation
    nii_clustered_map.SetOrigin(delta_map_img.GetOrigin())  # Set origin
    
    sitk.WriteImage(nii_clustered_map, os.path.join(store_path, feature_name + '.nrrd'), True)
