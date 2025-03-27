'''Script to test for delta radiomic maps functions. '''
from get_map import generate_delta_map
import utils.clustering as cl

import numpy as np 
import matplotlib.pyplot as plt
import SimpleITK as sitk


def test_generate_delta_map():
    '''Function that aims to test the generation of a delta-radiomic map. '''
    mask_paths = ['./data/Patient01_mridian_ttt_1_gtv.nii', './data/Patient01_mridian_ttt_5_gtv.nii']
    map_paths = ['./outputs/Patient_01_F1/original_firstorder_Entropy.nrrd', './outputs/Patient_01_F5/original_firstorder_Entropy.nrrd']
    generate_delta_map(mask_paths, map_paths, 'original_firstorder_Entropy', './outputs/Patient_01_F1_F5/')

    delta_map = np.load('./outputs/Patient_01_F1_F5/original_firstorder_Entropy.npy')
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(delta_map[:, :, 36], cmap='viridis'), plt.colorbar(), plt.title('Entropy Delta radiomic map')
    plt.show()

def test_cluster_delta_map(): 
    '''Function that aims to test the clustering of a delta-radiomic map. ''' 

    delta_map_path = './outputs/Patient_01_F1_F5/original_firstorder_Entropy.nrrd'
    mask_path = './outputs/Patient_01_F1_F5/original_firstorder_Entropy_mask.npy'

    delta_map = np.load('./outputs/Patient_01_F1_F5/original_firstorder_Entropy.npy')

    cl.gen_clustered_map(delta_map_path, mask_path, './outputs/Patient_01_F1_F5/clustered/', 'original_firstorder_Entropy', k=3)

    clustered_map = np.load('./outputs/Patient_01_F1_F5/clustered/original_firstorder_Entropy.npy')

    # Display the clustered map
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(clustered_map[:, :, 36], cmap='viridis'), plt.colorbar(), plt.title('Clustered Entropy Delta radiomic map')
    plt.subplot(122), plt.imshow(delta_map[:, :, 36], cmap='viridis'), plt.colorbar(), plt.title('Entropy Delta radiomic map')
    plt.show()


def main(): 
    test_generate_delta_map()
    test_cluster_delta_map()

if __name__ == '__main__':
    main()