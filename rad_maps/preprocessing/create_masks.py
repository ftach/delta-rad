'''Functions to create mask from existing ones. '''
import nibabel as nib 
from skimage.morphology import disk, binary_dilation
import numpy as np 

def create_dilated_ptv_mask():
    folder_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/registered_data/'
    patients_filtered = ['Patient48', 'Patient76', 'Patient75', 'Patient72', 'Patient59', 'Patient46', 'Patient34', 'Patient36', 'Patient31', 'Patient12', 'Patient20', 'Patient22', 'Patient26', 'Patient39', 'Patient40']
    fractions = ['ttt_1', 'ttt_5'] 
    for p in patients_filtered: 
        for f in fractions:
            mask_path = folder_path + p + '/mask_dir/' + p + '_mridian_' + f + '_ptv.nii'
            mask = nib.load(mask_path)
            mask_data = mask.get_fdata()
            ptv5 = np.zeros(mask_data.shape)
            for k in range(mask_data.shape[2]):
                ptv5[:, :, k] = binary_dilation(mask_data[:, :, k], disk(5)) # 5px dilation
            nib.save(nib.Nifti1Image(ptv5.astype(np.uint8), mask.affine), mask_path.replace('.nii', '_5px.nii'))

def create_one_mask(): 
    folder_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/registered_data/'
    patients_filtered = ['Patient48']
    # patients_filtered = ['Patient48', 'Patient76', 'Patient75', 'Patient72', 'Patient59', 'Patient46', 'Patient34', 'Patient36', 'Patient31', 'Patient12', 'Patient20', 'Patient22', 'Patient26', 'Patient39', 'Patient40']
    for p in patients_filtered: 
        mask_path = folder_path + p + '/mask_dir/' + p + '_mridian_ttt_1_ptv.nii'
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata()

        one_mask = np.zeros(mask_data.shape)
        one_mask[90:200, 100:180, 10:60] = 1 # create a mask with the same size as the original one
        nib.save(nib.Nifti1Image(one_mask.astype(np.uint8), mask.affine), mask_path.replace('ttt_1_ptv.nii', 'full.nii'))
