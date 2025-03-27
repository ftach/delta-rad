'''Quick script to orient all the images in the right direction. '''

import os 
import SimpleITK as sitk
import numpy as np

def main(mask_type='gtv', simu = True): 
    patient_list = [p for p in os.listdir('/home/tachennf/Documents/delta-rad/rad_maps/Data/')]
    for p in patient_list:
        forbidden_patients = ['Patient32', 'Patient56', 'Patient57', 'Patient66', 'Patient14', 'Patient27', 'Patient80', 'Patient20', 'Patient32']
        if p in forbidden_patients:
            continue
        if simu: 
            mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_mridian_' + mask_type + '.nii' # standard path
            if os.path.exists(mask_path) == False:
                mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_MRIdian_gtv.nii' # other way to write path
                if os.path.exists(mask_path) == False: # if simu is missing 
                    continue
        else:
            mask_path = 'Data/' + p + '/mask_dir/' + p + '_mridian_ttt_1_' + mask_type + '.nii'
            if os.path.exists(mask_path) == False: # if fraction is missing
                continue

        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)
        mask_array = np.transpose(mask_array, (0, 2, 1))# orient GTV in the right direction
        mask = sitk.GetImageFromArray(mask_array)
        mask_path = mask_path.replace(mask_type, mask_type + '_oriented')
        sitk.WriteImage(mask, mask_path)
        print(mask_type, ' oriented for ', p)

if __name__ == '__main__':
    main('ptv', simu = True)

