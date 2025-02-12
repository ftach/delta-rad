'''Quick script to orient patient's GTV in the right direction. '''

import os 
import SimpleITK as sitk
import numpy as np

def main(): 
    patient_list = [p for p in os.listdir('/home/tachennf/Documents/delta-rad/rad_maps/Data/')]
    for p in patient_list:
        forbidden_patients = ['Patient32', 'Patient56', 'Patient57', 'Patient66', 'Patient14', 'Patient27', 'Patient80', 'Patient20', 'Patient32']
        if p in forbidden_patients:
            continue
        mask_path = 'Data/' + p + '/mask_dir/' + p + '_mridian_ttt_1_gtv.nii' 
        # mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_mridian_gtv.nii' # standard simu GTV path
        if os.path.exists(mask_path) == False:
            continue
            # mask_path = 'Data/' + p + '/mask_dir/' + p + '_IRM_simu_MRIdian_gtv.nii' # other way to write GTV path
            # if os.path.exists(mask_path) == False: # means that simu GTV does not exists 
            #     continue 
        gtv_simu = sitk.ReadImage(mask_path)
        gtv_simu_array = sitk.GetArrayFromImage(gtv_simu)
        gtv_simu_array = np.transpose(gtv_simu_array, (0, 2, 1))# orient GTV in the right direction
        gtv_simu = sitk.GetImageFromArray(gtv_simu_array)
        mask_path = mask_path.replace('gtv', 'gtv_oriented')
        sitk.WriteImage(gtv_simu, mask_path)
        print('GTV oriented for ', p)

if __name__ == '__main__':
    main()