'''Performs registration between simu and fraction masks. '''

import utils.registration as r
import SimpleITK as sitk 
import matplotlib.pyplot as plt 
import numpy as np
import os 
import nibabel as nib

def quick_test(): 
    '''Test the registration function on a single fraction (F5).
    Plots metrics evolution according to iterations to observe convergence. 
    Display Dice score computed between GTV after registration. '''

    # load simu 
    simu = sitk.ReadImage('/home/tachennf/Documents/delta-rad/data/ICM_0.35T/oriented_data/Patient01/img_dir/Patient01_IRM_simu_MRIdian.nii')
    simu = sitk.GetArrayFromImage(simu)

    # load F5 image
    f5 = sitk.ReadImage('/home/tachennf/Documents/delta-rad/data/ICM_0.35T/oriented_data/Patient01/img_dir/Patient01_mridian_ttt_5.nii')
    f5 = sitk.GetArrayFromImage(f5)

    # COMPARE GTVs 
    gtv_simu = sitk.ReadImage('/home/tachennf/Documents/delta-rad/data/ICM_0.35T/oriented_data/Patient01/mask_dir/Patient01_IRM_simu_MRIdian_gtv.nii') # charger gtv simu 
    gtv_simu = sitk.GetArrayFromImage(gtv_simu)

    gtv_f5 = sitk.ReadImage('/home/tachennf/Documents/delta-rad/data/ICM_0.35T/oriented_data/Patient01/mask_dir/Patient01_mridian_ttt_5_gtv.nii') # charger gtv F5 
    gtv_f5 = sitk.GetArrayFromImage(gtv_f5)

    registration_method, T, metric_values = r.affine_registration(gtv_simu, gtv_f5, mask=True, transformation='rigid', metric='pcc')

    # Plot the metric evolution
    plt.plot(metric_values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Metric Value (Mutual Information)")
    plt.title("Convergence of Registration")
    plt.show()
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(
        f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    )

    registered_f5_gtv = r.apply_3D_transform2(gtv_f5, gtv_simu, T, mask=True)

    dice_before = r.compute_dice(gtv_simu, gtv_f5)
    dice_after = r.compute_dice(gtv_simu, registered_f5_gtv)
    print(f"Dice before registration: {dice_before}")
    print(f"Dice after registration: {dice_after}")

def register_mridian_dataset(transformation='rigid', metric='pcc', mask='gtv'): 
    '''Performs the registration between simulation image (fixed) and fractions (moving) using the given metrics and transformations.
    Displays the average Dice score before and after registration.

    Parameters:
    transformation: str, transformation to use for registration. Options are 'rigid' and 'affine'. 
    metric: str, metric to use for registration. Options are 'pcc' and 'mi'. 

    Returns:
    None
    '''

    dice_after_list = []
    oriented_data_base_dir = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/oriented_data/'
    # patient_list = [p for p in os.listdir(oriented_data_base_dir)]
    patient_list = ['Patient76', 'Patient75', 'Patient72', 'Patient59', 'Patient46', 'Patient34', 'Patient35', 'Patient36', 'Patient31', 'Patient12', 'Patient20', 'Patient22', 'Patient26', 'Patient39', 'Patient40']
    error_patients = []
    for p in patient_list:
        # forbidden_patients = ['Patient32', 'Patient57', 'Patient14', 'Patient27', 'Patient80', 'Patient77', 'Patient38', 'Patient79', 'Patient11', 'Patient54', 'Patient18', 'Patient85']
        # # Patient77 has no GTV but a PTV
        # # Patient38 has a weird error 
        # # Patients to checks: 11, 54, 18, 85, 02, 65
        # if p in forbidden_patients:
        #     continue
        print(f"{p}")

        # check that we have patient folder 
        output_patient_dir = f'/home/tachennf/Documents/delta-rad/data/ICM_0.35T/registered_data/{p}/'

        # if os.path.exists(f'{oriented_data_base_dir}/{p}') == False: 
        #     continue

        img_paths = [f for f in os.listdir(f'{oriented_data_base_dir}/{p}/img_dir/')]
        img_paths = sorted(img_paths)
        if os.path.exists(output_patient_dir) == False: 
            os.makedirs(output_patient_dir)

        if os.path.exists(f'{output_patient_dir}/img_dir/') == False:
            os.makedirs(f'{output_patient_dir}/img_dir/')
            
        if os.path.exists(f'{output_patient_dir}/mask_dir/') == False:
            os.makedirs(f'{output_patient_dir}/mask_dir/')
        
        simu_name = img_paths.pop(0)
        simu_path = f'{oriented_data_base_dir}/{p}/img_dir/' + simu_name
        label_simu_path = f'{oriented_data_base_dir}/{p}/mask_dir/' + simu_name.replace('.nii', f'_{mask}.nii')
        
        i = 0 
        for path in img_paths: 
            #print(f"Processing fraction {i+1} for {p}")
            dice_after = 0 

            f_path = f'{oriented_data_base_dir}/{p}/img_dir/' + path
            assert f_path != simu_path, 'Simu path and fraction paths are the sames'
            label_f_path = f'{oriented_data_base_dir}/{p}/mask_dir/' + path.replace('.nii', f'_{mask}.nii')

            output_f_path = f'{output_patient_dir}/img_dir/' + path
            output_f_label_path = f'{output_patient_dir}/mask_dir/' + path.replace('.nii', f'_{mask}.nii')
            output_simu_path = f'{output_patient_dir}/img_dir/' + simu_name
            output_simu_label_path = f'{output_patient_dir}/mask_dir/' + simu_name.replace('.nii', f'_{mask}.nii')

            if os.path.exists(f_path) == False: # if fraction is missing 
                raise ValueError("Error, path should exist")
            
            # Load images 
            simu_array = sitk.GetArrayFromImage(sitk.ReadImage(simu_path))
            f_array = sitk.GetArrayFromImage(sitk.ReadImage(f_path))

            # Load masks 
            simu_label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_simu_path))
            f_label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_f_path))

            if p == "Patient20" or p == "Patient33":
                # transform image to float 
                simu_array = simu_array.astype(np.float32)
                f_array = f_array.astype(np.float32)
                simu_label_array = simu_label_array.astype(np.float32)
                f_label_array = f_label_array.astype(np.float32)

            registration_method, T, metric_values = r.sitk_affine_registration(simu_label_array, f_label_array, mask=True, transformation=transformation, metric=metric)

            # Apply the transformation to the fraction images
            registered_f_label_array = r.apply_3D_transform2(f_label_array, simu_label_array, T, mask=True)
            registered_f_array = r.apply_3D_transform2(f_array, simu_array, T, mask=False)

            # compute dice 
            dice_after = r.compute_dice(simu_label_array, registered_f_label_array)
            k = 0
            while dice_after < 0.7: 
                print("Error with sitk registration (Dice was {}), using dipy registration: test {}".format(dice_after, k+1))
                # Perform registration using dipy 
                registered_f_label_array, T, popt = r.dipy_registration(nib.load(label_simu_path), nib.load(label_f_path), transformation=transformation)
                registered_f_label_array = np.transpose(registered_f_label_array, (2, 1, 0))
                assert registered_f_label_array.shape == simu_label_array.shape, 'Error, registered image and simu image should have the same shape'
                # Apply the transformation to the fraction images
                registered_f = r.apply_dipy_affine_transformation(nib.load(f_path), nib.load(simu_path), T) 
                registered_f_array = registered_f.get_fdata()
                registered_f_array = np.transpose(registered_f_array, (2, 1, 0))
                assert registered_f_array.shape == simu_array.shape, 'Error, registered image and simu image should have the same shape'
                dice_after = r.compute_dice(simu_label_array, registered_f_label_array)
                k += 1
                if k > 5: 
                    print("Error with dipy registration, too many iterations")
                    error_patients.append(p)
                    break
            
            dice_after_list.append(dice_after)

            # save images
            sitk.WriteImage(sitk.GetImageFromArray(registered_f_array), output_f_path)
            sitk.WriteImage(sitk.GetImageFromArray(registered_f_label_array), output_f_label_path)
            sitk.WriteImage(sitk.GetImageFromArray(simu_array), output_simu_path)
            sitk.WriteImage(sitk.GetImageFromArray(simu_label_array), output_simu_label_path)

            i += 1 
    
    print("Registration is over. Here are the results: ")
    print(f"Average Dice after registration: {np.mean(dice_after_list)}")
    print("Error patients: ", error_patients)
