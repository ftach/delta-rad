'''Performs registration between simu and fraction masks. '''

import utils.registration as r
import SimpleITK as sitk 
import matplotlib.pyplot as plt 
import numpy as np
import os 

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

def register_mridian_dataset(transformation='rigid', metric='pcc', mask='gtv'): # TODO implement really when tests are OK
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
    patient_list = [p for p in os.listdir(oriented_data_base_dir)]
    patient_list = ['Patient01']
    for p in patient_list:
        # forbidden_patients = ['Patient32', 'Patient56', 'Patient57', 'Patient66', 'Patient14', 'Patient27', 'Patient80']
        # if p in forbidden_patients:
        #     continue
        print(f"{p}")

        # check that we have patient folder 
        if os.path.exists(f'{oriented_data_base_dir}/{p}') == False: 
            continue
        
        output_patient_dir = f'/home/tachennf/Documents/delta-rad/data/ICM_0.35T/registered_data/{p}/'
        if os.path.exists(output_patient_dir) == False: 
            os.makedirs(output_patient_dir)

        # check that we have simu image 
        if os.path.exists(f'{oriented_data_base_dir}/{p}/img_dir/{p}_IRM_simu_mridian.nii') or os.path.exists(f'{oriented_data_base_dir}/{p}/img_dir/{p}_IRM_simu_MRIdian.nii'): # simu exists
            simu_path = f'{oriented_data_base_dir}/{p}/img_dir/{p}_IRM_simu_mridian.nii'
            label_simu_path = f'{oriented_data_base_dir}/{p}/mask_dir/{p}_IRM_simu_mridian_{mask}.nii'
            if os.path.exists(simu_path) == False : # MRIdian path 
                simu_path = f'{oriented_data_base_dir}/{p}/img_dir/{p}_IRM_simu_MRIdian.nii'
                label_simu_path = f'{oriented_data_base_dir}/{p}/mask_dir/{p}_IRM_simu_MRIdian_{mask}.nii'


            for i in range(1, 6): 
                print(f"Processing fraction {i} for patient {p}")
                f_path = f'{oriented_data_base_dir}/{p}/img_dir/{p}_mridian_ttt_{i}.nii'
                label_f_path = f'{oriented_data_base_dir}/{p}/mask_dir/{p}_mridian_ttt_{i}_{mask}.nii'
                if os.path.exists(f_path) == False: # if fraction is missing 
                    continue

                if os.path.exists(f'{output_patient_dir}/img_dir/') == False:
                    os.makedirs(f'{output_patient_dir}/img_dir/')

                if os.path.exists(f'{output_patient_dir}/mask_dir/') == False:
                    os.makedirs(f'{output_patient_dir}/mask_dir/')

                # Load images 
                simu_array = sitk.GetArrayFromImage(sitk.ReadImage(simu_path))
                f_array = sitk.GetArrayFromImage(sitk.ReadImage(f_path))

                # Load masks 
                simu_label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_simu_path))
                f_label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_f_path))

                registration_method, T, metric_values = r.affine_registration(simu_label_array, f_label_array, mask=True, transformation=transformation, metric=metric)

                # Apply the transformation to the fraction images
                registered_f_label_array = r.apply_3D_transform2(f_label_array, simu_label_array, T, mask=True)
                registered_f_array = r.apply_3D_transform2(f_array, simu_array, T, mask=False)

                dice_after = r.compute_dice(simu_label_array, registered_f_label_array)
                dice_after_list.append(dice_after)

                # save images
                output_f_path = f'{output_patient_dir}/img_dir/{p}_mridian_ttt_{i}.nii'
                output_f_label_path = f'{output_patient_dir}/mask_dir/{p}_mridian_ttt_{i}_{mask}.nii'
                output_simu_path = f'{output_patient_dir}/img_dir/{p}_IRM_simu_mridian.nii'
                output_simu_label_path = f'{output_patient_dir}/mask_dir/{p}_IRM_simu_mridian_{mask}.nii'

                sitk.WriteImage(sitk.GetImageFromArray(registered_f_array), output_f_path)
                sitk.WriteImage(sitk.GetImageFromArray(registered_f_label_array), output_f_label_path)
                sitk.WriteImage(sitk.GetImageFromArray(simu_array), output_simu_path)
                sitk.WriteImage(sitk.GetImageFromArray(simu_label_array), output_simu_label_path)


        else: # simu does not exist 
             
            # TODO: take care of this issue 
            simu_path = f'{oriented_data_base_dir}/{p}/img_dir/{p}_mridian_ttt_1.nii' # simu is F1 
            gtv_simu_path = f'{oriented_data_base_dir}/{p}/mask_dir/{p}_mridian_ttt_1_gtv.nii'
            if os.path.exists(simu_path) == False: # MRIdian path
                simu_path = f'{oriented_data_base_dir}/{p}/img_dir/{p}_IRM_MRIdian.nii'
                gtv_simu_path = f'{oriented_data_base_dir}/{p}/mask_dir/{p}_IRM_MRIdian_gtv.nii'

            for i in range(2, 6): 
                f_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/img_dir/{p}_mridian_ttt_{i}.nii'
                if os.path.exists(f_path) == False: # if fraction is missing 
                    continue
                gtv_f_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/mask_dir/{p}_mridian_ttt_{i}_gtv.nii'   
                output_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/img_dir/registered_{p}_mridian_ttt_{i}.nii'
                
                dice_before, dice_after = r.register_images(simu_path, f_path, gtv_simu_path, gtv_f_path, output_path, normalization='histogram', transformation=transformation, metric=metric)

                dice_after_list.append(dice_after)
    
    print("Registration is over. Here are the results: ")
    print(f"Average Dice after registration: {np.mean(dice_after_list)}")
    print(dice_after_list)

