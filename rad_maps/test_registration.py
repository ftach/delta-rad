import registration as r
import nibabel as nib  
import SimpleITK as sitk 
import matplotlib.pyplot as plt 
import numpy as np
import os 
import sys 
import io 


def test_registration(): 
    # charger simu 
    simu = sitk.ReadImage('/home/tachennf/Documents/delta-rad/rad_maps/Data/Patient76/img_dir/Patient76_IRM_simu_mridian.nii')
    simu = sitk.GetArrayFromImage(simu)

    # charger image F5
    f5 = sitk.ReadImage('/home/tachennf/Documents/delta-rad/rad_maps/Data/Patient76/img_dir/Patient76_mridian_ttt_5.nii')
    f5 = sitk.GetArrayFromImage(f5)

    # COMPARE GTVs 
    gtv_simu = sitk.ReadImage('/home/tachennf/Documents/delta-rad/rad_maps/Data/Patient76/mask_dir/Patient76_IRM_simu_mridian_gtv.nii') # charger gtv simu 
    gtv_simu = sitk.GetArrayFromImage(gtv_simu)
    gtv_f5 = sitk.ReadImage('/home/tachennf/Documents/delta-rad/rad_maps/Data/Patient76/mask_dir/Patient76_mridian_ttt_5_gtv.nii') # charger gtv F5 
    gtv_f5 = sitk.GetArrayFromImage(gtv_f5)

    registration_method, T, metric_values = r.affine_registration(simu, f5)

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
    
    mse_before = r.compute_mse(simu, f5)
    print("MSE before registration: ", mse_before)

    registered_f5 = r.apply_3D_transform2(f5, simu, T, mask=False)

    mse_after = r.compute_mse(simu, registered_f5)
    print("MSE before registration: ", mse_after)

    # sauvegarder les images
    transformed_img = sitk.GetImageFromArray(registered_f5)
    transformed_img = sitk.TransformGeometry(transformed_img, T)

    # Print information about image 
    # f5_img = sitk.GetImageFromArray(f5)
    # print(f"origin before: {f5_img.GetOrigin()}\norigin after: {transformed_img.GetOrigin()}")
    # print(f"direction cosine before: {f5_img.GetDirection()}\ndirection cosine after: {transformed_img.GetDirection()}")


def main():
    mse_before_list = []
    mse_after_list = []

    patient_list = [p for p in os.listdir('/home/tachennf/Documents/delta-rad/rad_maps/Data/')]
    for p in patient_list:
        forbidden_patients = ['Patient32', 'Patient56', 'Patient57', 'Patient66', 'Patient14', 'Patient27', 'Patient80']
        if p in forbidden_patients:
            continue
        print(f"{p}")

        # check that we have patient folder 
        if os.path.exists(f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}') == False: 
            continue
        simu_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/img_dir/{p}_IRM_simu_mridian.nii'
        if os.path.exists(simu_path) or os.path.exists(f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/img_dir/{p}_IRM_MRIdian.nii'): # simu exists
            for i in range(1, 6): 
                f_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/img_dir/{p}_mridian_ttt_{i}.nii'
                if os.path.exists(f_path) == False: # if fraction is missing 
                    continue

                output_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/img_dir/registered_{p}_mridian_ttt_{i}.nii'

                if os.path.exists(simu_path): # standard path 
                    mse_before, mse_after = r.register_images(simu_path, f_path, output_path, normalization='histogram')

                else: # MRIdian path 
                    simu_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/img_dir/{p}_IRM_simu_MRIdian.nii'
                    mse_before, mse_after = r.register_images(simu_path, f_path, output_path, normalization='histogram')

                mse_before_list.append(mse_before)
                mse_after_list.append(mse_after)

        else: # simu does not exist 
            for i in range(2, 6): 
                simu_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/img_dir/{p}_mridian_ttt_1.nii' # simu is F1 
                
                f_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/img_dir/{p}_mridian_ttt_{i}.nii'
                if os.path.exists(f_path) == False: # if fraction is missing 
                    continue

                output_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/mask_dir/registered_{p}_mridian_ttt_{i}_gtv.nii'

                if os.path.exists(simu_path): # standard path 
                    mse_before, mse_after = r.register_images(simu_path, f_path, output_path, normalization='histogram')

                else: # MRIdian path 
                    simu_path = f'/home/tachennf/Documents/delta-rad/rad_maps/Data/{p}/img_dir/{p}_IRM_MRIdian.nii'
                    mse_before, mse_after = r.register_images(simu_path, f_path, output_path, normalization='histogram')

                mse_before_list.append(mse_before)
                mse_after_list.append(mse_after)
    
    print("Registration is over. Here are the results: ")
    print(f"Average MSE before registration: {np.mean(mse_before_list)}")
    print(f"Average MSE after registration: {np.mean(mse_after_list)}")
    print(mse_after_list)
if __name__ == '__main__': 
    main()
    # test_registration()