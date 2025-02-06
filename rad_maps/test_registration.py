import registration as r
import nibabel as nib  
import SimpleITK as sitk 
import matplotlib.pyplot as plt 

def main(): 
    # charger simu 
    simu = sitk.ReadImage('/home/tachennf/Documents/delta-rad/rad_maps/Data/Patient76/img_dir/Patient76_IRM_simu_mridian.nii')
    simu = sitk.GetArrayFromImage(simu)

    # charger image F5
    f5 = sitk.ReadImage('/home/tachennf/Documents/delta-rad/rad_maps/Data/Patient76/img_dir/Patient76_mridian_ttt_5.nii')
    f5 = sitk.GetArrayFromImage(f5)

    print(simu.shape, f5.shape)

    # Display intermediate images 

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
    
    registered_f5 = r.apply_3D_transform2(f5, simu, T, mask=False)

    # COMPARE GTVs 

    gtv_simu = sitk.ReadImage('/home/tachennf/Documents/delta-rad/rad_maps/Data/Patient76/mask_dir/Patient76_IRM_simu_mridian_gtv.nii') # charger gtv simu 
    gtv_simu = sitk.GetArrayFromImage(gtv_simu)
    gtv_f5 = sitk.ReadImage('/home/tachennf/Documents/delta-rad/rad_maps/Data/Patient76/mask_dir/Patient76_mridian_ttt_5_gtv.nii') # charger gtv F5 
    gtv_f5 = sitk.GetArrayFromImage(gtv_f5)

    mse_before = r.compute_mse(gtv_simu, gtv_f5)
    print("MSE before registration: ", mse_before)

    registered_gtv_f5 = r.apply_3D_transform2(gtv_f5, gtv_simu, T, mask=True)
    # comparer gtv simu et F5 

    mse_after = r.compute_mse(gtv_simu, registered_gtv_f5)
    print("MSE after registration:", mse_after)

    # plt.figure()
    # plt.subplot(131), plt.imshow(gtv_simu[:, :, 36]), plt.title('Simu')
    # plt.subplot(132), plt.imshow(gtv_f5[:, :, 36]), plt.title('F5')
    # plt.subplot(133), plt.imshow(registered_gtv_f5[:, :, 36]), plt.title('R-F5')
    # plt.show()
    print(registered_gtv_f5.shape)
    # sauvegarder les images
    transformed_img = sitk.GetImageFromArray(registered_gtv_f5)
    transformed_img = sitk.TransformGeometry(transformed_img, T)

    f5_img = sitk.GetImageFromArray(f5)

    print(
        f"origin before: {f5_img.GetOrigin()}\norigin after: {transformed_img.GetOrigin()}"
    )
    print(
        f"direction cosine before: {f5_img.GetDirection()}\ndirection cosine after: {transformed_img.GetDirection()}"
    )

    sitk.WriteImage(transformed_img, '/home/tachennf/Documents/delta-rad/rad_maps/Data/Patient76/mask_dir/registered_Patient76_mridian_ttt_5_gtv.nii')

if __name__ == '__main__': 
    main()