'''Functions to register fractions to simu image. 

Adapted from https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/ 

'''

import numpy as np 
import SimpleITK as sitk 
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import (
    _ni_support,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)

from dipy.align import affine_registration
from scipy.ndimage import affine_transform
from dipy.align.imaffine import AffineMap

def align_centers(mz_stack: np.ndarray, moving_mz_stack: np.ndarray, mask: bool = False): 
    ''' Align the physical centers of two images.
    
    Parameters:
        mz_stack (np.ndarray): The fixed image.
        moving_mz_stack (np.ndarray): The moving image.
        
    Returns:
        initial_transform: sitk.Transform, The initial transformation.
        moving_resampled: np.ndarray, The moving image resampled.
    '''

    if mask:
        interp = sitk.sitkNearestNeighbor
    else:
        interp = sitk.sitkLinear

    initial_transform = sitk.CenteredTransformInitializer(
        sitk.GetImageFromArray(mz_stack),
        sitk.GetImageFromArray(moving_mz_stack),
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    moving_resampled = sitk.Resample(
        sitk.GetImageFromArray(moving_mz_stack),
        sitk.GetImageFromArray(mz_stack), # reference 
        initial_transform,
        interp,
        0.0,
        sitk.GetImageFromArray(moving_mz_stack).GetPixelID(),
    )

    return initial_transform, sitk.GetArrayFromImage(moving_resampled)

def sitk_affine_registration(fixed_img_array: np.ndarray, moving_img_array: np.ndarray, mask: bool = False, transformation: str = 'rigid', metric: str = 'mi'):
    '''
    Register two images using an affine transformation.

    Parameters:
        fixed_img_array (np.ndarray): The fixed image.
        moving_img_array (np.ndarray): The moving image.
        mask (bool): Whether the image is a mask. Changes the interpolation method.
        transformation (str): The transformation to use. Options are 'rigid' and 'affine'.
        metric (str): The metric to use. Options are 'mi' and 'pcc'.


    Returns:
        registration_method: sitk.ImageRegistrationMethod: The registration method.
        sitk.Transform: The transformation.
        list: The metric values at each iteration.
    '''

    if mask:
        interp = sitk.sitkNearestNeighbor
    else:
        interp = sitk.sitkLinear

    if transformation == 'rigid':
        sitk_transformation = sitk.Euler3DTransform()
    elif transformation == 'affine':
        sitk_transformation = sitk.AffineTransform(3)

    initial_transform = sitk.CenteredTransformInitializer(
        sitk.GetImageFromArray(fixed_img_array),
        sitk.GetImageFromArray(moving_img_array),
        sitk_transformation,
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    moving_resampled = sitk.Resample(
        sitk.GetImageFromArray(moving_img_array),
        sitk.GetImageFromArray(fixed_img_array), # reference 
        initial_transform,
        interp,
        0.0,
        sitk.GetImageFromArray(moving_img_array).GetPixelID())

    registration_method = sitk.ImageRegistrationMethod()

    # Store metric values at each iteration
    metric_values = []

    def update_metric():
        """Callback function to store metric values at each iteration."""
        metric_values.append(registration_method.GetMetricValue())

    # Similarity metric settings.
    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    if metric == 'mi':
        registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=80)

    elif metric == 'pcc':
        registration_method.SetMetricAsCorrelation()
    
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    # Interpolator settings.
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    # registration_method.SetOptimizerAsGradientDescentLineSearch( #SetOptimizerAsGradientDescent 
    #     learningRate=1,
    #     numberOfIterations=100
    # )
    registration_method.SetOptimizerAsLBFGS2(numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])

    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
# 
    # # Don't optimize in-place, we would possibly like to run this cell multiple times.
    if transformation == 'rigid':
        final_transform = sitk.Euler3DTransform(initial_transform)
    elif transformation == 'affine':
        final_transform = sitk.AffineTransform(initial_transform)

    registration_method.SetInitialTransform(final_transform) # , inPlace=False

    registration_method.AddCommand(sitk.sitkIterationEvent, update_metric)

    final_transform = registration_method.Execute(sitk.GetImageFromArray(fixed_img_array), moving_resampled)
    
    # final_transform = final_transform.GetBackTransform()

    return registration_method, final_transform, metric_values


def dipy_registration(fixed_img: np.ndarray, moving_img: np.ndarray, transformation: str = 'rigid'):
    '''Register the images in the stack to the reference image.

    Parameters
    ----------
    fixed_img : nibabel.nifti1.Nifti1Image, fixed image
    moving_img : nibabel.nifti1.Nifti1Image, moving image
    transformation : str, type of transformation to apply. Options are 'rigid' and 'affine'.

    Returns
    -------
    registered_img: list, list of registered nibabel.nifti1.Nifti1Image
    reg_affine: numpy.ndarray, affine transformation matrix
    fopt: float, final cost function value
    '''
    
    pipeline = ["center_of_mass", "translation"] # "center_of_mass", "translation", "rigid", 
    pipeline.append("rigid") #we always want to apply rigid registration
    if transformation == 'affine':
        pipeline.append("affine")
    level_iters = [10000, 1000, 100] # 10000 at the coarsest resolution, 1000 at the medium resolution and 100 at the finest.
    sigmas = [3.0, 1.0, 0.0] # smoothing gaussian kernel 
    factors = [4, 2, 1]
    nbins = 80
    
    # Register the image
    registered_f_label_array, reg_affine, _, fopt = affine_registration(
        moving_img,
        fixed_img,
        nbins=nbins,
        metric="MI",
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
        ret_metric=True, 
        sampling_proportion=0.1
    )
    return registered_f_label_array, reg_affine, fopt 

def apply_dipy_affine_transformation(moving_img, fixed_img, reg_affine):
    """
    Apply the affine transformation to the moving image to align it with the fixed image.

    Parameters
    ----------
    moving_img : nibabel.nifti1.Nifti1Image
        The moving image to be transformed.
    fixed_img : nibabel.nifti1.Nifti1Image
        The fixed image used as a reference.
    reg_affine : numpy.ndarray
        The affine transformation matrix returned by affine_registration.

    Returns
    -------
    aligned_img : nibabel.nifti1.Nifti1Image
        The aligned moving image.
    """
    # Get the data from the moving image
    moving_data = moving_img.get_fdata()

    # Create an AffineMap object
    affine_map = AffineMap(reg_affine,
                           fixed_img.shape, fixed_img.affine,
                           moving_img.shape, moving_img.affine)

    # Apply the affine transformation
    aligned_data = affine_map.transform(moving_data)

    # Create a new NIfTI image with the aligned data
    aligned_img = nib.Nifti1Image(aligned_data, fixed_img.affine, fixed_img.header)

    return aligned_img

def apply_3D_transform(moving_3D_array: np.ndarray, transform: sitk.Transform, mask: bool = False) -> np.ndarray: 
    ''' Apply a 3D transformation to an image.
    
    Parameters:
        moving_3D_array (np.ndarray): The image to transform.
        transform (sitk.Transform): The transformation to apply.
        mask (bool): Whether the image is a mask. Changes the interpolation method.
        
    Returns:
        np.ndarray: The transformed image.
    '''

    if mask: 
        interp = sitk.sitkNearestNeighbor
    else:
        interp = sitk.sitkLinear

    moving_image = sitk.GetImageFromArray(moving_3D_array)
    resampled_image = sitk.Resample(moving_image, transform, interp, 0.0, moving_image.GetPixelID())

    return sitk.GetArrayFromImage(resampled_image)

def apply_3D_transform2(moving_3D_array: np.ndarray, fixed_3D_array: np.ndarray, transform: sitk.Transform, mask: bool = False) -> np.ndarray:
    ''' Apply a 3D transformation to an image.
    
    Parameters:
        moving_3D_array (np.ndarray): The image to transform.
        fixed_3D_array (np.ndarray): The reference image.
        transform (sitk.Transform): The transformation to apply.
        mask (bool): Whether the image is a mask. Changes the interpolation method.
        
    Returns:
        np.ndarray: The transformed image.
    '''

    moving_image = sitk.GetImageFromArray(moving_3D_array)
    fixed_image = sitk.GetImageFromArray(fixed_3D_array)
    if mask: 
        interp = sitk.sitkNearestNeighbor
    else:
        interp = sitk.sitkLinear

    # print(transform.GetTranslation())
    # transform.Translate((0, 0, 8))
    # print(transform.GetTranslation())

    moving_resampled = sitk.Resample(
    moving_image,
    fixed_image,
    transform,
    interp,
    0.0,
    moving_image.GetPixelID(),
    )

    return sitk.GetArrayFromImage(moving_resampled)

def match_histograms(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ''' Match the histogram of an image to a reference image.
    
    Parameters:
        image (np.ndarray): The image to match.
        reference (np.ndarray): The reference image.
        
    Returns:
        np.ndarray: The matched image.
    '''
    
    image = sitk.GetImageFromArray(image)
    reference = sitk.GetImageFromArray(reference)
    
    matched = sitk.HistogramMatching(image, reference)
    
    return sitk.GetArrayFromImage(matched)

def resize_image(image: sitk.Image, new_size: tuple, interpolator=sitk.sitkLinear) -> sitk.Image:
    """ 
    Resizes a SimpleITK image to the specified new_size.

    Parameters:
        image (sitk.Image): Input SimpleITK image.
        new_size (tuple): Desired output size (x, y, z).
        interpolator: Interpolation method (default: sitk.sitkLinear).

    Returns:
        sitk.Image: Resized image.
    """
    original_size = image.GetSize()  # Get original image size
    original_spacing = image.GetSpacing()  # Get original spacing

    # Compute new spacing to preserve aspect ratio
    new_spacing = [
        (original_size[i] * original_spacing[i]) / new_size[i]
        for i in range(3)
    ]

    # Resample the image
    resampled_image = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),  # Identity transform
        interpolator,  # Linear interpolation (default)
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,  # Default pixel value (background)
        image.GetPixelID()
    )

    return resampled_image

def register_images(simu_path: str, f_path: str, simu_gtv_path: str, f_gtv_path: str, output_path: str, normalization: str = 'zscore', transformation: str = 'rigid', metric: str = 'mi') -> tuple:
    ''' Register the fraction image to the simulation image. Evaluate the registration using the Dice computed on GTV masks. 
    
    Parameters:
        simu_path (str): The path to the simulation image.  
        f_path (str): The path to the fraction image.
        simu_gtv_path (str): The path to the GTV of the simulation image.
        f_gtv_path (str): The path to the GTV of the fraction image.
        output_path (str): The path to save the registered fraction image.
        normalization (str): The normalization method to use. Options are 'zscore', 'histogram' and None.
        transformation (str): The transformation to use. Options are 'rigid' and 'affine'.
        metric (str): The metric to use. Options are 'mi' and 'pcc'.

    Returns:
        float: The Dice score before registration.
        float: The Dice score after registration.
    '''
 
    # load simu 
    simu = sitk.ReadImage(simu_path)
    simu = sitk.Cast(simu, sitk.sitkFloat32)
    simu = sitk.GetArrayFromImage(simu)
    
    # load image F5
    fraction = sitk.ReadImage(f_path)
    fraction = sitk.Cast(fraction, sitk.sitkFloat32)
    fraction = sitk.GetArrayFromImage(fraction)

    # load GTV simu
    gtv_simu = sitk.ReadImage(simu_gtv_path)
    gtv_simu = sitk.GetArrayFromImage(gtv_simu)

    # load GTV F5
    gtv_f = sitk.ReadImage(f_gtv_path)
    gtv_f = sitk.GetArrayFromImage(gtv_f)

    if ('Patient20' not in simu_path) or ('Patient32' not in simu_path):
        simu = np.transpose(simu, (0, 2, 1)) # invert x and y 
        gtv_simu = np.transpose(gtv_simu, (0, 2, 1)) # invert x and y
        fraction = np.transpose(fraction, (0, 2, 1)) # invert x and y
        gtv_f = np.transpose(gtv_f, (0, 2, 1)) # invert x and y

    if simu.shape != fraction.shape:
        print("Images have different shapes.")
        return None, None 

    # normalize images
    if normalization == 'zscore':
        simu = (simu - np.mean(simu)) / np.std(simu)
        fraction = (fraction - np.mean(fraction)) / np.std(fraction)
    elif normalization == 'histogram':
        fraction = match_histograms(fraction, simu)
    else:
        pass

    dice_before = compute_dice(gtv_simu, gtv_f)

    dice_after = dice_before
    run_counter = 0
    while dice_after <= dice_before:
        registration_method, T, metric_values = sitk_affine_registration(simu, fraction, mask=False, transformation=transformation, metric=metric) # register fraction image to simu (finer one)
        registered_fraction = apply_3D_transform2(fraction, simu, T, mask=False)
        registered_f_gtv = apply_3D_transform2(gtv_f, gtv_simu, T, mask=True)
        dice_after = compute_dice(gtv_simu, registered_f_gtv)
        run_counter += 1
        if run_counter > 1:
            print(f"Run {run_counter}, Dice before registration: {dice_before}, Dice after registration: {dice_after}")
        if run_counter == 5:
            break

    # sauvegarder les images
    sitk.WriteImage(sitk.GetImageFromArray(registered_fraction), output_path)
 
    # save well oriented simu image
    simu = sitk.GetImageFromArray(simu)
    simu_path = simu_path.replace('.nii', '_oriented.nii')
    sitk.WriteImage(simu, simu_path)

    return dice_before, dice_after

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float: 
    return np.mean((y_true - y_pred)**2)

def compute_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float: 
    intersection = np.sum(y_true * y_pred)
    return 2 * intersection / (np.sum(y_true) + np.sum(y_pred))


# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []


# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()


# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    
    # Plot the similarity metric values
    plt.plot(metric_values, "r")
    plt.plot(
        multires_iterations,
        [metric_values[index] for index in multires_iterations],
        "b*",
    )
    plt.xlabel("Iteration Number", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.show()


# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# metric_values list.
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))