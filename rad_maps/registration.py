'''Functions to register fractions to simu image. '''

import numpy as np 
import SimpleITK as sitk 
import matplotlib.pyplot as plt

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

def affine_registration(fixed_img_array: np.ndarray, moving_img_array: np.ndarray, mask: bool = False):
    '''
    Register two images using an affine transformation.

    Parameters:
        fixed_img_array (np.ndarray): The fixed image.
        moving_img_array (np.ndarray): The moving image.

    Returns:
        registration_method: sitk.ImageRegistrationMethod: The registration method.
        sitk.Transform: The transformation.
        list: The metric values at each iteration.
    '''

    if mask:
        interp = sitk.sitkNearestNeighbor
    else:
        interp = sitk.sitkLinear

    initial_transform = sitk.CenteredTransformInitializer(
        sitk.GetImageFromArray(fixed_img_array),
        sitk.GetImageFromArray(moving_img_array),
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
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
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.5)

    # Interpolator settings.
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescentLineSearch( #SetOptimizerAsGradientDescent 
        learningRate=1,
        numberOfIterations=100
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
# 
    # # Don't optimize in-place, we would possibly like to run this cell multiple times.
    final_transform = sitk.Euler3DTransform(initial_transform)
    registration_method.SetInitialTransform(final_transform) # , inPlace=False

    registration_method.AddCommand(sitk.sitkIterationEvent, update_metric)

    final_transform = registration_method.Execute(sitk.GetImageFromArray(fixed_img_array), moving_resampled)
    
    # final_transform = final_transform.GetBackTransform()

    return registration_method, final_transform, metric_values

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

def match_histograms(image: np.ndarray, reference: np.ndarray):
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

def resize_image(image, new_size, interpolator=sitk.sitkLinear):
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

def register_gtv(simu_path: str, f_path: str, gtv_simu_path: str, gtv_f_path: str, output_path: str, normalization: str = 'zscore'):
    ''' Register the GTVs of a fraction image to the simulation image.
    
    Parameters:
        simu_path (str): The path to the simulation image.  
        f_path (str): The path to the fraction image.
        gtv_simu_path (str): The path to the GTV of the simulation image.
        gtv_f_path (str): The path to the GTV of the fraction image.
        output_path (str): The path to save the registered GTV.
        normalization (str): The normalization method to use. Options are 'zscore', 'histogram' and None.

    Returns:
        float: The MSE before registration.
        float: The MSE after registration.
    '''
    
    # charger simu 
    simu = sitk.ReadImage(simu_path)
    simu = sitk.GetArrayFromImage(simu)

    # charger image F5
    fraction = sitk.ReadImage(f_path)
    fraction = sitk.GetArrayFromImage(fraction)

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

    registration_method, T, metric_values = affine_registration(simu, fraction)

    # COMPARE GTVs 
    gtv_simu = sitk.ReadImage(gtv_simu_path) # charger gtv simu 
    gtv_simu = sitk.GetArrayFromImage(gtv_simu)
    gtv_fraction = sitk.ReadImage(gtv_f_path) # charger gtv F5 
    gtv_fraction = sitk.GetArrayFromImage(gtv_fraction)

    mse_before = compute_mse(gtv_simu, gtv_fraction)
    # print("MSE before registration: ", mse_before)

    registered_gtv_fraction = apply_3D_transform2(gtv_fraction, gtv_simu, T, mask=True)

    # comparer gtv simu et F5 
    mse_after = compute_mse(gtv_simu, registered_gtv_fraction)
    # print("MSE after registration:", mse_after)

    # sauvegarder les images
    transformed_img = sitk.GetImageFromArray(registered_gtv_fraction)
    transformed_img = sitk.TransformGeometry(transformed_img, T)
# 
    sitk.WriteImage(transformed_img, output_path)

    return mse_before, mse_after

def compute_mse(y_true, y_pred): 
    return np.mean((y_true - y_pred)**2)

def compute_dice(y_true, y_pred):
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