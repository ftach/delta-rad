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
        sitk.AffineTransform(3),
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
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    # Interpolator settings.
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.01,
        numberOfIterations=100
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
# 
    # # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    registration_method.AddCommand(sitk.sitkIterationEvent, update_metric)

    final_transform = registration_method.Execute(sitk.GetImageFromArray(fixed_img_array), moving_resampled)
    
    final_transform = final_transform.GetBackTransform()

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
    moving_image = sitk.GetImageFromArray(moving_3D_array)
    fixed_image = sitk.GetImageFromArray(fixed_3D_array)
    if mask: 
        interp = sitk.sitkNearestNeighbor
    else:
        interp = sitk.sitkLinear

    moving_resampled = sitk.Resample(
    moving_image,
    fixed_image,
    transform,
    interp,
    0.0,
    moving_image.GetPixelID(),
    )

    return sitk.GetArrayFromImage(moving_resampled)

def compute_mse(y_true, y_pred): 
    return np.mean((y_true - y_pred)**2)


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