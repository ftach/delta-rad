'''Quick script to orient all the images in the right direction. '''

# For all the nifti images in 
# we orient them in the right direction and save them in /home/tachennf/Documents/delta-rad/data/ICM_0.35T/clean_data/


import os 
import SimpleITK as sitk
import numpy as np

def orient_img(img_path):
    '''Orient the image in the right direction. 
    
    Parameter: 
    img_path: str, path to the image to orient
    
    Return:
    img: sitk image, oriented image
    '''
    img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(img)
    img_array = np.transpose(img_array, (0, 2, 1))# orient img in the right direction
    img = sitk.GetImageFromArray(img_array)
    return img 

def main(original_data_path: str, clean_data_path: str): 
    patient_list = [p for p in os.listdir(original_data_path)]
    for p in patient_list:
        # create patient folder 
        if os.path.exists(clean_data_path + p) == False:
            os.makedirs(clean_data_path + p)
        # images 
        if os.path.exists(clean_data_path + p + '/img_dir/') == False:
            os.makedirs(clean_data_path + p + '/img_dir/')
        img_files = [f for f in os.listdir('/home/tachennf/Documents/delta-rad/data/ICM_0.35T/original_data/' + p + '/img_dir/') if f.endswith('.nii')]
        for i in img_files:
            img_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/original_data/' + p + '/img_dir/' + i
            oriented_img = orient_img(img_path)
            orient_img_path = clean_data_path + p + '/img_dir/' + i
            sitk.WriteImage(oriented_img, orient_img_path)
        # mask 
        if os.path.exists(clean_data_path + p + '/mask_dir/') == False:
            os.makedirs(clean_data_path + p + '/mask_dir/')
        mask_files = [f for f in os.listdir('/home/tachennf/Documents/delta-rad/data/ICM_0.35T/original_data/' + p + '/mask_dir/') if f.endswith('.nii')]
        for m in mask_files:
            mask_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/original_data/' + p + '/mask_dir/' + m
            oriented_mask = orient_img(mask_path)
            orient_mask_path = clean_data_path + p + '/mask_dir/' + m
            sitk.WriteImage(oriented_mask, orient_mask_path)
        print(p, 'oriented')
    print('All images and masks oriented')

if __name__ == '__main__':
    
    original_data_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/original_data/'
    clean_data_path = '/home/tachennf/Documents/delta-rad/data/ICM_0.35T/clean_data/'
    main(original_data_path, clean_data_path)

