'''Script to compare feature map parameters between patients.'''

import os 
import get_map as gm 

def main(): 
    # COMPUTE FEATURE MAPS
    params = 'params.yaml' # TODO: modify names of features computed

    fractions = ['ttt_1', 'ttt_3']

    patients = os.listdir('Data/')
    # patients.remove('BFcorr')
    # patients.remove('Data GIE')
    # patients.remove('Masks_WholePat')
    # patients.remove('NFcorr')
    patients_to_remove = ['Patient' + str(n) for n in [57, 32, 74, 82, 84, 85, 56, 63]]
    patients_filtered = [p for p in patients if patients not in patients_to_remove]

    patients_filtered = patients_filtered[0:2] # test TODO: remove this line when testing is done

    for f in fractions:
        for p in patients_filtered:
            image_path = 'Data/' + p + '/img_dir/' + p + '_mridian_' + f + '.nii'
            mask_path = 'Data/' + p + '/mask_dir/' + p + '_mridian_' + f + '_mask.nii' 
            gm.generate_feature_map(image_path, mask_path, params, 'Data/' + p + '/rad_maps/' + f + '/')

    # TODO: for each radiomics feature map, compute the intensity parameters, store it in a csv with patient ID as index 
    # TODO: compare statistics between patients using scipy or pengouin 

if __name__ == '__main__':
    main()