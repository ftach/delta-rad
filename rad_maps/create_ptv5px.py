'''Script to create special masks. 
'''

from preprocessing.create_masks import create_dilated_ptv_mask, create_one_mask

def main(): 
    patients_filtered = ['Patient48', 'Patient76', 'Patient75', 'Patient72', 'Patient59', 'Patient46', 'Patient34', 'Patient36', 'Patient31', 'Patient12', 'Patient20', 'Patient22', 'Patient26', 'Patient39', 'Patient40']

    create_dilated_ptv_mask(patients_filtered)
    # create_one_mask() # create one mask for all patients and fractions

if __name__ == '__main__':
    main()