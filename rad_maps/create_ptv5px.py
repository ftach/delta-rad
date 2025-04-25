'''Script to create special masks. 
'''

from preprocessing.create_masks import create_dilated_ptv_mask, create_one_mask

def main(): 
    # create_dilated_ptv_mask()
    create_one_mask() # create one mask for all patients and fractions

if __name__ == '__main__':
    main()