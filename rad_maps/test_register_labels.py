from registration.register_labels import *

def main(): 
    '''Test the registration of the labels. 
    '''
    # quick_test()
    register_mridian_dataset(transformation='rigid', metric='pcc', mask='gtv')

if __name__ == '__main__':
    main()