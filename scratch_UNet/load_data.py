import numpy as np
import os
import matplotlib.image as mpimg


'''
Load imags using mping function and return all data list
'''

'''
Load imags using mping function and return all data list
'''

def load_data_set(path, imag_indices, test=False):
    imgs = []
    print ('Loading {} satlite images...'.format(len(imag_indices)), end='') # for checking
    
    if not test:  #for training feature and groundtruth dataset
        for i in imag_indices:
            imag_name = path + 'satImage_{:03d}.png'.format(i) # the imag name format
            if os.path.isfile(imag_name):
                img = mpimg.imread(imag_name)   # read all imags in the file
                imgs.append(img)
            else:
                print('imag {} does not exists'.format(imag_name)) # for checking wrongly path
    
    else:  # for test dataset
        for i in imag_indices:
            imag_path = path + 'test_{:01d}/'.format(i)
            imag_name = imag_path + 'test_{:01d}.png'.format(i)
            if os.path.isfile(imag_name):
                imag = mpimg.imread(imag_name)
                imgs.append(imag)
            else:
                print('imag {} does not exists'.format(imag_name))
    return imgs