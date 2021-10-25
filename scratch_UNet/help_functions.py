import numpy as np
import torch

#---------------------Mirror Function-----------------------------------#
'''
Mirror function used in get_patch(non-overlapping) function 
in case of imag size is not multiplier of patch size.
'''

def extend_boder(imag, length):
    height, width = imag.shape[0], imag.shape[1]
    is_label = len(imag.shape) == 2
    
    '''to extend the right border'''
    if is_label:
        right_flipped = np.fliplr(imag[:, width - length:])
    else:
        right_flipped = np.fliplr(imag[:, width - length:, :])
        
    right_extend = np.concatenate((imag, right_flipped), axis=1)
    
    '''to extend the bottom border'''
    if is_label:
        bottom_extend = np.flipud(right_extend[height - length:, :])
    else:
        bottom_extend = np.flipud(right_extend[height - length:, :, :])
        
    extend = np.concatenate((right_extend, bottom_extend), axis=0)
    
    return extend

#------------------------------------------------------------#


#------------Array type label to binary torch type-----------#
'''
First of all, we should transform labels into binary form
'''

def label_to_torch(label):
    label_treshold = 0.5
     
    label_torch = torch.from_numpy(label).float().unsqueeze(0)
    
    label_torch[label_torch < label_treshold] = 0  # transform into binary range [0,1]
    label_torch[label_torch >= label_treshold] = 1
    
    return label_torch
#------------------------------------------------------------#



