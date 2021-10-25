import numpy as np


'''
Without overlapping with neighboring patches
Ps: if the size of imag is not multiplier of the patch size, 
we use mirroring method to extend the border with one patch 
size according to ref[1]
'''

def get_patch(imag, patch_size):
    patch_set = []
    height, width = imag.shape[0], imag.shape[1]
    
    imag_mir = extend_boder(imag, patch_size)
    
    is_label = len(imag.shape) == 2 # since the groundtruth data set only does not have chanel dimension
    
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if is_label:
                patch = imag[j:j+patch_size, i:i+patch_size]
            else:
                patch = imag[j:j+patch_size, i:i+patch_size, :]
            
            patch_set.append(patch)
    
    return patch_set


'''
With overlapping for mutual patches
Ps: it's better to use multiplier factors. For example: the imag size is 400x400, and 
the patch size is 40x40, then the step is better to set as 6/10/20/40/60.. 
which (400-40)=360 is a muptiplier of the step in order to get all the pixels information.
'''
from skimage.util import view_as_windows

def get_patch_ovelap(imag, patch_size, step):
    
    is_label = len(imag.shape) == 2
    
    if is_label:
        window_size = (patch_size, patch_size) # gt does not have channel dimention
    else:
        window_size = (patch_size, patch_size, 3) # image with RGB channel
        
    slide_windows = view_as_windows(imag, window_size, step) # slide window to pick up patches with overlapping
    
    patches = []
    for i in range(0, slide_windows.shape[0], 1):
        for j in range(0, slide_windows.shape[1], 1):
            if is_label:
                patches.append(slide_windows[i][j])
            else:
                patches.append(slide_windows[i][j][0])
    
    return patches    