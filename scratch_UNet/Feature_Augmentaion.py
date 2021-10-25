import cv2
import imutils
import numpy as np

#-------------------Flip process--------------------------#

def flip(x):  # x: dataset; dim: how to flip, 1:vertically; 2:horizontally
    
    hor = cv2.flip(x, 0)
    ver = cv2.flip(x, 1)

    return hor, ver
#--------------------------------------------------------#


#-------------------Rotation process--------------------------#

def rotation(x, angle):
    return imutils.rotate_bound(x, angle)


def remove_shadow(patch):
    """
    Due to the rotation, the image will be enlarged 
    and get more shadow areas(no information). Hence it 
    is better to remove these useless pixels and then
    get useful patches.
    """
    lt = not np.any(patch[0,0])
    rt = not np.any(patch[0,-1])
    lb = not np.any(patch[-1,0])
    rb = not np.any(patch[-1,-1])

    return lt or rt or lb or rb

#--------------------------------------------------------#