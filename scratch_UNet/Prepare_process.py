import numpy as np
from load_data import *
from get_patch import *
from Feature_Augmentaion import *

'''
To get all the mini-patches for trainning and testing
'''

def prepare_trainset(imag_path, label_path, imag_indices, patch_size, overlap, step):
    images = load_data_set(imag_path, imag_indices)
    labels = load_data_set(label_path, imag_indices)
    
    if overlap:
        image_patches = [patch for im in images for patch in get_patch_ovelap(im, patch_size, step)]
        label_patches = [patch for label in labels for patch in get_patch_ovelap(label, patch_size, step)]
    else:
        image_patches = [patch for im in images for patch in get_patch(im, patch_size)]
        label_patches = [patch for label in labels for patch in get_patch(label, patch_size)]
        
    return image_patches, label_patches

def prepare_testdata(test_path, test_indices):
    test_images = load_data_set(test_path, test_indices, test=True)
    return test_images


'''
prepare_augment_trainset is used to integrate flip and rotation process to get the training dataset
'''
def prepare_augment_trainset(imag_path, label_path, imag_indices, patch_size, overlap, step, Flip=False, rotation_angle=False):
    images = load_data_set(imag_path, imag_indices)
    labels = load_data_set(label_path, imag_indices)
    
    if Flip:
        flipped_images = [f for im in images for f in flip(im)]
        flipped_labels = [f for label in labels for f in flip(label)]
        
        images.extend(flipped_images)
        labels.extend(flipped_labels)
    
    if rotation_angle:
        rotated_images = [rotation(im, rotation_angle) for im in images]
        rotated_labels = [rotation(label, rotation_angle) for label in labels]
        
        images.extend(rotated_images)
        labels.extend(rotated_labels)
    
    if overlap:
        image_patches = [patch for im in images for patch in get_patch_ovelap(im, patch_size, step)]
        label_patches = [patch for label in labels for patch in get_patch_ovelap(label, patch_size, step)]
    if not overlap:
        image_patches = [patch for im in images for patch in get_patch(im, patch_size)]
        label_patches = [patch for label in labels for patch in get_patch(label, patch_size)]

    if not rotation_angle:
        return image_patches, label_patches

    if rotation_angle:
        image_patches_rotated = [image_patches[i] for i in range(len(image_patches)) if not remove_shadow(image_patches[i])]
        label_patches_ratated = [label_patches[i] for i in range(len(image_patches)) if not remove_shadow(image_patches[i])]
    
        return image_patches_rotated, label_patches_ratated