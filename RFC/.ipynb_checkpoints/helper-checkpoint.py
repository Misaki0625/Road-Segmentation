%matplotlib inline
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from PIL import Image
from skimage import feature
from skimage import color
from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
import scipy
import os,sys

'''
To build helper functions to run the random forest classifier
'''

def patch_set(im, w, h):
    patches = []
    width = im.shape[0]
    height = im.shape[1]
    
    for i in range(0,height,h): # loop across height in steps equal to your height patch size
        for j in range(0,width,w):  # loop across width in steps equal to your width patch size
            if (len(im.shape) < 3)==1: # judge whether the image is 2d or 3d (from images or groundtruth files)
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            patches.append(im_patch)
    return patches

def build_features(patch):
    feat_m = np.mean(patch, axis=(0,1))  # calculate the mean
    feat_v = np.var(patch, axis=(0,1))  # calculate the variance
    feat = np.append(feat_m, feat_v)
    
    # compute the hough transform to get new feature
    gray_patch = color.rgb2gray(patch)   # convert image to grayscale
    edges_patch = feature.canny(gray_patch) # use canny edge detector
    h, theta, d = hough_line(edges_patch) # compute hough transform of edges image
    h_std = np.std(np.std(h,axis = 0))
    feat = np.append(feat, h_std)
    
    return feat 

def gt_classify(patch, threshold):
    p = patch.mean()
    if p > threshold:
        return 1
    else: return 0
    
def feature_extract(patch_size, threshold):
    im_patches = [patch_set(image[i], patch_size, patch_size) for i in np.arange(len(image))]
    gt_patches = [patch_set(gt[i], patch_size, patch_size) for i in np.arange(len(gt))]
    
    im_patches = np.asarray([im_patches[i][j] for i in range(len(im_patches)) for j in range(len(im_patches[i]))])
    gt_patches = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    X = np.asarray([build_features(im_patches[i]) for i in range(len(im_patches))])
    y = np.asarray([gt_classify(gt_patches[i],threshold) for i in range(len(gt_patches))])
    
    # Normalize the features
    X_max = np.max(X,axis = 0)
    X = X/X_max
    
    return X,y

def train_test_set(X,y,test_size,random_state,report = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if report:
        print('feature number: '+ str('7'))
        print('Total feature extraction sample: {}'.format(X.shape[0]))
        print('Training size: {}'.format(len(y_train)))
        print('Testing size: {}'.format(len(y_test)))
    
    return X_train,X_test,y_train,y_test

def label_to_im(width, height, w, h, labels):
    """ Convert array of labels to an image
    w: patch width
    h: patch size"""
    im = np.zeros([width, height])
    idx = 0
    for i in range(0,height,h):
        for j in range(0,width,w):
            im[j:j+w, i:i+h] = labels[idx] # fill all pixels in a given patch with the same label
            idx += 1
    return im

def prediction_report(X_test,y_test,classifier,n_test,patch_size,plot=True):
    """Test classifier on training data not used for training
    You obtain true positive rate, F1 metric and visualization"""
    
    # Predict on the training set
    pre_y = classifier.predict(X_test) # gives 0 / 1 labels for all the training patches

    # Get non-zeros in prediction and grountruth arrays
    pre_yn = np.nonzero(pre_y)[0] # get index of predicted foreground patches
    yn = np.nonzero(y_test)[0] # get index of real foreground patches

    # Get metrics on the accuracy of the model
    TPR = len(list(set(yn) & set(pre_yn))) / float(len(pre_y)) # calculate the true positive rate
    print('Random Forest Classifer')
    print('True positive rate = {:.4f}'.format(TPR))
    f1 = metrics.f1_score(y_test, pre_y, average='binary')  # Calculate F1 metric of our classification 
    print('F1 metric = {:.4f}'.format(f1))
    
    if plot:
        #Make plots for a single image
        img_idx = n_test
        patches_idx = patch_set(image[img_idx], patch_size, patch_size)
        np.asarray([patches_idx[i][j] for i in range(len(patches_idx)) for j in range(len(patches_idx[i]))])
        X_idx= np.asarray([build_features(patches_idx[i]) for i in range(len(patches_idx))])
        prey_idx = classifier.predict(X_idx)
        w = gt[img_idx].shape[0] # image width
        h = gt[img_idx].shape[1] # image height
        pre_im = label_to_im(w, h, patch_size, patch_size, prey_idx)
    
        plt.figure(figsize = [10,5])
        ax = plt.subplot(131)
        plt.imshow(image[img_idx])
        plt.title('Original')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax = plt.subplot(132)
        plt.imshow(gt[img_idx],cmap='Greys_r')
        plt.title('Groundtruth')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax = plt.subplot(133)
        plt.imshow(pre_im,cmap='Greys_r')
        plt.title('Predicted')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)