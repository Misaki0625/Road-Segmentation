%matplotlib inline
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os,sys
from PIL import Image
from skimage import feature
from skimage import color
from skimage.transform import hough_line, hough_line_peaks,probabilistic_hough_line
import scipy
import imageio
import warnings
warnings.filterwarnings("ignore")

from helper import *
from submission import *

# Load the data
data_folder = 'training/'
img_dir = data_folder + 'images/'
gt_dir = data_folder + 'groundtruth/'
files = os.listdir(data_folder + 'images/')

image = []
gt = []
for i in range(len(files)):
    im = mpimg.imread(img_dir + files[i])
    g = mpimg.imread(gt_dir + files[i])
    image.append(im)
    gt.append(g)

# Try our classifier on the train and test set obtained from the training set
X, y = feature_extract(patch_size=16,threshold=0.25)
X_train, X_test, y_train, y_test = train_test_set(X,y,test_size=0.2,random_state=50,report=True)

RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)
prediction_report(X_test,y_test,classifier=RFC,n_test=82,patch_size=16,plot=True)

# Make the prediction and submission
# Load test images
image_filenames = []
n_test = 50
for i in range(1, n_test+1):
    image_filename = 'test_set_images/test_%d' % i + '/test_%d' % i + '.png'
    image_filenames.append(image_filename)
    
imgs_test = [mpimg.imread(image_filename) for image_filename in image_filenames]
w = imgs_test[0].shape[0]
h = imgs_test[0].shape[1]

# Train the classifier
patch_size = 16
num_trees = 70
tree_depth = 10000
rfc = RandomForestClassifier(n_estimators = num_trees, max_depth = tree_depth)
rfc.fit(X, y)

patches = [patch_set(imgs_test[i], patch_size, patch_size) for i in range(n_test)]
patches = np.asarray([patches[i][j] for i in range(len(patches)) for j in range(len(patches[i]))])
X_test = np.asarray([build_features(patches[i]) for i in range(len(patches))])
X_test_maxs = np.max(X_test,axis = 0)
X_test = X_test/X_test_maxs

# Make prediction
Z_test= rfc.predict(X_test)

# Save submission
from mask_to_submission import *
n_patches_per_image = len(patches)/n_test
for i in range(0,n_test):
    im_labels = Z_test[int(n_patches_per_image*i):int(n_patches_per_image*(i+1))]
    # Transform the patches into a complete black and white image
    predicted_im = label_to_im(w, h, patch_size, patch_size, im_labels)
    j = i + 1
    imageio.imwrite('prediction_RFC/resultImage_' + '%.3d' % j + '.png', predicted_im)
submission_filename = 'RFC_submission' + '_features' + str('7') + '_patch' + str(patch_size) + '_trees' + str(num_trees) + '.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = 'prediction_RFC/resultImage_' + '%.3d' % i + '.png'
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)

print('Results saved for submission.')