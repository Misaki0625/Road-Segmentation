import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision

from Tensor_dataset import *
from Model_U_Net import *
from Training_process import *
from Testing_process import *
from Submission import *

train_size = 100
test_size = 50

imag_path = '../training/images/'
label_path = '../training/groundtruth/'
test_path = '../test_set_images/'

imag_indices = np.arange(1, train_size + 1)
test_indices = np.arange(1, test_size + 1)

Test_set = Test_dataset(test_path, test_indices)
testloader = DataLoader(Test_set, batch_size = 1, shuffle=False)

patch_size = 80

Train_set = Patch_augment_dataset(imag_path, label_path, imag_indices, patch_size, overlap=True, step=20, Flip=True, rotation_angle=45)
trainloader = DataLoader(Train_set, batch_size = 128, shuffle=True)


layers, channel_sizes = Unet_setting(layers=4)
model = Unet(layers, channel_sizes).cuda()
criterion = torch.nn.BCEWithLogitsLoss().cuda()
optimizer = Adam(model.parameters(), lr=0.001)
epoches = 70
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
Training_process(model, epoches, criterion, optimizer, lr_scheduler, Train_set, trainloader, patch_size, cuda=True)


import pandas as pd       
df = pd.DataFrame(list())        # to create an empty .csv for submission
df.to_csv('submission.csv')

label_size = 608
subm_patch_size = 16
threshold = 0.25


prediction = Test(model, testloader, cuda=True)  # to predict
# to transform pixel predicition into patch approxiamation
patch_labels = [pred_to_patch_approx(pred, label_size, subm_patch_size, threshold) for pred in prediction]  
pred_submission = pixel_to_patch(patch_labels, subm_patch_size)   # to seperate each patch
  
    
submission_path = 'submission.csv'  

submission(submission_path, test_size, label_size, subm_patch_size, pred_submission)  
# write in the submission file