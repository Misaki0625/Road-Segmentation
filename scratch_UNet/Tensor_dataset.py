import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import img_as_ubyte
from help_functions import label_to_torch
from Prepare_process import *




'''
to construct the torch type training dataset
'''

class Patch_dataset(Dataset):
    
    def __init__(self, imag_path, label_path, imag_indices, patch_size, overlap, step):
        self.patch_feat, self.patch_label = prepare_trainset(imag_path, label_path, imag_indices, patch_size, overlap, step)
        
    def __getitem__(self,index): # design get function according to index
        patch_feat = torch.from_numpy(np.transpose(self.patch_feat[index],(2,0,1))) #each data patch contain 3 channels. Each channel is in patch_size x patch_size
        patch_label = label_to_torch(self.patch_label[index])
        
        return patch_feat, patch_label
    
    def __len__(self):
        return(len(self.patch_feat))
    

'''
to construct the torch type testing dataset
'''

class Test_dataset(Dataset):
    
    def __init__(self, test_path, test_indices):
        self.feat = prepare_testdata(test_path, test_indices)
        
    def __getitem__(self, index): # design get function according to index
        feat = torch.from_numpy(np.transpose(self.feat[index],(2,0,1))) #each data patch contain 3 channels. Each channel is in patch_size x patch_size
        return feat
    
    def __len__(self):
        return(len(self.feat))

    
'''
Patch_augment_dataset is used to couple with prepare_augment_trainset function which integrate Flip and Rotation process
'''

class Patch_augment_dataset(Dataset):
    
    def __init__(self, imag_path, label_path, imag_indices, patch_size, overlap, step, Flip, rotation_angle):
        self.patch_feat, self.patch_label = prepare_augment_trainset(imag_path, label_path, imag_indices, patch_size, overlap, step, Flip, rotation_angle)
        
    def __getitem__(self,index): # design get function according to index
        patch_feat = torch.from_numpy(np.transpose(self.patch_feat[index],(2,0,1))) #each data patch contain 3 channels. Each channel is in patch_size x patch_size
        patch_label = label_to_torch(self.patch_label[index])
        
        return patch_feat, patch_label
    
    def __len__(self):
        return(len(self.patch_feat))
        