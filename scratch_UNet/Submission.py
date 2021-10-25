import torch
import numpy as np
from torch.autograd import Variable


'''
To transform pixel-wise prediction 
into path-wise prediction according 
to threshold approximation (0.25)
'''
def pred_to_patch_approx(labels, label_size, subm_patch_size, threshold):
    '''
    <label_size> means the test image pixel size
    <subm_patch_size> means the submitted patch size
    '''
    patch_labels = np.zeros([label_size, label_size])
    
    for i in range(0, label_size, subm_patch_size):
        for j in range(0, label_size, subm_patch_size):
            label_ave = np.mean(labels[i : i + subm_patch_size, j : j + subm_patch_size])
            if label_ave < threshold:
                patch_labels[i : i + subm_patch_size, j : j + subm_patch_size] = 0
            else:
                patch_labels[i : i + subm_patch_size, j : j + subm_patch_size] = 1
    return patch_labels


'''
To seperated each predicted imag into patchs
'''

def pixel_to_patch(images, subm_patch_size):
    patch_lst = []
    
    for image in images:
        patch = []
        for i in range(0, image.shape[0], subm_patch_size):
            for j in range(0, image.shape[1], subm_patch_size):
                patch.append(image[j : j + subm_patch_size, i : i + subm_patch_size])
        patch_lst.extend(patch)
    
    return patch_lst

'''
To create submission file.csv
'''

def submission(submission_path, test_size, label_size, subm_patch_size, pred_submission):
    with open(submission_path, 'w+') as f:
        f.write('id,prediction\n')
        patch_num = label_size**2 / subm_patch_size**2 # 608^2 / 16^2 = 1444 each image has 1444 patches
        patch_num_onerow = label_size / subm_patch_size # 608 / 16 = 38 each row/column has 38 attached patches
        
        for imag_num in range(1,test_size+1,1): #No.1-No.50 image
            for i in range(0,label_size,subm_patch_size): 
                for j in range(0,label_size,subm_patch_size): # 0 - 608 in path length of 16
                    index = int((imag_num - 1)*patch_num + (i/subm_patch_size)*patch_num_onerow + j/subm_patch_size)
                    f.writelines('{:03d}_{}_{}, {}\n'.format(imag_num,i,j,np.mean(pred_submission[index])))