from Testing_process import *
from Submission import *
from Model_U_Net import *
from Tensor_dataset import *
from torch.utils.data import DataLoader

'''
to load the trained model parameters
'''
PATH = 'U_Net_parameters.pt'
layers, channel_sizes = Unet_setting(layers=4)
net = Unet(layers, channel_sizes).cuda()
net.load_state_dict(torch.load(PATH))  # to load the trained model parameters


test_size = 50
test_indices = np.arange(1, test_size + 1)
test_path = '../test_set_images/'
Test_set = Test_dataset(test_path, test_indices)
testloader = DataLoader(Test_set, batch_size = 1, shuffle=False)  # to get the test dataloader


import pandas as pd       
df = pd.DataFrame(list())        # to create an empty .csv for submission
df.to_csv('submission.csv')


label_size = 608
subm_patch_size = 16
threshold = 0.25


prediction = Test(net, testloader, cuda=True)  # to predict
# to transform pixel predicition into patch approxiamation
patch_labels = [pred_to_patch_approx(pred, label_size, subm_patch_size, threshold) for pred in prediction]  
pred_submission = pixel_to_patch(patch_labels, subm_patch_size)   # to seperate each patch
  
    
submission_path = 'submission.csv'  

submission(submission_path, test_size, label_size, subm_patch_size, pred_submission)  
# write in the submission file
