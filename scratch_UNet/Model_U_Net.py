import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import img_as_ubyte
from help_functions import label_to_torch
from Prepare_process import *
import torch.nn as nn
import torch.nn.functional as F


###--------------------Main U_Net Structure-------------------------------------------###
'''
To construct the U-Net structure including 
'contracting' steps and 'expansion' steps
'''

class Unet(nn.Module):
    
    def __init__(self, layers, channel_sizes):
        super(Unet, self).__init__()
        
        '''
        We could investigate different number of layers of the architecture,
        mainly including 3,4,5 layers(refer to downward steps)
        '''
        self.layers = layers # to get the model layers
        
        channel_input = 3 # input channel
        channel_l1 = channel_sizes['l1_channel'] # for every layer the output channel number
        channel_l2 = channel_sizes['l2_channel']
        channel_l3 = channel_sizes['l3_channel']
        channel_l4 = channel_sizes['l4_channel']
        channel_l5 = channel_sizes['l5_channel']
        channel_l6 = channel_sizes['l6_channel']
        
        # the first three layers are fixed
        self.downlayer_1 = Contracting(channel_input, channel_l1) 
        self.downlayer_2 = Contracting(channel_l1, channel_l2, dropout = True)
        self.downlayer_3 = Contracting(channel_l2, channel_l3, dropout = True)
        
        if layers == 3:
            self.center = Expansion(channel_l3, channel_l4, channel_l3)
        
        if layers == 4:  #adding one more contractiong layer and expansion layer
            self.downlayer_4 = Contracting(channel_l3, channel_l4, dropout = True)
            
            self.center = Expansion(channel_l4, channel_l5, channel_l4)
            
            self.uplayer_4 = Expansion(channel_l5, channel_l4, channel_l3)
        
        if layers == 5:  #adding two more contractiong layers and expansion layers
            self.downlayer_4 = Contracting(channel_l3, channel_l4, dropout = True)
            self.downlayer_5 = Contracting(channel_l4, channel_l5, dropout = True)
            
            self.center = Expansion(channel_l5, channel_l6, channel_l5)
            
            self.uplayer_5 = Expansion(channel_l6, channel_l5, channel_l4)
            self.uplayer_4 = Expansion(channel_l5, channel_l4, channel_l3)
        
        self.uplayer_3 = Expansion(channel_l4, channel_l3, channel_l2)
        self.uplayer_2 = Expansion(channel_l3, channel_l2, channel_l1)
        #the last layer is to output the same size of input and only one channel (predicted-gt)
        self.uplayer_1 = Expansion(channel_l2, channel_l1, 1, procedure='output')
        
    def forward(self, x):
        down_out1, copy1 = self.downlayer_1(x) # copy for later concat in expansion processing
        down_out2, copy2 = self.downlayer_2(down_out1)
        down_out3, copy3 = self.downlayer_3(down_out2)
        
        if self.layers == 3:
            center = self.center(down_out3)
            up_out3 = self.uplayer_3(torch.cat([center, copy3], axis=1))
            up_out2 = self.uplayer_2(torch.cat([up_out3, copy2], axis=1))
        
        elif self.layers == 4:
            down_out4, copy4 = self.downlayer_4(down_out3)
            center = self.center(down_out4)
            up_out4 = self.uplayer_4(torch.cat([center, copy4], axis=1))
            up_out3 = self.uplayer_3(torch.cat([up_out4, copy3], axis=1))
            up_out2 = self.uplayer_2(torch.cat([up_out3, copy2], axis=1))
        
        else:
            down_out4, copy4 = self.downlayer_4(down_out3)
            down_out5, copy5 = self.downlayer_5(down_out4)
            center = self.center(down_out5)
            up_out5 = self.uplayer_5(torch.cat([center, copy5], axis=1))
            up_out4 = self.uplayer_4(torch.cat([up_out5, copy4], axis=1))
            up_out3 = self.uplayer_3(torch.cat([up_out4, copy3], axis=1))
            up_out2 = self.uplayer_2(torch.cat([up_out3, copy2], axis=1))
          
        up_layer1 = self.uplayer_1(torch.cat([up_out2, copy1], axis=1))
        sigm = nn.Sigmoid()
        
        return sigm(up_layer1) # to transform output into [0,1]
    
    
    
###--------------------Contracting Process (downstream)-------------------------------------------###
class Contracting(nn.Module):
    '''
    This is the basic structure for downward step --- 'contracting'
    '''
    
    def __init__(self, input_channel, output_channel, dropout = False):
        super(Contracting, self).__init__()
        
        self.batchnorm1 = nn.BatchNorm2d(input_channel)  # to normalize each batch
        self.batchnorm2 = nn.BatchNorm2d(output_channel)
        
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
        
        self.maxpooling = nn.MaxPool2d(kernel_size = 2, stride = 2) 
        # to pack the dimention in half --- 'downward step'
        
        self.dropout = dropout
        if self.dropout:
            self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        '''
        the process is: normalize the data --> conv1 --> activation 
        --> normalize --> conv2 --> activation --> maxpooling
        And save the output before maxpooling in order to concat with upstreams later
        '''
        x = self.relu(self.conv1(self.batchnorm1(x)))
        x = self.relu(self.conv2(self.batchnorm2(x)))
        
        copy = x.clone()
        
        if self.dropout:
            x = self.dropout(x)
        
        x = self.maxpooling(x)
            
        return x, copy         

    
###--------------------Expansion Process (upstream)-------------------------------------------###

class Expansion(nn.Module):
    '''
    As the same way, we could construct the upward step --- 'expansion'
    '''
    
    def __init__(self, input_channel, transition_channel, output_channel, procedure='middle'): 
        '''
        Because the upward step will halve the channel, 
        there exists three different channel numbers
        'procedure' represents the step is in the middle upward step or final output step 
        '''
        super(Expansion, self).__init__()
        
        self.batchnorm1 = nn.BatchNorm2d(input_channel)  # to normalize each batch
        self.batchnorm2 = nn.BatchNorm2d(transition_channel)
        
        self.conv1 = nn.Conv2d(input_channel, transition_channel, kernel_size = 3, padding =1)
        self.conv2 = nn.Conv2d(transition_channel, transition_channel, kernel_size = 3, padding =1)
        self.relu = nn.ReLU()
        
        '''
        Since the last step has output layer, 
        the transformation is different upward step.
        Hence it is better to seperate both situations.
        '''
        self.is_final = False   #default is not at last output layer
        if procedure == 'middle':
            self.upconv = nn.ConvTranspose2d(transition_channel, output_channel, kernel_size=2, stride=2)
            # upward step
        else:
            self.is_final = True
            self.out = nn.Conv2d(transition_channel, 1, kernel_size=1)
            # the last output layer: the output channel is 1 and no slide window
        
    def forward(self, x):
        '''
        the process is: normalize the data --> conv1 --> activation 
        --> normalize --> conv2 --> activation --> convtranspose to 
        halve the channel(if it is the last output step, the last 
        step is one conv process)
        
        '''
        x = self.relu(self.conv1(self.batchnorm1(x)))
        x = self.relu(self.conv2(self.batchnorm2(x)))
        
        if not self.is_final: # this is middle upward step
            return self.upconv(x)
        else:                 # this is the last output step
            return self.out(x)

        
        
###--------------------U_Net model default configuration-------------------------------------------###

def Unet_setting(layers):
    """
    Generate the Unet channel configuration.
    Only accepting 3/4/5 layers configuration
    """
    
    channel_sizes = {
        'l1_channel': 16,
        'l2_channel': 32,
        'l3_channel': 64,
        'l4_channel': 128,
        'l5_channel': 256,
        'l6_channel': 512
        }
    
        
    return (layers, channel_sizes)
