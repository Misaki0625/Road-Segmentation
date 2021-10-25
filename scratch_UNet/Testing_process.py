import torch
import numpy as np
from torch.autograd import Variable


'''
To construct one prediction process
'''

def Test(model, testloader, cuda=False):
    predicted = []
    model.eval()
    
    for data in testloader:
        if cuda:
            data = data.cuda()
        data = Variable(data)
        
        pred = model(data)
        #print(pred)
        pred = np.rint(pred.squeeze().data.cpu().numpy()) # to transform tensor into array
        predicted.append(pred)
    
    return predicted