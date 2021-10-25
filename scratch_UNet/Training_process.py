import torch
import numpy as np
from torch.autograd import Variable

def Training_process(model, epoches, criterion, optimizer, lr_scheduler, dataset, dataloader, patch_size, cuda=False):
    
    print('Starting to train the U_Net model...')
    
    for epoch in range(epoches):
        
        train_loss = []
        #model.train()
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data # to get features and label
            #print( inputs,'\n',labels)
            
            if cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) # if GPU is available
            else:
                inputs, labels = Variable(inputs), Variable(labels) # using CPU
            
            optimizer.zero_grad() 
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
        # to adjust the learning rate according to the loss of each epoch
        lr_scheduler.step(int(np.mean(train_loss)*1000)) # After test the loss is changing at the third decimal so it's better to choose that one as the standard to adjust the learning rate
        

        print('training the No.{} epoch, loss: {:.4f}'.format(epoch+1, np.mean(train_loss)))