import torch
import numpy as np
from torch.autograd import Variable


def Val_Training_process(model, epoches, criterion, optimizer, lr_scheduler, train_loader, val_loader, cuda=False):
    print('Starting to train the U_Net Model...')
    loss_set = []
    
    for epoch in range(epoches):
        train_loss = []
        for i, data in enumerate(train_loader):
            inputs, labels = data

            if cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        lr_scheduler.step(int(np.mean(train_loss) * 1000))
        loss_set.append(np.mean(train_loss))
        
        print('training the No.{} epoch, loss: {:.4f}'.format(epoch+1, np.mean(train_loss)))

    model.eval()
    val_loss = []
    
    for i, data in enumerate(val_loader):
        val_inputs, val_labels = data

        if cuda:
            val_inputs, val_labels = Variable(val_inputs.cuda()), Variable(val_labels.cuda())
        else: 
            val_inputs, val_labels = Variable(val_inputs), Variable(val_labels)

        pred = model(val_inputs)
        loss = criterion(pred, val_labels)
        val_loss.append(loss.item())
        
    return loss_set, np.mean(val_loss)