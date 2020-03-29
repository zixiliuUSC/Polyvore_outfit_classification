import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp
from matplotlib import pyplot as plt
from utils import Config
from model import fashion_model
from data import get_dataloader
#import os
#os.environ['CUDA_VISIBLE_DEVICES']="0"


def train_model(get_dataloader, model, criterion, optimizer, device, num_epochs, lr_decay,warmup=5):
    dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])
    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc = []
    test_acc = []
    unfreeze_num = 13
    ratio = 0.45
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for input1, input2, labels in tqdm(dataloaders[phase]):
                input1 = input1.to(device)
                input2 = input2.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(input1,input2)
                    _, pred = torch.max(outputs, 1)
                    #loss = criterion(outputs, labels)
                    #print(loss)

                    if (epoch+1) in [2,3,8,17,18,19,22,23]:
                        criterion.reduction = 'none'
                        loss_inst = criterion(outputs,labels)
                        num_inst = outputs.size(0)
                        num_hns = int(ratio * num_inst)
                        _, idxs = loss_inst.topk(num_hns) 
                        input1 = input1.index_select(0, idxs)
                        input2 = input2.index_select(0, idxs)
                        labels = labels.index_select(0, idxs)
                        outputs = model(input1, input2)
                        _,pred = torch.max(outputs,1)
                        criterion.reduction = 'mean'
                        loss = criterion(outputs, labels)
                        #loss = torch.mean(loss_inst.index_select(0, idxs))
                    else:
                        criterion.reduction = 'mean'
                        loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * input1.size(0)
                running_corrects += torch.sum(pred==labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase=='train':
                train_acc.append(epoch_acc.item())
            elif phase=='test':
                test_acc.append(epoch_acc.item())

            if phase=='test' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, osp.join(Config['root_path'], Config['checkpoint_path'], 'best_model.pth'))
                print('Best model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'best_model.pth')))
                print('Save best check point at epoch %d'%(epoch+1))
            elif phase=='test':
                print('best model is save on epoch %d and best accuracy is %f'%(best_epoch, best_acc))
        torch.save(model.state_dict(),osp.join(Config['root_path'], Config['checkpoint_path'], 'model%d.pth'%(epoch+1)))
        print('Model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'model%d.pth'%(epoch+1))))

        # warmup: unfrozen model parameter gradually. 
        if (epoch+1)%warmup==0:
            model.model = unfreeze(model.model,unfreeze_num)
            unfreeze_num = unfreeze_num - 3

        # Learning rate decay
        if epoch < num_epochs-1: 
            for param_group in optimizer.param_groups:
                print('lr: {:.6f} -> {:.6f}'.format(param_group['lr'], param_group['lr'] * lr_decay))
                param_group['lr'] *= lr_decay
        epochs = np.arange(epoch+1)
        plt.figure()
        plt.plot(epochs, train_acc, label='loss')
        plt.plot(epochs, test_acc, label='val_loss')
        plt.xlabel('epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.legend()
        plt.show()
        plt.savefig('learning_acc_temp.png', dpi=256)

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))
    return train_acc, test_acc

def unfreeze(model,num):
    child_counter = 0
    for child in model.children():
        if child_counter < num:
            print("child ",child_counter," was frozen")
            child_counter += 1
            for param in child.parameters():
                param.requires_grad = False
        else:
            print("child ",child_counter," was not frozen")
            child_counter += 1
    return model

if __name__=='__main__':
    #Config['num_epochs'] = 20
    #dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, classes)
    classes = 2
    model = fashion_model(classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=Config['learning_rate'])
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

    train_acc,test_acc = train_model(get_dataloader, model, criterion, optimizer, device, num_epochs=Config['num_epochs'],  lr_decay=Config['lr_decay'])
    
    epochs = np.arange(Config['num_epochs'])

    plt.figure()
    plt.plot(epochs, train_acc, label='loss')
    plt.plot(epochs, test_acc, label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
    plt.savefig('learning_acc.png', dpi=256)
