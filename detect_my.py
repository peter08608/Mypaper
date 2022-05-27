# -*- coding: utf-8 -*-
#Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
#Python
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
#Mine
from CNN_model import *
from Dataset import *
best_loss = 0.0
def valid(model, device, epoch, loss_func, valid_loader, batch_size):
    #####valid setting#####
    global best_loss
    model.eval()
    draw_loss = []
    loss_graph = 0.0
    with torch.no_grad():
        for step, (img, label) in enumerate(valid_loader):
            start = time.time()
            img, label = img.to(device), label.to(device)
            out = model(img)
            total_loss = loss_func(out,label)
    #####loss graph#####
            draw_loss.append(total_loss.item())
    x = range(0,len(draw_loss))
    plt.plot(x, draw_loss, '.-')
    draw_loss = np.mean(draw_loss)
    plt.title('Valid : BATCH_SIZE = ' + str(batch_size) + 'loss mean = ' + str(draw_loss))
    plt.xlabel('per '+str(batch_size))
    plt.ylabel('LOSS')
    plt.savefig('./Loss_graph/valid/EPOCH_'+str(epoch).zfill(6)+'.png')
    plt.cla()
    
    print('\n*Valid* : [%.2fs EPOCH:%d total_loss:%.8f]' % (time.time()-start, epoch, draw_loss))
    
    #####save last model#####
    if epoch == 0:
        best_loss = draw_loss
        torch.save(model,'./Loss_graph/train/save/best.pt')
    elif abs(best_loss) > abs(draw_loss) :
        best_loss = draw_loss
        torch.save(model,'./Loss_graph/train/save/best.pt')
        
def main():
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('GPU State:', device)

    torch.set_printoptions(profile="default")#打印tensor的精度，https://blog.csdn.net/Fluid_ray/article/details/109556867
    #####PATH#####
    train_root = './detect_data_separate/trains'
    valid_root = './detect_data_separate/valid'
    image_folder = 'images'
    label_folder = 'labels'

    #####Dataset setting#####
    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize((600,600)),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                        ]) 
    #####train dataloder#####    
    batch_size = 32
    train_data=MyDataset(root=train_root, image_folder=image_folder, label_folder=label_folder, transform=train_augmentation)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    #####test dataloder##### 
    valid_data=MyDataset(root=valid_root, image_folder=image_folder, label_folder=label_folder, transform=train_augmentation)
    valid_loader = DataLoader(dataset=valid_data, batch_size=8)
   
    #####model setting#####
    #model = Net().to(device)   #自製model
    model= models.resnet34(pretrained=True)
    #model= models.vgg16(pretrained=True) #會梯度爆炸
    #* 修改全連線層的輸出 *#
    
    #適用:resnet
    num_ftrs = model.fc.in_features#in_feature is the number of inputs for your linear layer
    model.fc = nn.Linear(num_ftrs, 2)
    '''
    #適用:vgg
    num_ftrs = model.classifier[6].in_features#in_feature is the number of inputs for your linear layer
    model.classifier[6] = nn.Linear(num_ftrs, 2)
    '''
    model = model.to(device)
    print(model)
    
    #
    LR = 0.001
    EPOCH_SET = 1000

    optimizer = torch.optim.SGD(model.parameters(),lr=LR)
    loss_func = nn.MSELoss()
    #loss_func = nn.BCELoss()
    
    #####start traing#####
    train(model, device, LR, EPOCH_SET, optimizer, loss_func, train_loader, batch_size)
    
if __name__ == "__main__":
    main()