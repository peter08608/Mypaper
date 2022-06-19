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
best_EPOCH = 0
valid_loss = []
def valid(model, device, epoch, EPOCH_SET, loss_func, valid_loader, batch_size, save_path):
    #####valid setting#####
    global best_loss
    global best_EPOCH
    global valid_loss
    model.eval()
    draw_loss = []
    loss_graph = 0.0
    start = time.time()
    with torch.no_grad():
        for step, (img, label) in enumerate(valid_loader):
            img, label = img.to(device), label.to(device)
            out = model(img)
            total_loss = loss_func(out,label)
    #####loss graph#####
            draw_loss.append(total_loss.item())
    x = range(0,len(draw_loss))
    plt.plot(x, draw_loss, '.-')
    draw_loss = np.mean(draw_loss)
    valid_loss.append(draw_loss)
    plt.title('Valid : BATCH_SIZE = ' + str(batch_size) + 'loss mean = ' + str(draw_loss))
    plt.xlabel('per '+str(batch_size))
    plt.ylabel('LOSS')
    plt.savefig(os.path.join(save_path,'valid/each_EOPCH/EPOCH_'+str(epoch).zfill(6)+'.png'))
    plt.cla()
    
    print('Valid_EPOCH : (%d / %d):[mean_loss:%.8f , %.2fs]' % (epoch, EPOCH_SET, draw_loss, time.time()-start))
    
    #####save best model#####
    if epoch == 0:
        best_EPOCH = epoch
        best_loss = draw_loss
        torch.save(model,os.path.join(save_path,'train/save/best.pt'))
    elif abs(best_loss) > abs(draw_loss) :
        best_EPOCH = epoch
        best_loss = draw_loss
        torch.save(model,os.path.join(save_path,'train/save/best.pt'))
    print('###current BEST EPOCH:'+str(best_EPOCH)+'###\n')
def train(model, device, LR, EPOCH_SET, optimizer, loss_func, train_loader, batch_size, save_path):
    #####train setting#####
    draw_loss = []
    global valid_loss
    for epoch in range(EPOCH_SET):
        model.train()
        print_mse = ''
        total_loss = 0
        with tqdm(total=len(train_loader), ncols=120, ascii=True) as t:
            
            for step, (img, label) in enumerate(train_loader):
                '''
                #####tensorboard#####
                writer = SummaryWriter()
                '''
                start = time.time()
                img, label = img.to(device), label.to(device) 
                #print(img)
                '''
                angle_out, dis_out = model(img)
                angle_out, dis_out = angle_out.squeeze(-1), dis_out.squeeze(-1) #訓練時batch大於1 時，loss就不下降，訓練效果很差。
                                                                                #而batch =1 時可以正常訓練。
                                                                                #後發現提示警告，預測的batch維度與真實的batch維度不同，
                                                                                #按照提示需要統一維度，可用squeeze將預測維度從（64，1）壓縮為（64）

                angle_loss = loss_func(angle_out.float(), angle.float())        #不加float()會報錯
                distance_loss = loss_func(diss_out.float(), distance.float())    #RuntimeError: Found dtype Double but expected Float”
                optimizer.zero_grad()
                angle_loss.backward(retain_graph=True)
                distance_loss.backward()
                optimizer.step()
                print('[%.2fs (%d / %d) angle_loss:%.4f distance_loss:%.4f]' % (time.time()-start, epoch, EPOCH_SET , angle_loss, distance_loss))
                '''
                
                out = model(img)
                
                optimizer.zero_grad()
                loss = loss_func(out,label)
                total_loss += loss.item() #test 此處要修正
                loss.backward()
                optimizer.step()
                
                #####progress bar#####
                t.set_description('Training : (%d / %d)' % (epoch, EPOCH_SET-1))
                t.set_postfix({'loss':'{:.8f}'.format(loss.item()),'time':'{:.4f}'.format(time.time()-start)})
                t.update()
                
                
                '''
                #####tensorboard#####
                grid = torchvision.utils.make_grid(img)
                writer.add_image('images', grid, 0)
                writer.add_graph(model, img)
                writer.add_scalar('Loss/angle_loss', angle_loss, step)
                writer.add_scalar('Loss/distance_loss', distance_loss, step)
                writer.close()
                '''
            
            #tqdm._instances.clear()
            t.close()
            mean_loss = total_loss / len(train_loader)
            print('Train_EPOCH : (%d / %d):[mean_loss:%.8f]' % (epoch, EPOCH_SET, mean_loss))
            #####save last model#####
            torch.save(model,os.path.join(save_path,'train/save/last.pt'))
            #####valid#####
            valid(model, device, epoch, EPOCH_SET, loss_func, train_loader, batch_size, save_path)
            #####loss graph#####
            draw_loss.append(mean_loss)
        #draw train loss
        x = range(0,len(draw_loss))
        plt.plot(x, draw_loss, '.-')
        plt.title('Train : BATCH_SIZE = '+str(batch_size)+'; LEARNING_RATE:'+str(LR))
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
        plt.savefig(os.path.join(save_path,'train/train_loss.png'))
        plt.cla()
        #draw valid loss
        x = range(0,len(valid_loss))
        plt.plot(x, valid_loss, '.-')
        plt.title('Train : BATCH_SIZE = '+str(batch_size)+'; LEARNING_RATE:'+str(LR))
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
        plt.savefig(os.path.join(save_path,'valid/valid_loss.png'))
        plt.cla()
      
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_printoptions(profile="defult")#打印tensor的精度，https://blog.csdn.net/Fluid_ray/article/details/109556867
    #####PATH#####
    train_root = './2022-6-19_follow_blackBG_whiteTG/detect_data_separate/trains'
    valid_root = './2022-6-19_follow_blackBG_whiteTG/detect_data_separate/valid'
    image_folder = 'images'
    label_folder = 'labels'

    #####Dataset setting#####
    Resize_set = torchvision.transforms.Resize((640,360),antialias = True)
    ColorJitter_set = transforms.ColorJitter(brightness=(1, 10), contrast=(1, 10), saturation=(1, 10), hue=(-0.2, 0.2))#亮度(brightness)、對比(contrast)、飽和度(saturation)和色調(hue)
    ByteToFloat = torchvision.transforms.ConvertImageDtype(torch.float)
    '''
    train_augmentation = torchvision.transforms.Compose([Resize_set,
                                                        ColorJitter_set,
                                                        torchvision.transforms.ToTensor(),
                                                        #torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                        ]) 
    '''
    train_augmentation = nn.Sequential(Resize_set,
                                        ColorJitter_set,
                                        ByteToFloat
                                        )
    #####train dataloder#####    
    batch_size = 8
    train_data=MyDataset(root=train_root, device=device, image_folder=image_folder, label_folder=label_folder, transform=train_augmentation)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    #####test dataloder##### 
    valid_data=MyDataset(root=valid_root, device=device, image_folder=image_folder, label_folder=label_folder, transform=train_augmentation)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size)
   
    #####model setting#####
    
    #model = Net()   #自製model
    model= models.resnet50(pretrained=True)
    #model= models.vgg16(pretrained=True) #會梯度爆炸
    
    #*** 修改全連線層的輸出 ***#
    
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
    print('GPU State:', device)
    #
    LR = 0.001
    EPOCH_SET = 1000

    optimizer = torch.optim.SGD(model.parameters(),lr=LR)
    loss_func = nn.MSELoss()
    #loss_func = nn.BCELoss()
    
    #####start traing#####
    save_path = mkdir()
    train(model, device, LR, EPOCH_SET, optimizer, loss_func, train_loader, batch_size, save_path)

def mkdir():
    first = 1
    while True:
        path = './runs/train/exp'+str(first)
        if not os.path.isdir(path):
            os.makedirs(os.path.join(path,'train','save'))
            os.makedirs(os.path.join(path,'valid','each_EOPCH'))
            break
        first += 1
    return path
            
    
if __name__ == "__main__":
    main()