# -*- coding: utf-8 -*-
#Pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

def detect_depth_dis(path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('GPU State:', device)
    
    model=torch.load('./train_run/5-27_resnet34/train/save/last.pt')
    model.eval()
    num_ftrs = model.fc.in_features#in_feature is the number of inputs for your linear layer
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    #print(model)
    
    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize((600,600)),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                        ]) 
    img = default_loader(path)
    img = train_augmentation(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    out = model(img)
    return out
    