# -*- coding: utf-8 -*-
#Pytorch
import cv2

from PIL import Image
def default_loader(img_pros):
    img_pros = Image.fromarray(cv2.cvtColor(img_pros, cv2.COLOR_BGR2RGB))
    img_pros = img_pros.convert('RGB')
    return img_pros

import torch
import torchvision
import numpy
import math
def detect_depth_dis( img_origin, img_pros, output, save_dir, opt, device):
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print('GPU State:', device)
    
    #model=torch.load('./train_run/6-11_resnet101/train/save/last.pt')
    model=torch.load(opt.myweight, map_location=device)
    model.eval()
    model = model.to(device)
    #print(model)
    Resize_set = torchvision.transforms.Resize((640,360),antialias = True) #這裡要注意(y,x)
    ByteToFloat = torchvision.transforms.ConvertImageDtype(torch.float)

    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize((640,360)),
                                                        torchvision.transforms.ToTensor(),
                                                        #torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                        ]) 

    img = default_loader(img_pros)
    
    img = train_augmentation(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    out = model(img)
    img_detect = out.cpu().detach().numpy()
    angle = img_detect[0][0]*180 #draw angle
    dis = img_detect[0][1]*50  #draw dis
    img_origin = cv2.resize(img_origin, (640, 360), interpolation=cv2.INTER_AREA)
    y, x, channel = img_origin.shape

    img_pros = cv2.resize(img_pros, (160, 90), interpolation=cv2.INTER_AREA)
    img = numpy.ones((y,x+200,3), numpy.uint8)*255
    img[0:y, 0:x] = img_origin
    img[y-90:y, x+20:x+180] = img_pros
    

    img = cv2.line(img, (x,50), (x+200,50), (0,0,0), 2)
    img = cv2.line(img, (x+100,50), ((x+100)+round(math.cos(math.radians(angle))*dis),50+round(math.sin(math.radians(angle))*dis)), (0,0,255), 2)
    
    angle_txt = str(round(img_detect[0][0]*180,2)) #real angle
    dis_txt = str(round(img_detect[0][1]*1000,2)) #real dis
    cv2.putText(img, 'angle : '+angle_txt, (x, 200), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'dis : '+dis_txt+'mm', (x, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    #cv2.imshow('image',img)
    #cv2.waitKey(20)
    output.write(img)
    