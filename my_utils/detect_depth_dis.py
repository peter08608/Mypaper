# -*- coding: utf-8 -*-
#Pytorch
import cv2

def two_pattern_test(x, img_detect, p):
    split_filename = str(p.name).split('.')
    label_GT = os.path.join(p.parents[1], 'label_GT', split_filename[0]+'_0.txt')
    f = open(label_GT)
    f = f.readlines()
    f = f[0]
    f = f.split(' ')
    angle_GT, dis_GT = f
    angle_GT, dis_GT = float(angle_GT), float(dis_GT)
    
    angle = img_detect[0][0] #predict angle
    dis = img_detect[0][1]  #predict dis
    return x, dis, dis_GT, angle, angle_GT
    
def two_pattern_test2(x, predata,img_detect, p):
    import math
    split_filename = str(p.name).split('.')
    label_GT = os.path.join(p.parents[1], 'label_GT', split_filename[0]+'_1.txt')
    f = open(label_GT)
    f = f.readlines()
    f = f[0]
    f = f.split(' ')
    angle_GT, dis_GT = f
    angle_GT, dis_GT = float(angle_GT), float(dis_GT)
    
    angle = img_detect[0][0] #predict angle
    dis = img_detect[0][1]  #predict dis
    
    r = (209-142)/1280
    angle_range = abs((predata[0]-x)*r)
    cos_B = angle_range * (math.pi/180)
    
    fin = 0
    angle_loss = 0
    angle_MAE = 0
    if predata[0] > x:
        b2 = (predata[1]**2)+(dis**2)-(2*dis*predata[1]*(math.cos(cos_B)))
        b = b2**0.5
        cosA = (b**2 + predata[1]**2 - dis**2)/(2*b*predata[1])
        if cosA > 1:
            cosA = 1
        elif cosA < -1:
            cosA = -1
        cosA = math.acos(cosA)
        A = (180/math.pi)*cosA
        fin = (180-A)/180
        angle_loss = (predata[4] - fin)**2
        angle_MAE = abs(predata[4] - fin)
        
        
    elif predata[0] <= x:
        b2 = (dis**2)+(predata[1]**2)-(2*predata[1]*dis*(math.cos(cos_B)))
        b = b2**0.5
        cosA = (b**2 + dis**2 - predata[1])/(2*b*dis)
        if cosA > 1:
            cosA = 1
        elif cosA < -1:
            cosA = -1
        cosA = math.acos(cosA)
        A = (180/math.pi)*cosA
        fin = (180-A)/180
        angle_loss = (angle_GT - fin)**2
        angle_MAE = abs(angle_GT - fin)
        
    dis_loss =( (predata[1]-predata[2])**2 + (dis-dis_GT)**2 )/2
    dis_MAE = ( abs(predata[1]-predata[2]) + abs(dis-dis_GT) )/2
    
    return angle_loss, dis_loss, angle_MAE, dis_MAE

def calculate_RMSE_and_MAE(img_detect, p):
    split_filename = str(p.name).split('.')
    label_GT = os.path.join(p.parents[1], 'label_GT', split_filename[0]+'_0.txt')
    f = open(label_GT)
    f = f.readlines()
    f = f[0]
    f = f.split(' ')
    angle_GT, dis_GT = f
    angle_GT, dis_GT = float(angle_GT), float(dis_GT)
    
    angle = img_detect[0][0] #predict angle
    dis = img_detect[0][1]  #predict dis
    
    angle_loss = (angle - angle_GT)**2 #RMSE
    dis_loss = (dis - dis_GT)**2
    
    angle_MAE = abs(angle - angle_GT) # MAE
    dis_MAE = abs(dis - dis_GT)
    return angle_loss, dis_loss, angle_MAE, dis_MAE

from PIL import Image
def default_loader(img_pros):
    img_pros = Image.fromarray(cv2.cvtColor(img_pros, cv2.COLOR_BGR2RGB))
    img_pros = img_pros.convert('RGB')
    return img_pros

import torch
import torchvision
import numpy
import math
import os 
import time
def detect_depth_dis( img_origin, img_pros, output, opt, model, device, p):
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print('GPU State:', device)
    start = time.time()
    #參數設定
    sh_x, sh_y = opt.show_size.split(',')
    re_y, re_x = opt.resize.split(',')
    Resize_set = torchvision.transforms.Resize((int(re_y),int(re_x)),antialias = True) #這裡要注意(y,x)
    ToTensor = torchvision.transforms.ToTensor()

    resouce_nor = torchvision.transforms.Compose([Resize_set,
                                                ToTensor,
                                                ]) 
    
    #已裁切圖輸入模型
    img = default_loader(img_pros)
    img = resouce_nor(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    out = model(img)
    
    #將數據視覺化
    img_detect = out.cpu().detach().numpy()
    angle = img_detect[0][0]*180 #draw angle
    dis = img_detect[0][1]*50  #draw dis
    img_origin = cv2.resize(img_origin, (int(sh_x), int(sh_y)), interpolation=cv2.INTER_AREA) 
   
    y, x, channel = img_origin.shape
    re_x, re_y = round(int(re_x)/3), round(int(re_y)/3)
    img_pros = cv2.resize(img_pros, (re_x, re_y), interpolation=cv2.INTER_AREA)
    img = numpy.ones((y,x+200,3), numpy.uint8)*255
    img[0:y, 0:x] = img_origin
    img[y-re_y:y, x+20:x+20+re_x] = img_pros
    
    cv2.putText(img, 'target', (x+70, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    img = cv2.line(img, (x,50), (x+200,50), (0,0,0), 2)
    img = cv2.line(img, (x+100,50), ((x+100)+round(math.cos(math.radians(angle))*dis),50+round(math.sin(math.radians(angle))*dis)), (0,0,255), 2)
    
    angle_txt = str(round(img_detect[0][0]*180,2)) #real angle
    dis_txt = str(round(img_detect[0][1]*1000,2)) #real dis
    cv2.putText(img, 'angle : '+angle_txt, (x, 200), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'dis : '+dis_txt+'mm', (x, 250), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    #cv2.imshow('image',img)
    #cv2.waitKey(20)
    output.write(img)
    
    pro_time = time.time() - start
    
    angle_loss, dis_loss, angle_MAE, dis_MAE = calculate_RMSE_and_MAE(img_detect, p)
    return angle_loss, dis_loss, angle_MAE, dis_MAE, pro_time
    '''
    if check == False:
        return two_pattern_test(x, img_detect, p)
    elif check == True:
        return two_pattern_test2(x, predata, img_detect, p)
    '''