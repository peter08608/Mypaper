import math
import os
import cv2
import numpy as np
from detect_depth_dis import detect_depth_dis

def detect(root,out):
    All_image_name = os.listdir(root)
    for Image_name in All_image_name:
        path = os.path.join(root,Image_name)
        img_detect = detect_depth_dis(path)
        img_detect = img_detect.cpu().detach().numpy()
        print(img_detect[0])
        angle = abs(img_detect[0][0]*180)
        dis = abs(img_detect[0][1]*200)
        print(angle,':',dis)

        img_orige = cv2.imread(path)
        y, x, channel = img_orige.shape

        img = np.ones((y,x+200,3), np.uint8)*255
        img[0:y, 0:x] = img_orige

        img = cv2.line(img, (x,50), (x+200,50), (0,0,255), 2)
        img = cv2.line(img, (x+100,50), ((x+100)+round(math.cos(math.radians(angle))*dis),50+round(math.sin(math.radians(angle))*dis)), (0,0,255), 2)
        
        out.write(img)
    out.release()
    '''
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
if __name__ == "__main__":
    root = r"C:\Users\PeterChuang\Desktop\Mypaper\detect_data_separate\valid\images"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (800,600))
    
    detect(root,out)