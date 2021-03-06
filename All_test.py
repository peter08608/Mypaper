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

        img_origin = cv2.imread(path)
        y, x, channel = img_origin.shape

        img = np.ones((y,x+200,3), np.uint8)*255
        img[0:y, 0:x] = img_origin

        img = cv2.line(img, (x,50), (x+200,50), (0,0,255), 2)
        img = cv2.line(img, (x+100,50), ((x+100)+round(math.cos(math.radians(angle))*dis),50+round(math.sin(math.radians(angle))*dis)), (0,0,255), 2)
        
        out.write(img)
    out.release()
    '''
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
def txt_change():
    import os

    Path = r'C:\Users\PeterChuang\Desktop\yolov5\Mypaper\detect_data_separate\valid\old_labels'
    save_path = r'C:\Users\PeterChuang\Desktop\yolov5\Mypaper\detect_data_separate\valid\labels'
    allFileList = os.listdir(Path)
    MAX = 0
    for file in allFileList:
        path = os.path.join(Path,file)
        f = open(path, 'r')
        f = f.readlines()
        f = f[0]
        f = f.split(' ')
        f1 = float(f[0])
        f2 = float(f[1])
        txt = str(f1/180) + ' ' + str(f2/4000)
        path = os.path.join(save_path,file)
        f = open(path, 'w')
        f.write(txt)
        f.close()

def txt_list():
    import numpy
    origin_label_path = r"C:\Users\PeterChuang\Desktop\yolov5_forMypaper\pokemon_muti_pattern\label\00001.txt"
    All_origin_data = []
    f = open(origin_label_path)
    for line in f.readlines():
        line = line.split(' ')
        All_origin_data.append([float(line[0]),float(line[1])])
    f.close()
    All_origin_data = numpy.array(All_origin_data)#list to numpy
    min_dis_index = numpy.argmin(All_origin_data[:,0]) #find min dis
    min_dis_at_angle, min_dis = All_origin_data[min_dis_index]
    
    difference_array = numpy.absolute(All_origin_data[:,0] - 340)
    index_array = difference_array.argmin()
    target_angle, target_dis = All_origin_data[index_array]
    #print(min_dis_at_angle,':',min_dis)
    #print(All_origin_data[:,1])
    print(difference_array)
    print(index_array)
    print(target_angle,':', target_dis)
if __name__ == "__main__":
    txt_list()

    #root = r'C:\Users\PeterChuang\Desktop\Mypaper\detect_data_separate\valid\images'

    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (800,600))
    
    #detect(root,out)