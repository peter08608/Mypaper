import numpy

def take_label_info_2(angle_range_setting, f, target_wpoint, my_filename):
    All_origin_data = []
    for line in f.readlines():
        line = line.split(' ')
        All_origin_data.append([float(line[0]),float(line[1])])
    f.close()
    All_origin_data = numpy.array(All_origin_data)#list to numpy
    min_dis_index = numpy.argmin(All_origin_data[:,1]) #find min dis
    min_dis_at_angle, min_dis = All_origin_data[min_dis_index]
    difference_array = numpy.absolute(All_origin_data[:,0] - target_wpoint)
    index_array = difference_array.argmin()
    target_angle, dis = All_origin_data[index_array]
    angle = abs(min_dis_at_angle - target_angle)
    if angle > 180 :
        print('Warning the angle over 180! filename : ', my_filename)
    angle = 90 - angle
    return angle, dis

def take_label_info_1(angle_range_setting, f, target_wpoint, my_filename):
    min_dis = 9999999999999999.9
    cache_line = []
    for line in f.readlines():
        line = line.split(' ')
        if abs(target_wpoint - float(line[2])) < min_dis:
            min_dis = abs(target_wpoint - float(line[2]))
            cache_line = line
    f.close()
    angle = cache_line[3]
    dis = cache_line[5]
    return angle, dis
    
def crop_follow_blackBG(new_im0, xyxy_num_0, xyxy_num_1, xyxy_num_2, xyxy_num_3):
    from PIL import Image
    from rembg import remove
    
    y, x, channel = new_im0.shape
    crop_img = new_im0[xyxy_num_1:xyxy_num_3, xyxy_num_0:xyxy_num_2]
    crop_img = remove(crop_img)
    crop_img = Image.fromarray(crop_img)
    crop_img = crop_img.convert("RGB")
                            
    #cv2.imshow(str(p), crop_img)
    #cv2.waitKey()
                            
    new_img = numpy.zeros((y,x,3), numpy.uint8)
    new_img[xyxy_num_1:xyxy_num_3, xyxy_num_0:xyxy_num_2] = crop_img
    return new_img
    
def adaptive_center_crop(new_im0, fill_edge, xyxy_num_0, xyxy_num_1, xyxy_num_2, xyxy_num_3):
    from PIL import Image
    import math as m
    
    xyxy_num_0, xyxy_num_1, xyxy_num_2, xyxy_num_3 = xyxy_num_0 + fill_edge[0], xyxy_num_1 + fill_edge[1], xyxy_num_2 + fill_edge[0], xyxy_num_3 + fill_edge[1]
    mid_x, mid_y = m.ceil((xyxy_num_0+xyxy_num_2)/2), m.ceil((xyxy_num_1+xyxy_num_3)/2)
    l_x, l_y, r_x, r_y = mid_x-(m.ceil(fill_edge[0]/2)), mid_y-(m.ceil(fill_edge[1]/2)), mid_x+(m.ceil(fill_edge[0]/2)), mid_y+(m.ceil(fill_edge[1]/2))
    y, x, channel = new_im0.shape
    new_img = numpy.zeros((y + (fill_edge[1]*2), x + (fill_edge[0]*2),3), numpy.uint8)
    
    new_img[fill_edge[1]:fill_edge[1]+y, fill_edge[0]:fill_edge[0]+x] = new_im0
    new_img = new_img[l_y:r_y, l_x:r_x]
    return new_img
    