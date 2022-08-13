import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from my_utils.crop_img import *
from my_utils.detect_depth_dis import *

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
        
    ################################################
    #angel dis model
    my_model=torch.load(opt.myweight, map_location=device)
    my_model.eval()
    my_model = my_model.to(device)
    #output video
    sh_x, sh_y = opt.show_size.split(',')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(os.path.join(save_dir,'output.mp4'),fourcc, 30.0, (int(sh_x)+200, int(sh_y)))
    
    angle_loss_list = []
    dis_loss_list = []
    angle_MAE_list = []
    dis_MAE_list = []
    pro_time_list = []
    
    check = False
    runtime = 0
    save = []
    ################################################

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                #####################################################
                new_im0 = im0.copy()
                y, x, channel = new_im0.shape
                #cv2.imshow(str(p), im0)
                #cv2.waitKey()
                
                cropsz_x, cropsz_y = opt.crop_size.split(',')
                #####################################################
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img or True:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        #####################################################
                        xyxy_num_0 = int(xyxy[0].cpu().numpy())#left x
                        xyxy_num_1 = int(xyxy[1].cpu().numpy())#left y
                        xyxy_num_2 = int(xyxy[2].cpu().numpy())#right x
                        xyxy_num_3 = int(xyxy[3].cpu().numpy())#right y
                        
                        #new_img = crop_follow_blackBG(new_im0, xyxy_num_0, xyxy_num_1, xyxy_num_2, xyxy_num_3)
                        #new_img = crop_middle_blackBG(new_im0, (int(cropsz_x), int(cropsz_y)), xyxy_num_0, xyxy_num_1, xyxy_num_2, xyxy_num_3)
                        new_img = adaptive_center_crop(new_im0, (int(cropsz_x), int(cropsz_y)), xyxy_num_0, xyxy_num_1, xyxy_num_2, xyxy_num_3)#目標置中裁切(原始圖, 裁切大小(x,y), l_x, l_y, r_x, r_y)
                        '''
                        x_range = abs((xyxy_num_0-xyxy_num_2)/2)
                        if runtime == 0:
                            runtime = runtime + 1
                            check = False
                            save = detect_depth_dis(x_range, save, check, new_im0, new_img, output, opt, my_model, device, p)
                        elif runtime == 1:
                            runtime = 0
                            check = True
                            t = detect_depth_dis(x_range, save, check, new_im0, new_img, output, opt, my_model, device, p)
                            angle_loss, dis_loss, angle_MAE, dis_MAE = t
                            angle_loss_list.append(angle_loss)
                            dis_loss_list.append(dis_loss)
                            angle_MAE_list.append(angle_MAE)
                            dis_MAE_list.append(dis_MAE)
                            save = []
                        '''
                        t = detect_depth_dis( new_im0, new_img, output, opt, my_model, device, p)
                        angle_loss, dis_loss, angle_MAE, dis_MAE, pro_time = t
                        angle_loss_list.append(angle_loss)
                        dis_loss_list.append(dis_loss)
                        angle_MAE_list.append(angle_MAE)
                        dis_MAE_list.append(dis_MAE)
                        pro_time_list.append(pro_time)
                        
                        #####################################################



            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    
    print(f'Done. ({time.time() - t0:.3f}s)')
    
    ###############################
    output.release()
    
    angle_loss_arry, dis_loss_arry = numpy.array(angle_loss_list), numpy.array(dis_loss_list)
    angle_MAE_arry, dis_MAE_arry = numpy.array(angle_MAE_list), numpy.array(dis_MAE_list)
    pro_time_arry = numpy.array(pro_time_list)
    angle_MSE, dis_MSE = numpy.mean(angle_loss_arry), numpy.mean(dis_loss_arry)
    angle_RMSE, dis_RMSE = pow(angle_MSE, 0.5), pow(dis_MSE, 0.5)
    angle_MAE, dis_MAE = numpy.mean(angle_MAE_arry), numpy.mean(dis_MAE_arry)
    pro_time_mean = numpy.mean(pro_time_arry)
    f = open(os.path.join(save_dir, 'Precision.txt'), 'w')
    f.write(' angle_RMSE:'+str(angle_RMSE))
    f.write('\n dis_RMSE:'+str(dis_RMSE))
    f.write('\n angle_MAE:'+str(angle_MAE))
    f.write('\n dis_MAE:'+str(dis_MAE))
    f.write('\n pro_time_mean:'+str(pro_time_mean))
    f.close()
    ###############################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    parser.add_argument('--myweight', type=str, default='', help='my model.pt path(s)')
    parser.add_argument('--crop-size', type=str, default='640,640', help='crop img size[x, y]')
    parser.add_argument('--resize', type=str, default='640,640', help='transforms resize img [y, x]')
    parser.add_argument('--show-size', type=str, default='640,640', help='resize output video by img [x, y]')
    
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
            
    
