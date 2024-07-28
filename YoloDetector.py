

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from  yolov7 import *
from  yolov7 import models
from  yolov7.models import *
from  yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages,loadImage
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from HKC.FileUtility import  *
from HKC.CvUtility import  *
from HKC.RectUtility import RectUtil

class MyDetect:
    def __init__(self,weights):
        # self.names = FileUtility.read_text_list(labels_filename)
        self.imgsz = 640
        trace = True
        self.classes = None
        self.agnostic_nms = False
        self.augment = False

        device_ = ''

        # Initialize
        set_logging()
        self.device = select_device(device_)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride

        imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        # if trace:
        #     self.model = TracedModel(self.model, self.device, imgsz)



        # self.model = torch.jit.load(r'C:\Source\Repo\Konkor\Konkor\python\traced_model.pt', map_location=self.device)
        # self.half = False
        if self.half:
            self.model.half()  # to FP16

            # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]


        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once


    def detect(self,im0,    conf_thres = 0.25,      iou_thres = 0.45,offset= (0,0)):
        img = loadImage(im0, img_size=self.imgsz, stride=self.stride)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=self.augment)[0]
        # pred = self.model(img)[0]


        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        regions_info = self._get_region_info(pred,im0,img,offset)

        return regions_info


    def nms(self,regions_info,iou_thres = 0.45):
        result = []

        regions = regions_info
        regions = sorted(regions,  key=lambda x: x[1],reverse=True)


        while len(regions):
            max_region = regions.pop(0)
            j = 0
            while (j < len(regions)):
                iou_value =  RectUtil.IOU(max_region[0],regions[j][0])
                if iou_value >= iou_thres:
                    regions.pop(j)
                else : j +=1
            result.append(max_region)
        return result









    def detectex(self,im0, grid_cols, grid_rows, conflict_cols=0.0, conflict_rows=0.0,  conf_thres = 0.25,      iou_thres = 0.45):
        breaked_images = CvUtility.break_image(im0,grid_cols, grid_rows, conflict_cols, conflict_rows)

        regions_infos = []
        for breaked_image,roi in breaked_images:

            regions_infos.extend(self.detect(breaked_image,conf_thres,iou_thres,(roi[0],roi[1])) )

        return self.nms(regions_infos,iou_thres)








    def _get_region_info(self,pred,im0,img,offset= (0,0)):
        result = []
        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if xyxy[0] > xyxy[2]:
                        xyxy[0],xyxy[2] = xyxy[2],xyxy[0]
                    if xyxy[1] > xyxy[3]:
                        xyxy[1],xyxy[3] = xyxy[3],xyxy[1]

                    if offset == (0,0):

                        result.append((xyxy,conf,cls))
                    else :
                        xyxy_offset = (xyxy[0]+offset[0],xyxy[1]+offset[1],xyxy[2]+offset[0],xyxy[3]+offset[1])
                        result.append((xyxy_offset,conf,cls))

        return result


    def draw(self,im0,regions_info):
        for xyxy, conf, cls in regions_info:
            label = f'{self.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
            # CvUtility.imshowScale("view",im0,(1000,1000))
            # cv2.waitKey(0)





def detect_file_yolo(src_filename,dst_path,weights_filename):


    detector = MyDetect(weights_filename)
    im0 = cv2.imread(src_filename,1)
    regions_info = detector.detect(im0,iou_thres=0.65)
    detector.draw(im0,regions_info)

    return CvUtility.imwrite_branch(im0,dst_path,src_filename)


def detect_folder_yolo(src_path,dst_path,weights_filename):

    detector = MyDetect(weights_filename)

    src_filenames = FileUtility.getFolderImageFiles(src_path)
    dst_filenames = FileUtility.getDstFilenames2(src_filenames,src_path,dst_path)
    for i,src_filename in enumerate(src_filenames):
        print(src_filename)
        dst_filename = dst_filenames[i]
        im0 = cv2.imread(src_filename, 1)
        # regions_info = detector.detect(im0)
        regions_info = detector.detectex(im0,1,2,0,0.1)
        detector.draw(im0, regions_info)
        cv2.imwrite(dst_filename,im0)


