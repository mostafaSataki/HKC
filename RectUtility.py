import  cv2
import  numpy as np
from .FileUtility import *
import  os
import cv2
import time
from tqdm import tqdm
import math
import subprocess
from scipy.spatial import distance as dist
from  .CvUtility import *

class RectUtility:

  @staticmethod
  def rectTo2Points(rect):
    return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]

  @staticmethod
  def twoPointsToRect(r2):
    return [r2[0], r2[1], r2[2] - r2[0], r2[3] - r2[1]]

  @staticmethod
  def center2Rect(center,size):
    return [int(center[0] - size[0] /2),int(center[1] - size[1] /2),int( size[0]) ,int( size[1]) ]

  @staticmethod
  def flipHorzRect(rect ,shape):

    r2 = RectUtility.rectTo2Points(rect)

    x1p = shape[1] - r2[2]
    x2p = shape[1] - r2[0]

    return RectUtility.twoPointsToRect([x1p,r2[1],x2p,r2[3]])

  @staticmethod
  def toYoloRect(rct,back_rect):
    w = back_rect[2]
    h = back_rect[3]

    center = CvUtility.getRectCenter(rct)
    region = [center[0],center[1],rct[2]/2,rct[3]/2]
    return [region[0] / w ,region[1] / h , region[2] / w ,region[3] / h]

  @staticmethod
  def fromYoloRect(yolo_rect,back_rect):
    w = back_rect[0]
    h = back_rect[1]
    region = [yolo_rect[0] - yolo_rect[2],yolo_rect[1] - yolo_rect[3],yolo_rect[2] * 2 ,yolo_rect[3] * 2]
    return [int(region[0] * w),int(region[1] * h),int(region[2] * w) ,int(region[3] * h)]


  @staticmethod
  def resize(rect,scale):
    return ( int(rect[0] * scale[0]),  int(rect[1] * scale[1]) ,  int(rect[2] * scale[0]),  int(rect[3] * scale[1]))

  @staticmethod
  def cropRect(back_rect,rect):
    rect = CvUtility.intersection(back_rect,rect)
    back_rect_2p = RectUtility.rectTo2Points(back_rect)
    rect_2p = RectUtility.rectTo2Points(rect)

    res_rect_2p = rect_2p
    x1 = res_rect_2p[0] - back_rect_2p[0]
    y1 =  res_rect_2p[1] - back_rect_2p[1]

    x2 = res_rect_2p[2] - back_rect_2p[0]
    y2 = res_rect_2p[3] - back_rect_2p[1]

    return RectUtility.twoPointsToRect([x1,y1,x2,y2])


  @staticmethod
  def cropRects(back_rect,rects):
    result = []
    for r in rects :
      result.append(RectUtility.cropRect(back_rect,r))
    return result
