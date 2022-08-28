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
import math
from .PointUtility import *

class RectUtility:
  @staticmethod
  def boundingRect(points):
    if len(points) == 0:
      return
    p0 = points[0]
    result = [p0[0],p0[1],p0[0],p0[1]]
    for pnt in points:
      if pnt[0] < result[0]:
        result[0] = pnt[0]
      if pnt[1] < result[1]:
        result[1] = pnt[1]
      if pnt[0] > result[2]:
        result[2] = pnt[0]
      if pnt[1] > result[3]:
        result[3] = pnt[1]
    return RectUtility.twoPointsToRect(result)




  @staticmethod
  def moveRect(rct,offset):
    result = rct
    result[0] += offset[0]
    result[1] += offset[1]
    return result

  @staticmethod
  def moveRects(rects,offset):
    result = []
    for rct in rects:
      result.append(RectUtility.moveRect(rct,offset))
    return result

  @staticmethod
  def rectTo2Points(rect):
    return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]

  @staticmethod
  def rectTo2PointsList(rects):
    result = []
    for rct in rects:
      result.append(RectUtility.rectTo2Points(rct))
    return result

  @staticmethod
  def twoPointsToRect(r2):
    return [r2[0], r2[1], abs(r2[2] - r2[0]) +1,abs( r2[3] - r2[1])+1]

  @staticmethod
  def twoPointsToRectList(rects):
     result = []
     for rct in rects:
       result.append(RectUtility.twoPointsToRect(rct))
     return result
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


  @staticmethod
  def getRectPoints(rct):
    result = []
    result.append([rct[0],rct[1]])
    result.append([rct[0]+rct[2], rct[1]])
    result.append([rct[0]+rct[2], rct[1]+rct[3]])
    result.append([rct[0], rct[1]+rct[3]])
    return result

  @staticmethod
  def getRectCenter(region):
    return (int(region[0] + region[2]/2) ,int(region[1] + region[3]/2))

  @staticmethod
  def rotateRect(rct,center,angle,around_center = True):
    points = RectUtility.getRectPoints(rct)
    if around_center :
      cntr = RectUtility.getRectCenter(rct)
    else: cntr = center
    result = PointUtility.rotatePoints(points,cntr,angle)
    return RectUtility.boundingRect(result)



  @staticmethod
  def rotateRects(rects,center,angle,around_center = True, background = None):
    result = []

    cntr = center
    if background != None:
      cntr = RectUtility.getRectCenter(background)
      background_r = RectUtility.rotateRect(background,cntr,angle,False)

    for rct in rects:
      result.append(RectUtility.rotateRect(rct,cntr,angle,around_center))

    if background != None:
        offset = [-background_r[0],-background_r[1]]
        result =  RectUtility.moveRects(result,offset)

    return result

  @staticmethod
  def getImageRect(image):
    return [0,0,image.shape[1],image.shape[0]]


  @staticmethod
  def drawRects(image,regions,color = (0,255,0),thicknes = 2):
    result = image.copy()
    for region in regions:
       r = RectUtility.rectTo2Points(region)
       cv2.rectangle(result,(r[0],r[1]),(r[2],r[3]),color,thicknes)
    return result
  
  @staticmethod
  def is_empty(region):
    if region == None :
      return True
    else:
          return region[2] * region[3] == 0


