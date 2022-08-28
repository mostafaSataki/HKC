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


class PointUtility:
  @staticmethod
  def  getLineAngle( p1, p2):
     deg = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
     if (deg < 0):
        deg = 360 + deg;

     return deg;

  @staticmethod
  def pointsDistance(p1,p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

  @staticmethod
  def rotatePoint(src_point, center, angle):
    rad = PointUtility.pointsDistance (src_point ,center)
    teta = PointUtility.getLineAngle(center,src_point)

    new_teta = angle + teta;
    result = src_point
    result[0] = center[0] + round( rad * math.cos(math.radians(new_teta)))
    result[1] = center[1] + round(rad * math.sin(math.radians(new_teta)))
    return result

  @staticmethod
  def rotatePoints(src_points, center, angle):
    result = []
    for src_point in src_points :
      result.append(PointUtility.rotatePoint(src_point,center,angle))
    return result


