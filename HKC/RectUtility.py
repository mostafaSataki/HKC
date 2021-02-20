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

class RectUtility:

  @staticmethod
  def rectTo2Points(rect):
    return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]

  @staticmethod
  def twoPointsToRect(r2):
    return [r2[0], r2[1], r2[2] - r2[0], r2[3] - r2[1]]


  @staticmethod
  def flipHorzRect(rect ,shape):

    r2 = RectUtility.rectTo2Points(rect)

    x1p = shape[1] - r2[2]
    x2p = shape[1] - r2[0]

    return RectUtility.twoPointsToRect([x1p,r2[1],x2p,r2[3]])

  @staticmethod
  def resize(rect,scale):
    return ( int(rect[0] * scale[0]),  int(rect[1] * scale[1]) ,  int(rect[2] * scale[0]),  int(rect[3] * scale[1]))
