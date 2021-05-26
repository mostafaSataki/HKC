import enum
import os
import sys
import cv2
import  numpy as np
from .ModuleUtility import *




class FaceDetectType(enum.Enum):
   Cascade = 1
   Dnn = 2

class FaceDetectCascade:
  def __init__(self,model_name = 'haarcascade_frontalface_alt2.xml'):
    filename = ModuleUtility.joinApp(model_name,'models')
    self.model_ = cv2.CascadeClassifier(filename)


  def detect(self, image, scale_factor=1.3, min_neighbors=5):
    return self.model_ .detectMultiScale(image, scale_factor, min_neighbors)

class FaceDetectDnn:


  def __init__(self,model_name ='opencv_face_detector_uint8.pb',config_name = 'opencv_face_detector.pbtxt',confidence = 0.5):
    self.confidence_ = confidence
    model_filename = ModuleUtility.joinApp(model_name,'models')
    config_filename = ModuleUtility.joinApp(config_name,'models')

    self.model_ = cv2.dnn.readNet(model_filename,config_filename)

  def detect(self,image):

    (self.h_, self.w_) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    self.model_.setInput(blob)
    self.detections_ = self.model_.forward()
    return self.getDetectResult()

  def getDetectResult(self):

    self.regions_ = []
    self.confidences_ = []

    for i in range(0,self.detections_.shape[2]):
      confidence = self.detections_[0, 0, i, 2]

      if confidence > self.confidence_:
        box = self.detections_[0, 0, i, 3:7] * np.array([self.w_, self.h_, self.w_, self.h_])
        (startX, startY, endX, endY) = box.astype("int")
        box = (startX,startY,endX - startX ,endY - startY)

        self.regions_.append(box)
        self.confidences_.append(confidence)

    return self.regions_



class FaceDetect:
  def __init__(self,type):
    self.type_ = type
    self.createModel()


  def detect(self, image):
     return self.model_.detect(image)

  def createModel(self):
    if self.type_ == FaceDetectType.Cascade:
      self.model_ = FaceDetectCascade()
    elif self.type_ == FaceDetectType.Dnn:
      self.model_ = FaceDetectDnn()



