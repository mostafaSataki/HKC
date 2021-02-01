from  ..FileUtility import  *

import sys
import glob
import  os
import re
import  random
import  shutil
import  cv2
from .AnnotationData import  *

import shutil
import argparse
import  math





class sizeSolver:
    def solve(self, src_image_file_name, src_xml_file_name,dst_image_file_name,dst_xml_file_name):
        self.src_image_file_name_ = src_image_file_name
        self.src_xml_file_name_ = src_xml_file_name
        self.dst_image_file_name_ = dst_image_file_name
        self.dst_xml_file_name_ = dst_xml_file_name


        self.image_ = cv2.imread(src_image_file_name,1)
        self.image_size_ = self.image_.shape[:2]
        self.min_size_ = (33.0,33.0)

        self.annotation_ = AnnotationData()
        self.annotation_.read(src_xml_file_name)

        self.xml_min_size_ =  self.annotation_.getObjectsMinSize()
        if (self.xml_min_size_[0] < self.min_size_[0] or self.xml_min_size_[1] < self.min_size_[1]):
           self.setScale()
           self.saveDst()
        else :
            shutil.copy(self.src_image_file_name_, dst_image_file_name)
            shutil.copy(self.src_xml_file_name_, dst_xml_file_name)

    def resize(self, src_image_file_name, src_xml_file_name, dst_image_file_name, dst_xml_file_name,scale):
      self.src_image_file_name_ = src_image_file_name
      self.src_xml_file_name_ = src_xml_file_name
      self.dst_image_file_name_ = dst_image_file_name
      self.dst_xml_file_name_ = dst_xml_file_name

      self.image_ = cv2.imread(src_image_file_name, 1)
      self.image_size_ = self.image_.shape[:2]
      self.min_size_ = (33.0, 33.0)

      self.annotation_ = AnnotationData()
      self.annotation_.read(src_xml_file_name)

      self.xml_min_size_ = self.annotation_.getObjectsMinSize()
      self.scale_ = scale
      self.saveDst()

    def setScale(self):
        scale = [1,1]
        if self.min_size_[0] > self.xml_min_size_[0]:
            scale[0] = self.min_size_[0] / self.xml_min_size_[0]
        if self.min_size_[1] > self.xml_min_size_[1]:
            scale[1] = self.min_size_[1] / self.xml_min_size_[1]

        self.scale_ = max(scale[0],scale[1])


    def setImageNewSize(self):
        self.image_new_size_ = (int(self.image_size_[1] * self.scale_),int( self.image_size_[0] * self.scale_))

    def saveDstImage(self):
        self.setImageNewSize()
        img2 = cv2.resize(self.image_, self.image_new_size_)
        quality = 75

        cv2.imwrite(self.dst_image_file_name_,img2, [cv2.IMWRITE_JPEG_QUALITY, quality])

    def saveDstXml(self):
        self.annotation_.doScale(self.scale_)
        self.annotation_.write(self.dst_xml_file_name_)




    def saveDst(self):
        self.saveDstImage()
        self.saveDstXml()







