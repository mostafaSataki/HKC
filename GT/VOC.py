from .GTBase import *
from .GtVocData import  *
import cv2
from xml.etree.ElementTree import Element, SubElement
from  .Labels import  *

class VOC(GTBase):
    def __init__(self):
        super(VOC, self).__init__()
        self.data_ = GTVOCData()
        self.labels_ = Labels()

    def load(self, xml_file_name):
        self.data_ = GTVOCData()
        self.data_.read(xml_file_name)

    def save(self, xml_file_name=None):
        self.data_.write(xml_file_name)

    def new(self,image_filename,image_shape = None):
        self.data_ = GTVOCData()
        self.data_.new(image_filename,image_shape)

    def add(self,region,label):
        self.data_.add(region,label)

    def getObjectsRegionsLabels(self):
        regions = []
        labels = []

        for obj in self.data_.objects_:
            regions.append(obj.region_.getCvRect())
            labels.append(obj.name_)
        return regions, labels




