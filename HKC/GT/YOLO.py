from .GTBase import *
from .GtYoloData import  *
from .Labels import  *
import  cv2
from HKC.FileUtility import *
import numpy as np

class YOLO(GTBase):
    def __init__(self):
        super(YOLO, self).__init__()
        self.data_ = GtYoloData()
        self.labels_ = Labels()

    def load(self, filename):
        self.data_.readFromFile(filename)

    def save(self, filename=None):
        self.data_.writeToFile(filename)

    def newByImageSize(self, image_size):
        self.data_ = GtYoloData(image_size)

    def new(self,filename):
        image = cv2.imread(filename)
        if np.shape(image) != ():
            self.newByImageSize(image.shape)
            self.data_.filename_ = FileUtility.changeFileExt(filename, 'txt')

    def addByIndex(self,region,index):
        self.data_.addCvRect(region,index)

    def addByLable(self,region,label):
        self.labels_.add(label)
        index = self.labels_.getIndex(label)
        self.addByIndex(region,index)



