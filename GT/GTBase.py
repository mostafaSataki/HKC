import abc
from .GTData import  *
import cv2

class GTBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def load(self, filename):
        pass

    @abc.abstractmethod
    def save(self, filename = None):
        pass

    @abc.abstractmethod
    def new(self,image_filename):
        image = cv2.imread(image_filename)
        self.data_.size_ = (image.shape[1],image.shape[0])

    # @abc.abstractmethod
    # def add(self,region,label_id):
    #     self.data_.add(region,label_id)

    def getObjectsCount(self):
        return len(self.data_.objects_)

    def getObjectsRegions(self):
        result = []
        for obj in self.data_.objects_ :
            result.append(obj.region_.getCvRect())
        return result

    # def getObjectsRegionsLabels(self):
    #     regions = []
    #     labels = []
    #
    #
    #
    #     for obj in self.data_.objects_:
    #         regions.append(obj.region_.getCvRect())
    #         labels.append(obj.label_id_)
    #     return regions,labels

