import abc
from ..RectUtility import *

class RegionBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.region_ = (0,0,0,0)

    @abc.abstractmethod
    def getCvRect(self):
        pass

    @abc.abstractmethod
    def setCvRect(self,rect):
        pass

    def get2Points(self):
        return RectUtility.rectTo2Points(self.getCvRect())
