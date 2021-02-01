import abc

class RegionBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.region_ = (0,0,0,0)

    @abc.abstractmethod
    def getCvRect(self):
        pass
    def setCvRect(self,rect):
        pass
