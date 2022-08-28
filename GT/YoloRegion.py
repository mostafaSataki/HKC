from .RegionBase import *

class YoloRegion(RegionBase):

    def __init__(self,image_size):
        super(YoloRegion, self).__init__()
        self.image_size_ = image_size


    def getCvRect(self):
        return ( round( (self.region_[0] - (self.region_[2] / 2) ) * self.image_size_[1]),
                 round((self.region_[1] - (self.region_[3] / 2) ) * self.image_size_[0]),
                 round(self.region_[2] * self.image_size_[1]),round( self.region_[3] * self.image_size_[0]))

    def setCvRect(self,rect):
        r = (float(rect[0]) / self.image_size_[1], float(rect[1]) / self.image_size_[0],
             float(rect[2]) / self.image_size_[1], float(rect[3]) / self.image_size_[0])
        self.region_ = (r[0] + r[2] / 2, r[1] + r[3] / 2, r[2], r[3])

    def str(self):
        return ' '.join([str(v) for v in self.region_])

    def clone(self):
        result = YoloRegion(self.image_size_)
        result.region_ = self.region_
        result.image_size_ = self.image_size_
        return result

