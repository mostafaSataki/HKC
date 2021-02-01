from .RegionBase import *
from xml.etree  import ElementTree

class VocRegion(RegionBase):
    def getCvRect(self):
        return (self.region_[0],self.region_[1],self.region_[2] - self.region_[0],self.region_[3] - self.region_[1])

    def setCvRect(self,rect):
        self.region_ = (rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3])

    def str(self):
        pass
        # return "[{},{},{},{}]".format(self.region_[0], self.region_[1], self.width(), self.height())
    def read(self,data):
        r = []
        r.append(int(data.find('xmin').text))
        r.append(int(data.find('ymin').text))
        r.append(int(data.find('xmax').text))
        r.append(int(data.find('ymax').text))
        self.region_ = tuple(r)

    def write(self,data):
        ElementTree.SubElement(data, "xmin").text = str(self.region_[0])
        ElementTree.SubElement(data, "ymin").text = str(self.region_[1])
        ElementTree.SubElement(data, "xmax").text = str(self.region_[2])
        ElementTree.SubElement(data, "ymax").text = str(self.region_[3])

