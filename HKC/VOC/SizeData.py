from xml.etree import ElementTree

class SizeData:

    def __init__(self,width=224,height=224,depth= 3):
      self.width_ = width
      self.height_ = height
      self.depth_ = depth

    def set(self,width,height,depth):
      self.width_ = width
      self.height_ = height
      self.depth_ = depth
    def setImage(self,image):
      self.width_ = image.shape[1]
      self.height_ = image.shape[0]
      self.depth_ = image.shape[2]

    def read(self,data):
        self.width_ = int(data.find('width').text)
        self.height_ = int(data.find('height').text)
        self.depth_ = int(data.find('depth').text)

    def write(self,data):
        ElementTree.SubElement(data, "width").text =str( self.width_)
        ElementTree.SubElement(data, "height").text = str(self.height_)
        ElementTree.SubElement(data, "depth").text = str(self.depth_)
