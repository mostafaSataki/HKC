from xml.etree import ElementTree

class BndboxData:

  def __init__(self,region = None):
    if region == None:
      self.x_min_ = 0
      self.y_min_ = 0
      self.x_max_ = 0
      self.y_max_ = 0
    else :
      self.x_min_ =  region[0]
      self.y_min_ =  region[1]
      self.x_max_ = region[0] + region[2]
      self.y_max_ = region[1] + region[3]

  def str(self):
    return "[{},{},{},{}]".format(self.x_min_,self.y_min_,self.width(),self.height())

  def width(self):
    return self.x_max_ - self.x_min_ + 1

  def height(self):
    return self.y_max_ - self.y_min_ + 1


  def read(self, data):
    self.x_min_ = int(data.find('xmin').text)
    self.y_min_ = int(data.find('ymin').text)
    self.x_max_ = int(data.find('xmax').text)
    self.y_max_ = int(data.find('ymax').text)

  def write(self, data):
    ElementTree.SubElement(data, "xmin").text = str(self.x_min_)
    ElementTree.SubElement(data, "ymin").text = str(self.y_min_)
    ElementTree.SubElement(data, "xmax").text = str(self.x_max_)
    ElementTree.SubElement(data, "ymax").text = str(self.y_max_)
