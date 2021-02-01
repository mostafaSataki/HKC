from .BndboxData import  *

class ObjectData:

  def __init__(self,name="",region=None):
    self.name_ = name
    self.pose_ = "Unspecified"
    self.truncated_ = 0
    self.difficult_ = 0
    self.bndbox_ = BndboxData(region)



  def read(self, data):
    self.name_ = data.find('name').text
    self.pose_ = data.find('pose').text
    self.truncated_ = int(data.find('truncated').text)
    self.difficult_ = int(data.find('difficult').text)
    self.bndbox_ = BndboxData()
    self.bndbox_.read(data.find('bndbox'))

  def write(self, data):
    ElementTree.SubElement(data, "name").text = self.name_
    ElementTree.SubElement(data, "pose").text = self.pose_
    ElementTree.SubElement(data, "truncated").text = str(self.truncated_)
    ElementTree.SubElement(data, "difficult").text = str(self.difficult_)
    bndbox = ElementTree.SubElement(data, "bndbox")
    self.bndbox_.write(bndbox)