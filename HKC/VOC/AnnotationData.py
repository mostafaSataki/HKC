
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from .SourceData import  *
from  .SizeData import  *
from .ObjectData import  *
import  os

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class AnnotationData:
    @property
    def folder(self):
      return self._folder

    @folder.setter
    def folder(self, value):
      self._folder = value

    @property
    def filename(self):
      return self._filename

    @filename.setter
    def filename(self, value):
      self._filename = value

    @property
    def path(self):
      return self._path

    @filename.setter
    def filename(self, value):
      self._path = value

    @property
    def source(self):
      return self._source.database

    @source.setter
    def source(self, value):
      self._source.database = value

    @property
    def segmented(self):
      return self._segmented

    @source.setter
    def segmented(self, value):
      self._segmented = value

    @property
    def segmented(self):
      return self._n

    @source.setter
    def segmented(self, value):
      self._segmented = value


    def __init__(self):
      self._folder = ""
      self._filename = ""
      self._path = ""
      self._source = SourceData()
      self._segmented = 0
      self._size = SizeData()
      self.objects = []

    def setFilePath(self,filename):
      path,f_name = os.path.split(filename)
      self._filename = f_name
      self._path = path

    def addObject(self, region, object_class):
      obj = ObjectData(object_class, region)
      self.objects.append(obj)

    def getObjectsCount(self):
        return len(self.objects)



    def read(self,xml_file_name):
        xmlFile = ElementTree.parse(xml_file_name)
        self._folder = xmlFile.find('folder').text
        self._filename = xmlFile.find('filename').text
        self._path = xmlFile.find('path').text
        # self.source_ = SourceData()
        self._source.read( xmlFile.find('source'))
        self._segmented = xmlFile.find('segmented').text
        # self.size_ = SizeData()
        self._size.read(xmlFile.find('size'))

        objects = xmlFile.findall('object')
        # self.objects_ = []
        for obj in objects:

            cur_obj = ObjectData()
            cur_obj.read(obj)
            self.objects.append(cur_obj)

    def getObjectsMinSize(self):
        max_value = 100000
        result = [max_value, max_value]

        for obj in self.objects:
            x_value = obj.bndbox_.x_max_  - obj.bndbox_.x_min_
            y_value = obj.bndbox_.y_max_ - obj.bndbox_.y_min_

            if (x_value < result[0]):
                result[0] = x_value

            if (y_value < result[1]):
                result[1] = y_value

        return  result
    def doScale(self,scale):
        self._size.width_ = int( self._size.width_ * scale)
        self._size.height_ = int( self._size.height_ * scale)

        for obj in self.objects:
            obj.bndbox_.x_max_ = int(obj.bndbox_.x_max_ * scale)
            obj.bndbox_.x_min_ = int(obj.bndbox_.x_min_* scale)
            obj.bndbox_.y_max_ = int(obj.bndbox_.y_max_ * scale)
            obj.bndbox_.y_min_ = int(obj.bndbox_.y_min_ * scale)

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''


    def genXML(self):
        """
            Return XML root
        """
        # Check conditions

        top = Element('annotation')
        # if self.verified:
        #     top.set('verified', 'yes')

        SubElement(top, 'folder').text = self._folder
        SubElement(top, 'filename').text = self._filename
        SubElement(top, 'path').text =  self._path
        source = SubElement(top, 'source')
        self._source.write(source)

        size = SubElement(top, 'size')
        self._size.write(size)
        SubElement(top, 'segmented').text = self._segmented
        for obj in self.objects:
            cur_obj = SubElement(top, 'object')
            obj.write(cur_obj)

        return top

    def write(self, xml_file_name):
        root = self.genXML()
        # self.appendObjects(root)
        out_file = None
        out_file = codecs.open(xml_file_name, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()
