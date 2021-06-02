from .GTData import *
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
from .VocRegion import  *
import codecs
from  ..FileUtility import  *
import  cv2


class GtVocItem( GTItem):
    def __init__(self):
        self.region_ = VocRegion()
        self.name_ = ""
        self.pose_ = "Unspecified"
        self.truncated_ = 0
        self.difficult_ = 0

    def read(self,data):
        self.name_ = data.find('name').text
        self.pose_ = data.find('pose').text
        self.truncated_ = int(data.find('truncated').text)
        self.difficult_ = int(data.find('difficult').text)
        self.region_.read(data.find('bndbox'))

    def write(self, data):
        ElementTree.SubElement(data, "name").text = self.name_
        ElementTree.SubElement(data, "pose").text = self.pose_
        ElementTree.SubElement(data, "truncated").text = str(self.truncated_)
        ElementTree.SubElement(data, "difficult").text = str(self.difficult_)
        bndbox = ElementTree.SubElement(data, "bndbox")
        self.region_.write(bndbox)


class GTVOCData(GTData):
    def __init__(self):
        super(GTVOCData, self).__init__()
        self.folder_ = ""
        self.path_ = ""
        self.segmented_ = 0
        self.database_ = 'Unknown'

    def _readSize(self,data):
        s = list(self.size_)
        s[1] = int(data.find('width').text)
        s[0] = int(data.find('height').text)
        s[2] = int(data.find('depth').text)
        self.size_ = tuple(s)

    def _writeSize(self,data):
        ElementTree.SubElement(data, "width").text = str(self.size_[1])
        ElementTree.SubElement(data, "height").text = str(self.size_[0])
        ElementTree.SubElement(data, "depth").text = str(self.size_[2])

    def _readObjects(self,xml):
        objects = xml.findall('object')
        for obj in objects:
            item = GtVocItem()
            item.read(obj)
            self.objects_.append(item)

    def read(self, xml_fileanme):
        xml = ElementTree.parse(xml_fileanme)
        self.folder_ = xml.find('folder').text
        self.filename_ = xml.find('filename').text
        self.path_ = xml.find('path').text
        self.segmented_ = xml.find('segmented').text
        self.database_ = xml.find('source').find('database').text
        self._readSize(xml.find('size'))
        self._readObjects(xml)

    def _prettify(self, elem):
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding='utf-8').replace("  ".encode(), "\t".encode())

    def _writeObjects(self,top):
        pass

    def _generateXML(self):
        top = Element('annotation')

        SubElement(top, 'folder').text = self.folder_
        SubElement(top, 'filename').text = self.filename_
        SubElement(top, 'path').text = self.path_
        ElementTree.SubElement(SubElement(top, 'source'), "database").text = self.database_
        self._writeSize(SubElement(top, 'size'))
        SubElement(top, 'segmented').text = str(self.segmented_)


        for obj in self.objects_:
           obj.write(SubElement(top, 'object'))



        return top

    def write(self,xml_filename):
        root = self._generateXML()

        if xml_filename == None:
            xml_filename = FileUtility.changeFileExt(self.path_,'xml')
        out_file = codecs.open(xml_filename, 'w', encoding='utf-8')

        prettifyResult = self._prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

    def new(self,image_filename,image_shape = None):
        self.filename_ = FileUtility.getFilename(image_filename)
        self.folder_ = FileUtility.upFolderName(image_filename)
        self.path_ = image_filename
        self.segmented_ = 0
        self.database_ = 'Unknown'

        if image_shape == None:
           image = cv2.imread(image_filename)
           self.size_ = image.shape
        else : self.size_ = image_shape



    def add(self, region, label):
        item = GtVocItem()
        item.region_.setCvRect(region)
        item.name_ = label
        self.objects_.append(item)
