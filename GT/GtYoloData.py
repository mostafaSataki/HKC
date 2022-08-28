from .GTData import  *
from .YoloRegion import  *
import os
from  ..FileUtility import  *
import cv2


class GtYoloItem( GTItem):
    def __init__(self,image_size):
        self.label_id_ = 0
        self.region_ = YoloRegion(image_size)

    def clone(self):
        result = GtYoloItem(self.region_.image_size_)
        result.region_ = self.region_.clone()
        result.label_id_ = self.label_id_
        return result

    def toStr(self):
        return str(self.label_id_)+' '+ self.region_.str()

    def fromStr(self,str):
        tokens = str.split(' ')
        self.label_id_ = int(tokens[0])
        self.region_.region_ = (float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]))


class GtYoloData(GTData):

    def __init__(self,image_shape =(240,320,3)):
        super(GtYoloData, self).__init__()
        self.size_ = image_shape


    def setImageSize(self,size):
        self.size_ = size

    def readFromFile(self,filename,defaul_size = ()):
        if not os.path.exists(filename):
            return

        image_filename = FileUtility.getImagePair(filename)
        if image_filename == None:
            if defaul_size != ():
               self.size_ = defaul_size
            else :  return
        else :
            image = cv2.imread(image_filename)
            if np.shape(image) != ():
                self.size_ = image.shape
            else :return
        self.clear()
        self.filename_ = filename
        with open(filename,'r') as file :
            lines = file.readlines()
            for line in lines:
               item = GtYoloItem(self.size_)
               item.fromStr(line)
               self.objects_.append(item)
            file.close()

    def writeToFile(self,filename = None):
        if filename == None and not self.filename_:
            return

        if filename != None:
            self.filename_ = filename

        lines = []
        for item in self.objects_:
            lines.append(item.toStr()+'\n')

        with open(self.filename_,'w') as file :
           file.writelines(lines)
           file.close()


    def addYoloRegion(self, region, label_id):
        item = GtYoloItem(self.size_)
        item.label_id_ = label_id
        item.region_.region_  = region
        self.objects_.append(item)

    def addCvRect(self,rect,label_id):
        item = GtYoloItem(self.size_)
        item.label_id_ = label_id
        item.region_.setCvRect(rect)
        self.objects_.append(item)

    def clone(self):
        result = GtYoloData()
        result.size_ = self.size_
        result.filename_ = self.filename_

        for obj in self.objects_:
            result.objects_.append(obj.clone())

        return result
