from .GTBase import *
from .GtYoloData import  *
from .Labels import  *
import  cv2
from HKC.FileUtility import *
import numpy as np
from tqdm import tqdm
from shutil import copyfile


class YOLO(GTBase):
    def __init__(self,defaul_size = ()):
        super(YOLO, self).__init__()
        self.data_ = GtYoloData()
        self.labels_ = Labels()
        self.default_size_ = defaul_size

    def load(self, filename):
        self.data_.readFromFile(filename,self.default_size_)

    def save(self, filename=None):
        self.data_.writeToFile(filename)

    def newByImageSize(self, image_size):
        self.data_ = GtYoloData(image_size)

    def new(self,filename):
        image = cv2.imread(filename)
        if np.shape(image) != ():
            self.newByImageSize(image.shape)
            self.data_.filename_ = FileUtility.changeFileExt(filename, 'txt')

    def addByIndex(self,region,index):
        self.data_.addCvRect(region,index)

    def addByLable(self,region,label):
        self.labels_.add(label)
        index = self.labels_.getIndex(label)
        self.addByIndex(region,index)

    def getObjectsRegionsLabels(self):
        regions = []
        labels = []

        for obj in self.data_.objects_:
            regions.append(obj.region_.getCvRect())
            labels.append(obj.label_id_)
        return regions, labels


class YOLOFolder :
    @staticmethod
    def createYoloBackgroundFolder(src_path,dst_path):
        src_image_files = FileUtility.getFolderImageFiles(src_path)


        dst_image_files = FileUtility.getDstFilenames2(src_image_files,src_path,dst_path)
        dst_gt_files = FileUtility.changeFilesExt(dst_image_files,'txt')

        for i in tqdm(range(len(src_image_files)), ncols=100):
            src_image_file = src_image_files[i]


            dst_image_file = dst_image_files[i]
            dst_gt_file = dst_gt_files[i]

            copyfile(src_image_file, dst_image_file)
            with open(dst_gt_file,'w')  as f :
                f.write('')
                f.close()


    @staticmethod
    def createYoloFolder(src_path,dst_path, label_index ,label):
        src_image_files = FileUtility.getFolderImageFiles(src_path)

        dst_image_files = FileUtility.getDstFilenames2(src_image_files,src_path,dst_path)
        dst_image_files = FileUtility.changeFilesnamePrefix(dst_image_files,label+"_")

        dst_gt_files =  FileUtility.changeFilesExt(dst_image_files,'txt')


        FileUtility.copyFilesByName(src_image_files,dst_image_files)


        for i in tqdm(range(len(dst_image_files)), ncols=100):
            dst_image_file = dst_image_files[i]
            dst_gt_file = dst_gt_files[i]

            src_image = cv2.imread(dst_image_file)

            yolo_rct = [0.5,0.5,1.0,1.0   ]

            with open(dst_gt_file,'w') as f:
                s = str(label_index) + ' '
                count = 4
                sep = ' '
                for i in range(count):
                    s += str(yolo_rct[i])
                    if i < count - 1:
                        s += sep
                f.write(s+'\n')
        f.close()





    @staticmethod
    def createYoloRootFolder(src_path,dst_path):

        sub_folders = FileUtility.getSubfolders(src_path)
        FileUtility.createClearFolder(dst_path)
        classes_file = os.path.join(dst_path,'classes.txt')
        with open(classes_file,'w') as f:
            label_index = 0
            for sub_folder in sub_folders:
                src_cur_path = os.path.join(src_path,sub_folder)
                dst_cur_path = os.path.join(dst_path, sub_folder)

                if sub_folder == "0":
                    YOLOFolder.createYoloBackgroundFolder(src_cur_path,dst_path)
                else :
                    f.write(sub_folder+'\n')
                    YOLOFolder.createYoloFolder(src_cur_path,dst_path,label_index,sub_folder)
                    label_index += 1



        f.close()


