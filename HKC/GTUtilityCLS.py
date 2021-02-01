from .FileUtility import *
from .CvUtility import *
from .GTUtility import *

import enum
from tqdm import tqdm
import tempfile
import shutil
import zipfile
import  cv2
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

import os
import io
import pandas as pd
import tensorflow as tf

# from PIL import Image
# from object_detection.utils import dataset_util
# from collections import namedtuple, OrderedDict


class GTUtilityClS:
    @staticmethod
    def getFolderInfo():
        filenames = []
        labels = []
        return filenames, labels

    @staticmethod
    def saveXML(filesname, labels, xml_filename):
        pass

    @staticmethod
    def splitGT(filenames, labels):
        train_filenames, train_labels, test_filenames, test_labels
        return train_filenames, train_labels, test_filenames, test_labels

    @staticmethod
    def folder2XML(src_path, dst_path, train_per=0.8):
        filenames, labels = GTUtilityClS.getFolderInfo(src_path)
        train_filenames, train_labels, test_filenames, test_labels = GTUtilityClS.splitGt(filenames,labels,train_per)

        train_xml_filename = os.path.join(dst_path,'train.xml')
        test_xml_filename = os.path.join(dst_path, 'test.xml')

        GTUtilityClS.saveXML(train_filenames, train_labels, train_xml_filename)
        GTUtilityClS.saveXML(test_filenames, test_labels, test_xml_filename)

    @staticmethod
    def splitFolderImageFiles(src_path,train_per = 0.8):
        
        src_files = FileUtility.getFolderImageFiles(src_path)
        train_indexs,val_indexs =  GTUtility.getGTRandomIndexs(len(src_files),train_per)
        train_files = Utility.getListByIndexs(src_files,train_indexs)
        val_files = Utility.getListByIndexs(src_files, val_indexs)

        return train_files,val_files

        

    @staticmethod
    def splitGTFolder(src_path,dst_path,train_per = 0.8):
        labels = ['train', 'validation']

        for label in labels :
            cur_dst_path = os.path.join(dst_path,label)
            FileUtility.copyFullSubFolders(src_path,cur_dst_path)


        sub_folders = FileUtility.getSubfolders(src_path)

        for sub_folder in sub_folders:
           cur_src_path = os.path.join(src_path,sub_folder)

           train_filesname,val_filesname =  GTUtilityClS.splitFolderImageFiles(cur_src_path,train_per)
           files_list = [train_filesname,val_filesname]


           for i,label in enumerate(labels):
               cur_dst_path = os.path.join(dst_path,label)
               # cur_dst_path = os.path.join(cur_dst_path,sub_folder)

               cur_src_files  = files_list[i]
               cur_dst_files = FileUtility.getDstFilenames2(cur_src_files,src_path,cur_dst_path)

               FileUtility.copyFilesByName(cur_src_files,cur_dst_files)





