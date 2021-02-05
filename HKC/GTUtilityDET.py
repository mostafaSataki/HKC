from .FileUtility import *
from .CvUtility import *
from .RectUtility import *
from  .GTUtility import *
from .GT.VOC import *
from .GT.YOLO import *
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

import  tempfile

# from PIL import Image
from object_detection.utils import dataset_util
# from collections import namedtuple, OrderedDict


class GTFormat(enum.Enum):
    YOLO = 1
    VOC = 2
    
  
class CopyMethod(enum.Enum):
   stretch = 1;
   valid = 2;
   expand = 3

class GTUtilityDET:


  @staticmethod
  def getGtFolderFormat(gt_path):
      image_filesname = FileUtility.getFolderImageFiles(gt_path)
      if len(image_filesname) == 0 :
          return None

      image_filename = image_filesname[0]
      voc_filename = FileUtility.changeFileExt(image_filename,'xml')
      if os.path.exists(voc_filename):
          return GTFormat.VOC

      yolo_filename = FileUtility.changeFileExt(image_filename,'txt')
      if os.path.exists(yolo_filename):
          return  GTFormat.YOLO

      return None
  @staticmethod
  def getGtFileFormat(gt_filename):
      result = None
      ext = FileUtility.getFileExt(gt_filename)
      if ext == 'txt':
          result = GTFormat.YOLO
      elif ext == 'voc':
          result = GTFormat.VOC
      return  result

  @staticmethod
  def getGtExt(format):
      if format == GTFormat.YOLO:
          return 'txt'
      elif format == GTFormat.VOC:
          return  'xml'

  @staticmethod
  def allGTFormats():
      return [GTFormat.YOLO,GTFormat.VOC]

  @staticmethod
  def getImageGtPair(image_filename,gt_format = None):
      tokens = FileUtility.getFileTokens(image_filename)
      if gt_format == None :
          all_formats = GTUtilityDET.allGTFormats()

          for format in all_formats :
             gt_filename = os.path.join(tokens[0],tokens[1]+'.'+GTUtilityDET.getGtExt(format))
             if os.path.exists(gt_filename):
               return gt_filename,format

      else: return  os.path.join(tokens[0],tokens[1]+'.'+GTUtilityDET.getGtExt(gt_format)),gt_format


  @staticmethod
  def _convertYolo2VocPath(src_path, dst_path):
      if not os.path.exists(dst_path):
          os.makedirs(dst_path)
      src_image_filesname = FileUtility.getFolderImageFiles(src_path)

      src_yolo = YOLO()
      src_yolo.labels_.load(src_path)

      dst_voc = VOC()

      for i in tqdm(range(1, len(src_image_filesname)), ncols=100):
          src_image_filename = src_image_filesname[i]
          src_gt_filename = FileUtility.changeFileExt(src_image_filename, 'txt')

          dst_image_filename = FileUtility.getDstFilename2(src_image_filename, src_path, dst_path)

          FileUtility.copyFile(src_image_filename, dst_image_filename)
          src_yolo.load(src_gt_filename)
          dst_voc.new(dst_image_filename)

          for obj in src_yolo.data_.objects_:
              label = src_yolo.labels_.getLabel(obj.label_id_)
              dst_voc.add(obj.region_.getCvRect(), label)

          dst_voc.save()


  @staticmethod
  def convertYolo2Voc(src_path, dst_path):
      if FileUtility.checkRootFolder(src_path):
          GTUtilityDET._convertYolo2VocPath(src_path,dst_path)
      else :
          sub_folders = FileUtility.getSubfolders(src_path)
          for sub_folder in sub_folders :
              GTUtilityDET.convertYolo2Voc(os.path.join(src_path,sub_folder),os.path.join(dst_path,sub_folder))


  @staticmethod
  def convertVoc2Yolo(src_path, dst_path):
      src_image_filesname = FileUtility.getFolderImageFiles(src_path)

      src_voc = VOC()
      # src_voc.labels_.load(src_path)

      dst_yolo = YOLO()

      for i, src_image_filename in enumerate(src_image_filesname):
          src_gt_filename = FileUtility.changeFileExt(src_image_filename, 'xml')

          dst_image_filename = FileUtility.getDstFilename2(src_image_filename, src_path, dst_path)

          FileUtility.copyFile(src_image_filename, dst_image_filename)
          src_voc.load(src_gt_filename)
          dst_yolo.new(dst_image_filename)

          for obj in src_voc.data_.objects_:
              dst_yolo.addByLable(obj.region_.getCvRect(), obj.name_)

          dst_yolo.save()

      dst_yolo.labels_.save(dst_path)

  @staticmethod
  def convertGt(src_path,dst_path,src_format = None,dst_format=None):
      if src_format == None:
          src_format = getGtFolderFormat(src_path)
      if dst_format == None:
          dst_format = getGtFolderFormat(dst_path)

      if src_format == None or dst_format == None:
          return

      if src_format == GTFormat.YOLO and dst_format == GTFormat.VOC:
          GTUtilityDET.convertYolo2Voc(src_path,dst_path)
      elif src_format == GTFormat.VOC and dst_format == GTFormat.YOLO:
          GTUtilityDET.convertVoc2Yolo(src_path,dst_path)



  @staticmethod
  def getObjectsCount(gt_filename):
      result = 0
      gt_format = GTUtilityDET.getGtFileFormat(gt_filename)
      if gt_format == None:
          return result

      if gt_format == GTFormat.YOLO:
          gt = YOLO()
      elif gt_format == GTFormat.VOC:
          gt = VOC()
      gt.load(gt_filename)
      return gt.getObjectsCount()

      
  @staticmethod
  def getObjectsRegions(gt_filename):
      result = []
      gt_format = GTUtilityDET.getGtFileFormat(gt_filename)
      if gt_format == None:
          return result

      if gt_format == GTFormat.YOLO:
          gt = YOLO()
      elif gt_format == GTFormat.VOC:
          gt = VOC()
      gt.load(gt_filename)
      return gt.getObjectsRegions()


  @staticmethod
  def removeBlankGT(path):
    gt_format = GTUtilityDET.getGtFolderFormat(path)
    image_filesname = FileUtility.getFolderImageFiles(path)

    empty_count = 0
    for i in tqdm(range(len(image_filesname)), ncols=100):
        image_filename = image_filesname[i]
        gt_filename,_ = GTUtilityDET.getImageGtPair(image_filename,gt_format)

        if not os.path.exists(gt_filename) or GTUtilityDET.getObjectsCount(gt_filename) == 0 :
            if os.path.exists(gt_filename):
              os.remove(gt_filename)
            os.remove(image_filename)
            empty_count += 1

    print("empty count:", empty_count)




  @staticmethod
  def mergeAllGTFiles(zip_path, dst_path, prefix,start_counter = 0, remove_blank_gt=True, recompress_qulaity=30,pad_count = 7,
                      clear_similar_frames=True,psnr_thresh = 19,distance_thresh = 100,area_thresh = 100):

      if clear_similar_frames :
          remove_blank_gt = True
      zip_files = FileUtility.getFolderFiles(zip_path, 'zip')
      
      FileUtility.copyFullSubFolders(zip_path,dst_path,True)


      counter = 0
      for i in tqdm(range(len(zip_files)), ncols=100):
          zip_file = zip_files[i]
          temp_path = tempfile.mkdtemp()


          with zipfile.ZipFile(zip_file, 'r') as zip_ref:
              zip_ref.extractall(temp_path)

          gt_format = GTUtilityDET.getGtFolderFormat(temp_path)
          gt_ext = GTUtilityDET.getGtExt(gt_format)

          if remove_blank_gt:
              GTUtilityDET.removeBlankGT(temp_path)

          if recompress_qulaity != None:
              CvUtility.recompressImages(temp_path)

          if clear_similar_frames :
              GTUtilityDET.clearSimilarGTFrames(temp_path,psnr_thresh,distance_thresh ,area_thresh)
              GTUtilityDET.removeBlankGT(temp_path)

          src_image_files = FileUtility.getFolderImageFiles(temp_path)
          src_gt_files = FileUtility.changeFilesExt(src_image_files, gt_ext)

          dst_image_files,_ = FileUtility.changeFilesname(src_image_files, prefix, counter,pad_count)
          dst_gt_files,_ = FileUtility.changeFilesname(src_gt_files, prefix, counter,pad_count)

          cur_dst_path = FileUtility.file2Folder(FileUtility.getDstFilename2(zip_file, zip_path, dst_path))

          dst_image_files = FileUtility.getDstFilenames2(dst_image_files, temp_path, cur_dst_path,True)
          dst_gt_files = FileUtility.getDstFilenames2(dst_gt_files, temp_path, cur_dst_path,True)

          FileUtility.copyFilesByName(src_image_files, dst_image_files)
          FileUtility.copyFilesByName(src_gt_files, dst_gt_files)

          counter += len(dst_gt_files)

          FileUtility.deleteFolderContents(temp_path)

          # shutil.rmtree(temp_path)

  @staticmethod
  def getGTData(image_filename):
      gt_filename,_ = GTUtilityDET.getImageGtPair(image_filename)
      image = cv2.imread(image_filename)
      regions = GTUtilityDET.getObjectsRegions(gt_filename)
      return image,regions


  @staticmethod
  def similartGT(image1_filename,image2_filename,psnr_thresh = 24,distance_thresh = 100,area_thresh = 100):
      image1,regions1 = GTUtilityDET.getGTData(image1_filename)
      image2, regions2 = GTUtilityDET.getGTData(image2_filename)
      if len(regions1) == 0 or len(regions2) == 0:
          return
      return CvUtility.similariy(image1,regions1[0],image2,regions2[0])

  @staticmethod
  def clearSimilarGTFrames(src_path,psnr_thresh = 19,distance_thresh = 100,area_thresh = 100):
    image_filesname = FileUtility.getFolderImageFiles(src_path)
    ref_id = 0
    ref_image_filename = image_filesname[ref_id]

    ref_img, ref_regions = GTUtilityDET.getGTData(ref_image_filename)

    for i in tqdm(range(1,len(image_filesname)), ncols=100):
        cur_image_filename = image_filesname[i]
        cur_img ,cur_regions = GTUtilityDET.getGTData(cur_image_filename)
        is_similar = CvUtility.similariy(ref_img,ref_regions[0], cur_img,cur_regions[0])
        if not is_similar:
          ref_image_filename = cur_image_filename
          ref_img = cur_img
          ref_regions = cur_regions
        else : os.remove(image_filesname[i])

  @staticmethod
  def createGT(format,image,regions,labels):
      pass

  @staticmethod
  def loadGT(image_filename):
      gt_filename, gt_format = GTUtilityDET.getImageGtPair(image_filename)
      if gt_format == GTFormat.YOLO:
          gt = YOLO()
      elif gt_format == GTFormat.VOC:
          gt = VOC()

      gt.load(gt_filename)
      
      return gt,gt_filename
      

  @staticmethod
  def flipHorz(image_filename):
      gt,gt_filename = GTUtilityDET.loadGT(image_filename)
      
      image = cv2.imread(image_filename)
      
      for obj in gt.data_.objects_:
          obj.region_.setCvRect( RectUtility.flipHorzRect(obj.region_.getCvRect(),image.shape))

      image = cv2.flip(image,1)
      cv2.imwrite(image_filename,image,[cv2.IMWRITE_JPEG_QUALITY,30])
      gt.save(gt_filename)

  @staticmethod
  def flipHorzBatch(src_path, dst_path, post_fix=""):

      FileUtility.copyFullSubFolders(src_path, dst_path)

      src_image_filesname, src_gt_filesname = GTUtilityDET.getGtFiles(src_path)

      dst_image_filesname = FileUtility.getDstFilenames2(src_image_filesname, src_path, dst_path)
      dst_gt_filesname = FileUtility.getDstFilenames2(src_gt_filesname, src_path, dst_path)

      dst_image_filesname = FileUtility.changeFilesnamePostfix(dst_image_filesname, "_FH")
      dst_gt_filesname = FileUtility.changeFilesnamePostfix(dst_gt_filesname, "_FH")

      FileUtility.copyFilesByName(src_image_filesname,dst_image_filesname)
      FileUtility.copyFilesByName(src_gt_filesname, dst_gt_filesname)


      for i in tqdm(range(1,len(dst_image_filesname)), ncols=100):
          dst_image_filename = dst_image_filesname[i]
          GTUtilityDET.flipHorz(dst_image_filename)


  @staticmethod
  def createGt(image,regions,label,format):
          if format == GTFormat.YOLO:
              gt = YOLO()
          elif format == GTFormat.VOC:
              gt = VOC()



  @staticmethod
  def cropGT(src_path,dst_path):
      src_files = FileUtility.getFolderImageFiles(src_path)
      dst_files = FileUtility.getDstFilenames2(src_files,src_path,dst_path)

      FileUtility.copyFullSubFolders(src_path,dst_path)

      gt_format = GTUtilityDET.getGtFolderFormat(src_path)

      for i in tqdm(range(1, len(src_files)), ncols=100):
          src_image_file = src_files[i]
          dst_image_file = dst_files[i]

          # src_gt_filename = GTUtility.getImageGtPair(src_file,gt_format)
          image,regions = GTUtilityDET.getGTData(src_image_file)

          for j,region in enumerate(regions):
              croped_image =  CvUtility.imageROI(image,region)
              cur_dst_image_filename = dst_image_file
              if j > 0 :
                cur_dst_image_filename = FileUtility.changeFilesnamePostfix(dst_image_file)
              cv2.imwrite(cur_dst_image_filename,croped_image)


  @staticmethod
  def getGtFiles(src_path):
      gt_format = GTUtilityDET.getGtFolderFormat(src_path)
      gt_ext = GTUtilityDET.getGtExt(gt_format)

      image_filenames = FileUtility.getFolderImageFiles(src_path)
      gt_filenames = FileUtility.changeFilesExt(image_filenames,gt_ext)

      return image_filenames,gt_filenames



  @staticmethod
  def getGTFiles(image_filenames,gt_filenames,indexs):
      image_filenames = Utility.getListByIndexs(image_filenames, indexs)
      gt_filenames = Utility.getListByIndexs(gt_filenames, indexs)
      return image_filenames,gt_filenames

  @staticmethod
  def splitGT(src_path,train_per = 0.8):
      image_filenames ,gt_filenames = GTUtilityDET.getGtFiles(src_path)
      train_indexs ,test_indexs = GTUtility.getGTIndexs(len(image_filenames),train_per,IndexType.random)

      train_image_filenames,train_gt_filenames = GTUtilityDET.getGTFiles(image_filenames,gt_filenames,train_indexs)
      test_image_filenames  = Utility.getListByIndexs(image_filenames, test_indexs)
      test_gt_filenames = Utility.getListByIndexs(gt_filenames, test_indexs)

      return train_image_filenames,train_gt_filenames,test_image_filenames,test_gt_filenames


  #
  # @staticmethod
  # def copySplitGT(src_path,dst_path,train_per = 0.8,clear_dst = False,copy_to_root = False):
  #     branchs = ['train','test']
  #
  #     if not os.path.exists(dst_path):
  #         os.mkdir(dst_path)
  #
  #     FileUtility.createDstBrach(dst_path, branchs, clear_dst)
  #     if copy_to_root == False :
  #         for branch in branchs:
  #            FileUtility.copyFullSubFolders(src_path,os.path.join(dst_path,branch))
  #
  #     src_train_image_filenames, src_train_gt_filenames, src_test_image_filenames, src_test_gt_filenames = GTUtilityDET.splitGT(src_path,train_per)
  #
  #     dst_train_image_filenames = FileUtility.getDstFilenames2(src_train_image_filenames,src_path,os.path.join(dst_path,branchs[0]),copy_to_root)
  #     dst_train_gt_filenames = FileUtility.getDstFilenames2(src_train_gt_filenames, src_path, os.path.join(dst_path,branchs[0]),copy_to_root)
  #
  #     dst_test_image_filenames = FileUtility.getDstFilenames2(src_test_image_filenames, src_path, os.path.join(dst_path,branchs[1]),copy_to_root)
  #     dst_test_gt_filenames = FileUtility.getDstFilenames2(src_test_gt_filenames, src_path, os.path.join(dst_path,branchs[1]),copy_to_root)
  #
  #     FileUtility.copyFilesByName(src_train_image_filenames,dst_train_image_filenames)
  #     FileUtility.copyFilesByName(src_train_gt_filenames, dst_train_gt_filenames)
  #     FileUtility.copyFilesByName(src_test_image_filenames, dst_test_image_filenames)
  #     FileUtility.copyFilesByName(src_test_gt_filenames, dst_test_gt_filenames)
  #

  @staticmethod
  def GT2Csv(src_path,csv_filename):
      image_filenames ,gt_filenames = GTUtilityDET.getGtFiles(src_path)
      gt_list = []

      for i in tqdm(range(1, len(image_filenames)), ncols=100):
          image_filename = image_filenames[i]
          gt_filename = gt_filenames[i]
          gt,_ = GTUtilityDET.loadGT(image_filename)


          for obj in gt.data_.objects_:
              r = obj.region_.get2Points()

              value = (gt.data_.filename_,
                       gt.data_.size_[0],gt.data_.size_[1],
                       obj.name_,r[0],r[1],r[2],r[3])
              gt_list.append(value)

      column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
      xml_df = pd.DataFrame(gt_list, columns=column_name)

      xml_df.to_csv(csv_filename, index=None)



  @staticmethod
  def GT2CsvBranchs(src_path,dst_path,clear_dst = False):
      if clear_dst:
         FileUtility.createClearFolder(dst_path)

      if not os.path.exists(dst_path):
         os.mkdir(dst_path)

      sub_folders = FileUtility.getSubfolders(src_path)
      for sub_folder in sub_folders :
         cur_folder = os.path.join(src_path,sub_folder)
         GTUtilityDET.GT2Csv(cur_folder,os.path.join(dst_path,sub_folder+".csv"))

  # @staticmethod
  # def classLabelIndex(row_label,labels):
  #         index = labels.getIndex(row_label)
  #         if index >= 0:
  #             index += 1
  #         else : index = -1
  #
  #         return index
  #
  #
  # @staticmethod
  # def split(df, group):
  #         data = namedtuple('data', ['filename', 'object'])
  #         gb = df.groupby(group)
  #         return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

  @staticmethod
  def getIndex(label,labels):
      index =  Utility.getIndex(class_, labels)
      if index >= 0 :
          index += 1
      return index

  @staticmethod
  def createTFExample(images_path,groups,tf_rec_filename ,labels,branch):

          writer = tf.io.TFRecordWriter(tf_rec_filename)
          xmin = []
          xmax = []
          ymin = []
          ymax = []
          class_ = []
          width = 0
          height = 0
          cur_full_filename = ''
          encoded_jpg  = None
          image_format =  b'jpg'
          filename = ''

          def addSample():
              tf_example = tf.train.Example(features=tf.train.Features(feature={
                  'image/height': dataset_util.int64_feature(height),
                  'image/width': dataset_util.int64_feature(width),
                  'image/filename': dataset_util.bytes_feature(cur_full_filename.encode('utf8')),
                  'image/source_id': dataset_util.bytes_feature(cur_full_filename.encode('utf8')),
                  'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                  'image/format': dataset_util.bytes_feature(image_format),
                  'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                  'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                  'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                  'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                  'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                  'image/object/class/label': dataset_util.int64_list_feature(classes)}))
              writer.write(tf_example.SerializeToString())


          cur_filename = ""
          new_file = False
          for i in tqdm(range(1, len(groups)), ncols=100):
              filename = groups['filename'][i]
              xmin = groups['xmin'][i]
              xmax = groups['xmax'][i]
              ymin = groups['ymin'][i]
              ymax = groups['ymax'][i]
              class_ = groups['class'][i]
              width = groups['width'][i]
              height = groups['height'][i]

              save_flag = False
              if not cur_filename:
                  cur_filename = filename
                  new_file = True
              elif filename != cur_filename:
                  cur_filename = filename
                  save_flag = True
                  new_file = True
              else : new_file = False

              if save_flag:
                  addSample()

              if new_file :
                  cur_full_filename =os.path.join( os.path.join(images_path,branch), cur_filename)
                  with tf.io.gfile.GFile(cur_full_filename, 'rb') as fid:
                      encoded_jpg = fid.read()

                  # encoded_jpg_io = io.BytesIO(encoded_jpg)
                  # image = Image.open(encoded_jpg_io)
                  # width, height = image.size

                  # filename = group.filename.encode('utf8')
                  xmins = []
                  xmaxs = []
                  ymins = []
                  ymaxs = []
                  classes_text = []
                  classes = []



              xmins.append(xmin / width)
              xmaxs.append(xmax / width)
              ymins.append(ymin / height)
              ymaxs.append(ymax / height)
              classes_text.append(class_.encode('utf8'))
              classes.append(GTUtilityDET.getIndex(class_,labels))



          if save_flag:
              addSample()
          writer.close()


  @staticmethod
  def extractCSVLabels(csv_filename):
      all_labels = pd.read_csv(csv_filename, sep=',', usecols=['class'])

      return list(set(all_labels['class']))


  @staticmethod
  def csv2TFRec(images_path, csv_filename,tf_rec_filename,branch,labels):
      # labels = GTUtilityDET.extractCSVLabels(csv_filename)
      writer = tf.io.TFRecordWriter(tf_rec_filename)
      grouped = pd.read_csv(csv_filename)
      GTUtilityDET.createTFExample(images_path,grouped,tf_rec_filename,labels,branch)
      writer.close()

  @staticmethod
  def csv2TFRecBranchs(images_path, csv_path, dst_path,labels):

      if not os.path.exists(dst_path):
          os.mkdir(dst_path)

      branchs = FileUtility.getFolderFiles(csv_path,'csv',False,False)
      for branch in branchs :
         GTUtilityDET.csv2TFRec(images_path, os.path.join(csv_path, branch+'.csv'),os.path.join(dst_path, branch+'.record'),branch,labels)


 

  @staticmethod
  def copySplitGT2(src_path, dst_path, train_per=0.8, copy_to_root=False, select_type=IndexType.random, clear_dst=False,branchs = ['train', 'test']):

      FileUtility.createDstBrach(dst_path, branchs, clear_dst)



      branch_state = False
      if not FileUtility.checkRootFolder(src_path):
          branch_state = True
          if copy_to_root == False:
              for branch in branchs :
                 FileUtility.copyFullSubFolders(src_path,os.path.join( dst_path,branch))

      dst_path_train = os.path.join(dst_path,branchs[0])
      dst_path_test = os.path.join(dst_path, branchs[1])

      if branch_state and (select_type == IndexType.begin_branch or select_type == IndexType.end_branch):
          sub_folders = FileUtility.getSubfolders(src_path)
          for sub_folder in sub_folders:
              src_cur_branch = os.path.join(src_path, sub_folder)
              dst_cur_branch_train = os.path.join(dst_path_train, sub_folder)
              dst_cur_branch_test = os.path.join(dst_path_test, sub_folder)


              image_filenames, gt_filenames = GTUtilityDET.getGtFiles(src_cur_branch)
              train_indexs,test_indexs  = GTUtility.getGTIndexs(len(image_filenames), train_per, select_type)

              src_train_image_filenames, src_train_gt_filenames = GTUtilityDET.getGTFiles(image_filenames, gt_filenames, train_indexs)
              src_test_image_filenames, src_test_gt_filenames = GTUtilityDET.getGTFiles(image_filenames, gt_filenames, test_indexs)


              if copy_to_root:
                  dst_train_image_filenames = FileUtility.getDstFilenames2(src_train_image_filenames, src_cur_branch, dst_path_train,
                                                                     copy_to_root)
                  dst_train_gt_filenames = FileUtility.getDstFilenames2(src_train_gt_filenames, src_cur_branch, dst_path_train,
                                                                  copy_to_root)

                  dst_test_image_filenames = FileUtility.getDstFilenames2(src_test_image_filenames, src_cur_branch,
                                                                           dst_path_test,
                                                                           copy_to_root)
                  dst_test_gt_filenames = FileUtility.getDstFilenames2(src_test_gt_filenames, src_cur_branch,
                                                                        dst_path_test,
                                                                        copy_to_root)


              else:
                  dst_train_image_filenames = FileUtility.getDstFilenames2(src_train_image_filenames, src_cur_branch,
                                                                     dst_cur_branch_train, copy_to_root)
                  dst_train_gt_filenames = FileUtility.getDstFilenames2(src_train_gt_filenames, src_cur_branch, dst_cur_branch_train,
                                                                  copy_to_root)

                  dst_test_image_filenames = FileUtility.getDstFilenames2(src_test_image_filenames, src_cur_branch,
                                                                     dst_cur_branch_test, copy_to_root)
                  dst_test_gt_filenames = FileUtility.getDstFilenames2(src_test_gt_filenames, src_cur_branch, dst_cur_branch_test,
                                                                  copy_to_root)

              FileUtility.copyFilesByName(src_train_image_filenames, dst_train_image_filenames)
              FileUtility.copyFilesByName(src_train_gt_filenames, dst_train_gt_filenames)
              FileUtility.copyFilesByName(src_test_image_filenames, dst_test_image_filenames)
              FileUtility.copyFilesByName(src_test_gt_filenames, dst_test_gt_filenames)


      else:
          image_filenames, gt_filenames = GTUtilityDET.getGtFiles(src_path)
          train_indexs, test_indexs = GTUtility.getGTIndexs(len(image_filenames), train_per, select_type)

          src_train_image_filenames, src_train_gt_filenames = GTUtilityDET.getGTFiles(image_filenames, gt_filenames, train_indexs)
          src_test_image_filenames, src_test_gt_filenames = GTUtilityDET.getGTFiles(image_filenames, gt_filenames,test_indexs)

          dst_path_train = os.path.join(dst_path,branchs[0])
          dst_path_test = os.path.join(dst_path, branchs[1])


          dst_train_image_filenames = FileUtility.getDstFilenames2(src_train_image_filenames, src_path, dst_path_train, copy_to_root)
          dst_train_gt_filenames = FileUtility.getDstFilenames2(src_train_gt_filenames, src_path, dst_path_train, copy_to_root)

          dst_test_image_filenames = FileUtility.getDstFilenames2(src_test_image_filenames, src_path, dst_path_test, copy_to_root)
          dst_test_gt_filenames = FileUtility.getDstFilenames2(src_test_gt_filenames, src_path, dst_path_test, copy_to_root)

          FileUtility.copyFilesByName(src_train_image_filenames, dst_train_image_filenames)
          FileUtility.copyFilesByName(src_train_gt_filenames, dst_train_gt_filenames)
          FileUtility.copyFilesByName(src_test_image_filenames, dst_test_image_filenames)
          FileUtility.copyFilesByName(src_test_gt_filenames, dst_test_gt_filenames)


  @staticmethod
  def copyGTAsPer(src_path, dst_path, per=1.0, copy_to_root=False, select_type=IndexType.random, clear_dst=False):
        if clear_dst:
            FileUtility.createClearFolder(dst_path)

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        branch_state = False
        if not FileUtility.checkRootFolder(src_path):
            branch_state = True
            if not copy_to_root:
                FileUtility.copyFullSubFolders(src_path, dst_path)

        if branch_state and (select_type == IndexType.begin_branch or select_type == IndexType.end_branch):
            sub_folders = FileUtility.getSubfolders(src_path)
            for sub_folder in sub_folders:
                src_cur_branch = os.path.join(src_path, sub_folder)
                dst_cur_branch = os.path.join(dst_path, sub_folder)

                image_filenames, gt_filenames = GTUtilityDET.getGtFiles(src_cur_branch)
                indexs, _ = GTUtility.getGTIndexs(len(image_filenames), per, select_type)
                src_image_filenames, src_gt_filenames = GTUtilityDET.getGTFiles(image_filenames, gt_filenames, indexs)

                if copy_to_root:
                    dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames, src_cur_branch, dst_path,
                                                                       copy_to_root)
                    dst_gt_filenames = FileUtility.getDstFilenames2(src_gt_filenames, src_cur_branch, dst_path,
                                                                    copy_to_root)
                else:
                    dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames, src_cur_branch, dst_cur_branch,
                                                                       copy_to_root)
                    dst_gt_filenames = FileUtility.getDstFilenames2(src_gt_filenames, src_cur_branch, dst_cur_branch,
                                                                    copy_to_root)

                FileUtility.copyFilesByName(src_image_filenames, dst_image_filenames)
                FileUtility.copyFilesByName(src_gt_filenames, dst_gt_filenames)


        else:
            image_filenames, gt_filenames = GTUtilityDET.getGtFiles(src_path)
            indexs, _ = GTUtility.getGTIndexs(len(image_filenames), per, select_type)

            src_image_filenames, src_gt_filenames = GTUtilityDET.getGTFiles(image_filenames, gt_filenames, indexs)

            dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames, src_path, dst_path, copy_to_root)
            dst_gt_filenames = FileUtility.getDstFilenames2(src_gt_filenames, src_path, dst_path, copy_to_root)

            FileUtility.copyFilesByName(src_image_filenames, dst_image_filenames)
            FileUtility.copyFilesByName(src_gt_filenames, dst_gt_filenames)

  @staticmethod
  def convertGT2TFRec(src_path,dst_path,labels, train_per = 0.8,clear_dst = True):
      GTUtilityDET.copySplitGT2(src_path,dst_path,train_per,True,clear_dst = clear_dst)
      GTUtilityDET.GT2CsvBranchs(dst_path,dst_path)
      GTUtilityDET.csv2TFRecBranchs(dst_path, dst_path, dst_path,labels)

  @staticmethod
  def createLabelMap(dst_filename, labels):
      with open(dst_filename, "w") as file:
          for i, label in enumerate(labels):
              str = 'item {{\n\tid: {0}\n\tname: {1}\n}}\n'.format(i + 1, label)
              file.write(str)

  @staticmethod
  def readLabelMap(filename):
      result = []
      with open(filename, 'r') as file:
          lines = file.readlines()

      for line in lines:
          flag, res = Utility.readField(line, 'name: ')
          if flag:
              print(res)
              result.append(res)

      return result

