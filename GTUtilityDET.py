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

# from PIL import Image
# from object_detection.utils import dataset_util
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
  def splitGT(src_path,train_per = 0.8):
      image_filenames ,gt_filenames = GTUtilityDET.getGtFiles(src_path)
      train_indexs ,test_indexs = GTUtility.getGTRandomIndexs(len(image_filenames),train_per)

      train_image_filenames = Utility.getListByIndexs(image_filenames,train_indexs)
      train_gt_filenames = Utility.getListByIndexs(gt_filenames, train_indexs)

      test_image_filenames = Utility.getListByIndexs(image_filenames, test_indexs)
      test_gt_filenames = Utility.getListByIndexs(gt_filenames, test_indexs)

      return train_image_filenames,train_gt_filenames,test_image_filenames,test_gt_filenames



  @staticmethod
  def copySplitGT(src_path,dst_path,train_per = 0.8,clear_dst = False,csv_file = False):
      branchs = ['train','test']

      FileUtility.createDstBrach(dst_path, branchs, clear_dst)
      for branch in branchs:
         FileUtility.copyFullSubFolders(src_path,os.path.join(dst_path,branch))

      src_train_image_filenames, src_train_gt_filenames, src_test_image_filenames, src_test_gt_filenames = GTUtilityDET.splitGT(src_path,train_per)

      dst_train_image_filenames = FileUtility.getDstFilenames2(src_train_image_filenames,src_path,os.path.join(dst_path,branchs[0]))
      dst_train_gt_filenames = FileUtility.getDstFilenames2(src_train_gt_filenames, src_path, os.path.join(dst_path,branchs[0]))

      dst_test_image_filenames = FileUtility.getDstFilenames2(src_test_image_filenames, src_path, os.path.join(dst_path,branchs[1]))
      dst_test_gt_filenames = FileUtility.getDstFilenames2(src_test_gt_filenames, src_path, os.path.join(dst_path,branchs[1]))

      FileUtility.copyFilesByName(src_train_image_filenames,dst_train_image_filenames)
      FileUtility.copyFilesByName(src_train_gt_filenames, dst_train_gt_filenames)
      FileUtility.copyFilesByName(src_test_image_filenames, dst_test_image_filenames)
      FileUtility.copyFilesByName(src_test_gt_filenames, dst_test_gt_filenames)





  @staticmethod
  def voc2Csv(src_path,csv_filename):
      image_filenames ,gt_filenames = GTUtilityDET.getGtFiles(src_path)
      xml_list = []

      for gt_filename in gt_filenames :
          voc = VOC()
          voc.load(gt_filename)


  @staticmethod
  def voc2CsvBatch(src_path,dst_path):
      sub_folders = FileUtility.getSubfolders(src_path)
      for sub_folder in sub_folders :
         cur_folder = os.path.join(src_path,sub_folder)
         GTUtilityDET.voc2Csv(cur_folder,os.path.join(dst_path,sub_folder))




# def xml_to_csv(path):
# 
#     xml_list = []
#     for xml_file in glob.glob(path + '/*.xml'):
#         tree = ET.parse(xml_file)
#         root = tree.getroot()
#         for member in root.findall('object'):
#             value = (root.find('filename').text,
#                      int(root.find('size')[0].text),
#                      int(root.find('size')[1].text),
#                      member[0].text,
#                      int(member[4][0].text),
#                      int(member[4][1].text),
#                      int(member[4][2].text),
#                      int(member[4][3].text)
#                      )
#             xml_list.append(value)
#     column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
#     xml_df = pd.DataFrame(xml_list, columns=column_name)
#     return xml_df
# 
# 
# def main():
#     src_path = '/root/datalab'
#     for folder in ['train','test']:
#         image_path = os.path.join(src_path, folder)
#         xml_df = xml_to_csv(image_path)
#         xml_df.to_csv(os.path.join( src_path, folder + '_labels.csv'), index=None)
#         print('Successfully converted xml to csv.')
# 
# 
# 
# 
# 
# 



  @staticmethod
  def classLabelIndex(row_label,labels):
          index = labels.getIndex(row_label)
          if index >= 0:
              index += 1
          else : index = -1

          return index


  @staticmethod
  def split(df, group):
          data = namedtuple('data', ['filename', 'object'])
          gb = df.groupby(group)
          return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

  @staticmethod
  def create_tf_example(group, path,labels):
          with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
              encoded_jpg = fid.read()
          encoded_jpg_io = io.BytesIO(encoded_jpg)
          image = Image.open(encoded_jpg_io)
          width, height = image.size

          filename = group.filename.encode('utf8')
          image_format = b'jpg'
          xmins = []
          xmaxs = []
          ymins = []
          ymaxs = []
          classes_text = []
          classes = []

          for index, row in group.object.iterrows():
              xmins.append(row['xmin'] / width)
              xmaxs.append(row['xmax'] / width)
              ymins.append(row['ymin'] / height)
              ymaxs.append(row['ymax'] / height)
              classes_text.append(row['class'].encode('utf8'))
              classes.append(class_text_to_int(row['class']))

          tf_example = tf.train.Example(features=tf.train.Features(feature={
              'image/height': dataset_util.int64_feature(height),
              'image/width': dataset_util.int64_feature(width),
              'image/filename': dataset_util.bytes_feature(filename),
              'image/source_id': dataset_util.bytes_feature(filename),
              'image/encoded': dataset_util.bytes_feature(encoded_jpg),
              'image/format': dataset_util.bytes_feature(image_format),
              'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
              'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
              'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
              'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
              'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
              'image/object/class/label': dataset_util.int64_list_feature(classes),
          }))
          return tf_example

  @staticmethod
  def csv2TFRecord(csv_filename,tf_rec_filename,labels):
      writer = tf.compat.v1.python_io.getGTTFRecordWriter(tf_rec_filename)

      examples = pd.read_csv(csv_filename)
      grouped = split(examples, 'filename')
      for i in tqdm(range(1, len(grouped)), ncols=100):
          group  = grouped[i]
          tf_example = create_tf_example(group, image_dir)
          writer.write(tf_example.SerializeToString())

      writer.close()

  @staticmethod
  def csv2TFRecordBatch(src_path,dst_path,labels):
      GTUtilityDET.csv2TFRecord(os.path.join("train.csv"),os.path.join("train.record"))
      GTUtilityDET.csv2TFRecord(os.path.join("test.csv"), os.path.join("test.record"))



 



      
      












