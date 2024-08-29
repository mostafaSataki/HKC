from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import glob
import os
import re
import random
import shutil
from  .FileUtility import  *
from  .Utility import  *
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from .CvUtility import *



class TrainUtility:
    @staticmethod
    def split_data(src_dir, dst_dir, valid_per=0.2, test_per=0):
        branches = ['train', 'val', 'test']
        FileUtility.createClearFolder(dst_dir)
        train_per = 1 - valid_per - test_per
        pers = [train_per, valid_per]

        if test_per != 0:
            pers.append(test_per)

        # Create destination folders
        class_names = FileUtility.getSubfolders(src_dir)
        for i in range(len(pers)):
            dst_branch = os.path.join(dst_dir, branches[i])
            FileUtility.createClearFolder(dst_branch)
            FileUtility.createSubfolders(dst_branch, class_names)

        def process_class(class_name):
            src_class_path = os.path.join(src_dir, class_name)
            src_class_files = FileUtility.getFolderImageFiles(src_class_path)
            count = len(src_class_files)
            indexs = [i for i in range(count)]
            random.shuffle(indexs)
            start = 0

            for i in range(len(pers)):
                branch = branches[i]
                per = pers[i]
                dst_branch_path = os.path.join(dst_dir, branch)
                dst_class_path = os.path.join(dst_branch_path, class_name)

                branch_count = int(count * per)
                for j in range(start, start + branch_count):
                    src_file = src_class_files[indexs[j]]
                    dst_file = FileUtility.getDstFilename2(src_file, dst_class_path, src_class_path, True)
                    FileUtility.copyFile(src_file, dst_file)
                start += branch_count

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_class, class_name) for class_name in class_names]
            for _ in tqdm(as_completed(futures), total=len(futures), ncols=100, desc="Splitting Data"):
                pass



# from keras.models import load_model
#
#
# import os
# import io
# import pandas as pd
# import tensorflow as tf
#
# from PIL import Image
# # from object_detection.utils import dataset_util
# from collections import namedtuple, OrderedDict
# from  keras import  backend as K
# from .CryptoUtility import *
# import h5py
#
#
#
# class TrainUtility:
#
#   @staticmethod
#   def randomCopy(src_path,dst_path,train_per = 0.7,cut_flag = False):
#
#     train_path = os.path.join(dst_path, 'train')
#     test_path = os.path.join(dst_path, 'test')
#
#     if os.path.exists(train_path):
#         FileUtility.deleteFolderContents(train_path)
#     else :os.makedirs(train_path)
#
#     if os.path.exists(test_path):
#         FileUtility.deleteFolderContents(test_path)
#     else :os.makedirs(test_path)
#
#
#     FileUtility.copy2Branchs(src_path,train_per,test_path,train_per,cut_flag)
#
#   @staticmethod
#   def xml_to_csv(path):
#       xml_list = []
#       for xml_file in glob.glob(path + '/*.xml'):
#           tree = ET.parse(xml_file)
#           root = tree.getroot()
#           for member in root.findall('object'):
#               value = (root.find('filename').text,
#                        int(root.find('size')[0].text),
#                        int(root.find('size')[1].text),
#                        member[0].text,
#                        int(member[4][0].text),
#                        int(member[4][1].text),
#                        int(member[4][2].text),
#                        int(member[4][3].text)
#                        )
#               xml_list.append(value)
#       column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
#       xml_df = pd.DataFrame(xml_list, columns=column_name)
#       return xml_df
#
#
#   @staticmethod
#   def convertXml2Csv(src_path):
#       folders = FileUtility.getSubfolders()
#       for folder in folders:
#           image_path = os.path.join(src_path, folder)
#           xml_df = TrainUtility.xml_to_csv(image_path)
#           xml_df.to_csv(os.path.join( src_path, folder + '_labels.csv'), index=None)
#           print('Successfully converted xml to csv.')
#
#
#   @staticmethod
#   def class_text_to_int(filename):
#      pass
#
#   @staticmethod
#   def __split(df, group):
#       data = namedtuple('data', ['filename', 'object'])
#       gb = df.groupby(group)
#       return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
#
#   @staticmethod
#   def create_tf_example(group, path):
#     with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
#       encoded_jpg = fid.read()
#     encoded_jpg_io = io.BytesIO(encoded_jpg)
#     image = Image.open(encoded_jpg_io)
#     width, height = image.size
#
#     filename = group.filename.encode('utf8')
#     image_format = b'jpg'
#     xmins = []
#     xmaxs = []
#     ymins = []
#     ymaxs = []
#     classes_text = []
#     classes = []
#
#     for index, row in group.object.iterrows():
#       xmins.append(row['xmin'] / width)
#       xmaxs.append(row['xmax'] / width)
#       ymins.append(row['ymin'] / height)
#       ymaxs.append(row['ymax'] / height)
#       classes_text.append(row['class'].encode('utf8'))
#       classes.append(TrainUtility.class_text_to_int(row['class']))
#
#     tf_example = tf.train.Example(features=tf.train.Features(feature={
#       'image/height': dataset_util.int64_feature(height),
#       'image/width': dataset_util.int64_feature(width),
#       'image/filename': dataset_util.bytes_feature(filename),
#       'image/source_id': dataset_util.bytes_feature(filename),
#       'image/encoded': dataset_util.bytes_feature(encoded_jpg),
#       'image/format': dataset_util.bytes_feature(image_format),
#       'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
#       'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
#       'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
#       'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
#       'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#       'image/object/class/label': dataset_util.int64_list_feature(classes),
#     }))
#     return tf_example
#
#   @staticmethod
#   def generate2(output_path, image_dir, csv_input):
#     writer = tf.python_io.TFRecordWriter(output_path)
#
#     examples = pd.read_csv(csv_input)
#     grouped = TrainUtility.split(examples, 'filename')
#     for group in grouped:
#       tf_example = TrainUtility.create_tf_example(group, image_dir)
#       writer.write(tf_example.SerializeToString())
#
#     writer.close()
#     print('Successfully created the TFRecords: {}'.format(output_path))
#
#
# # def main2():
# #   src_path = '/root/datalab'
# #
# #   output_path_train = os.path.join(src_path, "train.record")
# #   csv_input_train = os.path.join(src_path, r"train_labels.csv")
# #   image_dir_train = os.path.join(src_path, "train")
# #
# #   output_path_test = os.path.join(src_path, "test.record")
# #   csv_input_test = os.path.join(src_path, r"test_labels.csv")
# #   image_dir_test = os.path.join(src_path, "test")
# #
# #   generate2(output_path_train, image_dir_train, csv_input_train)
# #   generate2(output_path_test, image_dir_test, csv_input_test)
# #
# #
# # # main1()
# # main2()
#   @staticmethod
#   def getFilenameLablesFromFolder(src_path):
#
#     files = []
#     lables = []
#     folders = FileUtility.getSubfolders(src_path)
#
#
#     for folder in folders:
#       tokens = os.path.split(folder)
#       folder_name = tokens[-1]
#       lable = Utility.str2Int(folder_name)
#       if lable[1] == True:
#         full_folder =os.path.join(src_path,folder)
#         class_files =  FileUtility.getFolderImageFiles(full_folder)
#
#         for file_name in class_files:
#           class_tokens = os.path.split(file_name)
#           f_name = tokens[-1] + "/" + class_tokens[-1]
#           files.append(os.path.join(src_path, f_name))
#           lables.append(tokens[-1])
#
#     return files, lables
#
#
#
#   @staticmethod
#   def writeFileList(filename,files,labels,sep=','):
#     with open(filename,'w') as f:
#       for i ,file in enumerate(files):
#         f.write("{}{}{}\n".format( file,sep,labels[i]))
#
#       f.close()
#
#   @staticmethod
#   def convertFolder2List(src,dst,train_per = 0.7,train_fname='train_list.txt',test_fname='test_list.txt'):
#
#     files ,labels = TrainUtility.getFilenameLablesFromFolder(src)
#     total_count = len(files)
#     train_count = int(total_count * train_per)
#
#     shuffle_list = Utility.getRandomList(total_count)
#
#     shuffle_train = shuffle_list[0:train_count]
#     shuffle_test = shuffle_list[train_count:]
#
#     train_files =  [files[index] for index in shuffle_train ]
#     train_labels =  [labels[index] for index in shuffle_train]
#
#     test_files =  [files[index] for index in shuffle_test ]
#     test_labels =  [labels[index] for index in shuffle_test]
#
#
#
#     train_list_filename = os.path.join(dst,train_fname)
#     test_list_filename = os.path.join(dst, test_fname)
#
#     TrainUtility.writeFileList(train_list_filename,train_files,train_labels)
#     TrainUtility.writeFileList(test_list_filename, test_files, test_labels)
#
#     return train_list_filename,test_list_filename
#
#   @staticmethod
#   def convertOneHot(labels, class_count):
#     count = len(labels)
#     new_labels = np.zeros(( count, class_count))
#
#     for item in range(0, count):
#       new_labels[item][int(labels[item])] = 1
#     return new_labels
#
#   @staticmethod
#   def createDataset(src_filename,class_count,sample_size=(23,23), train_per=0.7, sep=','):
#     with open(src_filename, 'r') as f:
#       lines = f.readlines()
#
#     all_count = len(lines)
#     train_count = int(all_count * train_per)
#     test_count = all_count - train_count
#
#     channel = 1
#     width,height = sample_size
#
#
#     test_filenames = []
#     X_train = np.zeros((train_count, height, width, channel), dtype="f4")
#     y_train = np.zeros((train_count, 1), dtype='f4')
#
#     X_test = np.zeros((test_count, height, width, channel), dtype="f4")
#     y_test = np.zeros((test_count, 1), dtype='f4')
#
#     idx = np.random.permutation(all_count)
#
#     for i, l in enumerate(lines):
#       id = idx[i]
#       sp = l.split(sep)
#       img = cv2.imread(sp[0], 0)
#       if np.shape(img) == ():
#         continue
#       img = cv2.resize(img, (width, height))
#       img = img.astype('f4')
#       img = img / 255.0
#       img = img.reshape(height, width, channel)
#
#       if i < train_count:
#         X_train[i] = img
#         y_train[i][0] = float(sp[1])
#       else:
#         ii = i - train_count
#
#         X_test[ii] = img
#         y_test[ii][0] = float(sp[1])
#
#         test_filenames.append(sp[0])
#
#     y_train = TrainUtility.convertOneHot(y_train, class_count)
#     y_test = TrainUtility.convertOneHot(y_test, class_count)
#
#     return X_train, y_train, X_test, y_test, test_filenames
#
#   @staticmethod
#   def createDatasetFromFolder(src_filename, class_count, sample_size=(23, 23)):
#
#     test_filenames = FileUtility.getFolderImageFiles(src_filename)
#
#     train_count = len(test_filenames)
#
#     channel = 1
#     width, height = sample_size
#
#     # test_filenames = []
#     X_train = np.zeros((train_count, height, width, channel), dtype="f4")
#     y_train = np.zeros((train_count, 1), dtype='f4')
#
#
#
#
#     for i, filename in enumerate(test_filenames):
#
#
#       img = cv2.imread(filename, 0)
#       if np.shape(img) == ():
#         continue
#       img = cv2.resize(img, (width, height))
#       img = img.astype('f4')
#       img = img / 255.0
#       img = img.reshape(height, width, channel)
#
#       if i < train_count:
#         X_train[i] = img
#         # y_train[i][0] = float(sp[1])
#
#     y_train = TrainUtility.convertOneHot(y_train, class_count)
#
#
#     return X_train, y_train, test_filenames
#
#   @staticmethod
#   def getFileListClassCount(filename):
#     pass
#
#   @staticmethod
#   def saveCheckPoint(model, path, checkpoint_filename, graphdef_filename):
#     print("input:", model.input.op.name)
#     print("output:", model.output.op.name)
#     saver = tf.train.Saver()
#     chkp_filename = os.path.join(path, checkpoint_filename)
#     saver.save(K.get_session(), chkp_filename)
#
#     tf.train.write_graph(K.get_session().graph.as_graph_def(), path, graphdef_filename)
#
#
#   @staticmethod
#   def checkpointToFreeze(checkpoint_filename1, path):
#     pass
#
#   @staticmethod
#   def load_model(filename,key):
#       with open(filename, 'rb') as fh:
#           io_bytes = io.BytesIO(fh.read())
#
#           c = AESCipher(key)
#           buffer = c.decrypt_data(io_bytes.read())
#           io_bytes.flush()
#           io_bytes.seek(0)
#           io_bytes.write(buffer)
#           # io_bytes2 = io.BytesIO(buffer)
#       with h5py.File(io_bytes, 'r') as h5_file:
#           model = load_model(h5_file)
#       return model
#
#
#     # model = load_model(checkpoint_filename1)
#     #
#     # checkpoint_filename = "tf_model.ckpt"
#     # graphdef_filename = "tf_model.pbtxt"
#     # TrainUtility.saveCheckPoint(model, path, checkpoint_filename, graphdef_filename)
#     #
#     # python_path = sys.executable
#     # freeze_graph_path = "E:/tensorflow_repo/tensor_gpu2/Lib/site-packages/tensorflow/python/tools/freeze_graph.py"
#     # input_graph_path = os.path.join(path, "tf_model.pbtxt")
#     # checkpoint_path = os.path.join(path, "tf_model.ckpt")
#     # forzen_path = os.path.join(path, "forzen.pb")
#     # frozen_cut_path = os.path.join(path, "frozen_cut.pb")
#     # frozen_cut_opt_path = os.path.join(path, "frozen_cut_opt.pb")
#     #
#     # transform_graph_path = r"C:\Users\mostafa\_bazel_mostafa\32oed7gf\execroot\org_tensorflow\bazel-out\x64_windows-opt\bin\tensorflow\tools\graph_transforms\transform_graph.exe"
#     # site_packages_path = r"E:\tensorflow_repo\tensor_gpu_env\Lib\site-packages"
#     # optimize_for_inference_path = os.path.join(site_packages_path, r"tensorflow/python/tools/optimize_for_inference.py")
#     # # model.input.op.name
#     #
#     # command = [python_path, freeze_graph_path, '--input_graph=' + input_graph_path,
#     #            '--input_checkpoint=' + checkpoint_path, '--output_graph=' + forzen_path,
#     #            '--output_node_names=' + model.output.op.name, '--input_binary=false']
#     # p = subprocess.Popen(command, stdout=subprocess.PIPE)
#     #
#     # command = [transform_graph_path, '--in_graph=' + forzen_path,
#     #            '--out_graph=' + frozen_cut_path, '--inputs=' + model.input.op.name,
#     #            '--outputs=' + model.output.op.name,
#     #            '--transforms=strip_unused_nodes(type=float,shape=\"1,23,23,1\") fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms sort_by_execution_order ']
#     # p = subprocess.Popen(command, stdout=subprocess.PIPE)
#     #
#     # os.chdir(site_packages_path)
#     # command = ['python', optimize_for_inference_path,
#     #            '--input=' + frozen_cut_path, '--output=' + frozen_cut_opt_path, '--frozen_graph True',
#     #            '--input_names=' + model.input.op.name, '--output_names=' + model.output.op.name]
#     #
#     # print(command)
#     # p = subprocess.Popen(command, stdout=subprocess.PIPE)
