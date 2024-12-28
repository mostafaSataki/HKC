import  glob
import ntpath
import os
import shutil
from collections import defaultdict
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import cv2
import  numpy as np
import  subprocess
import os
import time
import datetime
from .Utility import  *
from pathlib import Path
import re
from tqdm import tqdm
import tarfile
import urllib.request
import zipfile
import tempfile
from  enum import  Enum
import csv
import os
import platform
from pathlib import Path
from .Utility import *
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

class MediaType(Enum):
  folder = 1
  file = 2


class FileUtility:
  @staticmethod
  def getImageExtensions():
    return ['bmp', 'jpg',  'tif', 'tiff','png']

  @staticmethod
  def getAudioExtensions():
    return ['wav', 'mp3',  'wma', 'aac','ogg']

  @staticmethod
  def getVideoExtensions():
    return ['mp4', 'avi',  'mov','mkv']

  @staticmethod
  def getModuleExtensions():
    return ['exe', 'dll',  'so']

  @staticmethod
  def removeFileExt(src_filename):
    return src_filename[0:-len(FileUtility.getFileExt(src_filename))-1]

  @staticmethod
  def removeFilename(src_filename):
     p,_,_ =  FileUtility.getFileTokens(src_filename)
     return p

  @staticmethod
  def getFileFolder(filename):
      tokens = FileUtility.getFileTokens(filename)
      f = os.path.basename(os.path.normpath(tokens[0]))
      return f


  @staticmethod
  def getFilesFolder(filesnames):
      result = []
      for filename in filesnames:
        result.append(FileUtility.getFileFolder(filename))
      return result

  @staticmethod
  def getFileLabel(filename):
    return int(FileUtility.getFileFolder(filename))

  @staticmethod
  def getFolderLabel(folder):
    return Path(folder).stem

  @staticmethod
  def getFoldersLabel(folders):
    result = []
    for folder in folders:
      result.append(FileUtility.getFolderLabel(folder))
    return result

  @staticmethod
  def getFilesLabel(filesnames):
    result = []
    for filename in filesnames:
      result.append(FileUtility.getFileLabel(filename))
    return result

  @staticmethod
  def getLabelsIndex(labels):
    unique_labels = list(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}


    indexed_labels = [label_to_index[label] for label in labels]

    return label_to_index,indexed_labels

  @staticmethod
  def getFilesUniqueLabels(filesnames):
    labels = FileUtility.getFilesLabel(filesnames)
    labels = list(set(labels))
    result = []
    for lable in labels:
      result.append(str(lable))
    return result



  @staticmethod
  def checkIsAudio(filename):
    ext = FileUtility.getFileExt(filename).lower()
    return  ext in FileUtility.getAudioExtensions()

  @staticmethod
  def checkIsImage(filename):
    ext = FileUtility.getFileExt(filename).lower()
    return  ext in FileUtility.getImageExtensions()

  @staticmethod
  def checkExt(filename,Exts):
    ext = FileUtility.getFileExt(filename).lower()
    return  ext in Exts

  @staticmethod
  def checkIsModule(filename):
    ext = FileUtility.getFileExt(filename).lower()
    return  ext in FileUtility.getModuleExtensions()


  @staticmethod
  def checkIsVideo(filename):
    ext = FileUtility.getFileExt(filename).lower()
    return  ext in FileUtility.getVideoExtensions()


  def getTifExtensions():
    return ['tif','tiff']

  def getTexExtensions():
    return ['txt']

  @staticmethod
  def checkTifImage(filename):
    ext = FileUtility.getFileExt(filename).lower()
    return ext in FileUtility.getTifExtensions()

  @staticmethod
  def getNextFolder(src_path):
    pass


  @staticmethod
  def getFileExt(filename):
    result = Path(filename).suffix

    result = re.sub('[.]', '', result)
    return result

  # @staticmethod
  # def getFolderFiles(path, ext = None, has_path = True, has_ext = True):
  #   result = []
  #   for (dirpath, dirnames, filenames) in os.walk(path):
  #     for filename in filenames:
  #         fname = FileUtility.getFilenameWithoutExt(filename)
  #         cur_ext = FileUtility.getFileExt(filename)
  #         if ext != None and cur_ext != ext:
  #           continue
  #
  #         if has_path :
  #           name = os.path.join(dirpath,fname)
  #         else :name = fname
  #
  #         if has_ext:
  #           name = name+'.'+cur_ext
  #         result.append(name)
  #         # result.append(os.path.join(dirpath,filename))
  #
  #   return  result

  @staticmethod
  def getFolderFilesname( path):
    return  [f for f in listdir(path) if isfile(join(path, f))]


  @staticmethod
  def hasPostfix(filename,postfix):
    tokens = FileUtility.getFileTokens(filename)
    return tokens[1].endswith(postfix)

  @staticmethod
  def getFilenameLastPostfix(filename):
    fname = FileUtility.getFilenameWithoutExt(filename)
    parts = fname.split('_')
    if len(parts) <= 1:
      return None
    else : return parts[-1]

  @staticmethod
  def getFilenameFirstPostfix(filename):
    fname = FileUtility.getFilenameWithoutExt(filename)
    parts = fname.split('_')
    if len(parts) < 1:
      return None
    else : return parts[0]
      
  @staticmethod
  def getFilenamesFirstPostfix(filenames,key_postfix = True):
    result = {}
    for filename in filenames:
      postfix = FileUtility.getFilenameFirstPostfix(filename)
      if key_postfix:
        if postfix in result:
          result[postfix].append(filename)  # If the postfix exists, append the filename to the list
        else:
          result[postfix] = [filename]  # If the postfix doesn't exist, create a new list with the filename
      else :
        if filename in result:
          result[filename].append(postfix)  # If the filename exists, append the postfix to the list
        else:
          result[filename] = [postfix]
    return result

  @staticmethod
  def getDirFirstPostfix(dir,key_postfix = True):
      filenames = FileUtility.getFolderFiles(dir)
      return FileUtility.getFilenamesFirstPostfix(filenames,key_postfix)


    
      
  @staticmethod
  def getFilenameTokens(filename):
    fname = FileUtility.getFilenameWithoutExt(filename)
    parts = fname.split('_')
    if len(parts) < 1:
      return None
    else : return parts


  @staticmethod
  def getFolderPostfix(src_dir):
    filenames = FileUtility.getFolderFiles(src_dir)
    postfixes = []
    for filename in filenames:
      postfix = FileUtility.getFilenameLastPostfix(filename)
      if postfix is not None:
        postfixes.append(postfix)

    if len(postfixes) == 0:
      return None
    # Count the frequency of each postfix
    postfix_counter = Counter(postfixes)

    # Find the most common postfix
    most_common_postfix, count = postfix_counter.most_common(1)[0]
    return most_common_postfix

    

  @staticmethod
  def getFolderImageFiles(path,postfix = None):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
        if FileUtility.checkIsImage(filename):
            if postfix is not None :
                if FileUtility.hasPostfix(filename,postfix):
                   result.append(os.path.join(dirpath, filename))
            else :result.append(os.path.join(dirpath, filename))

    return result

  @staticmethod
  def getFolderFiles(path,Exts =None):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
        if Exts:
          if FileUtility.checkExt(filename,Exts):
            result.append(os.path.join(dirpath, filename))
        else :result.append(os.path.join(dirpath, filename))

    return result

  @staticmethod
  def getFolderAllFiles(path):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
          result.append(os.path.join(dirpath, filename))

    return result


  @staticmethod
  def getFoldersImageFiles(paths,Exts):
    result = []
    for path in paths:
      if len(result) == 0:
        result = FileUtility.getFolderFiles(path,Exts)
      else: result += FileUtility.getFolderFiles(path,Exts)

    return result
  @staticmethod
  def getFoldersImageGTFiles(paths,GtExt):
    src_image_files = FileUtility.getFolderImageFiles(paths)
    src_gt_files,src_image_files  = FileUtility.changeFilesExt2(src_image_files, GtExt, True)
    return src_gt_files,src_image_files

  @staticmethod
  def getFoldersFiles(path,ext=None):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
        if ext:
          if FileUtility.checkExt(filename, ext):
            result.append(os.path.join(dirpath, filename))
        else :result.append(os.path.join(dirpath, filename))


    return result


  @staticmethod
  def getFolderAudioFiles(path):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
        if FileUtility.checkIsAudio(filename):
          result.append(os.path.join(dirpath, filename))

    return result

  @staticmethod
  def getFolderModuleFiles(path):
    result = []
    filenames = FileUtility.getFolderFiles(path)
    for filename in filenames:
        if FileUtility.checkIsModule(filename):
          result.append(os.path.join(path, filename))

    return result

  @staticmethod
  def getFolderNonImageFiles(path):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
        if not FileUtility.checkIsImage(filename):
          result.append(os.path.join(dirpath, filename))

    return result

  @staticmethod
  def getFolderFilesByExt(path,ext):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
        if FileUtility.checkFileExt(filename,ext):
          result.append(os.path.join(dirpath, filename))

    return result

  @staticmethod
  def getFolderVideoFiles(path):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
        if FileUtility.checkIsVideo(filename):
          result.append(os.path.join(dirpath, filename))

    return result


  @staticmethod
  def removeFilesPath(files):
    result = []
    for file in files :
      result.append(os.path.basename(file))
    return result

  @staticmethod
  def getFolderImageFilesWithoutPath(path):
    files = FileUtility.getFolderImageFiles(path)
    return FileUtility.removeFilesPath(files)




  @staticmethod
  def copy2Branchs(src_path, train_path, test_path, train_per=0.7, cut_flag=False):
    all_files = FileUtility.getFolderImageFiles(src_path)
    idx = np.random.permutation(len(all_files))

    train_count = len(all_files) * train_per
    for i, id in enumerate(idx):
      cur_filename = all_files[id]
      _, f_name = os.path.split(cur_filename)

      if i <= train_count:
        dst_filename = os.path.join(train_path, f_name)
      else:
        dst_filename = os.path.join(test_path, f_name)

      shutil.copyfile(cur_filename, dst_filename)

      if cut_flag:
        os.remove(cur_filename)

  @staticmethod
  def joins(filename,file_path,sub_folders):
    result = []
    for sub_folder in sub_folders:
      result.append(os.path.join(os.path.join(file_path,sub_folder),filename))
    return result
  
  @staticmethod
  def intersection(folder1,folder2):
    filenames1 = FileUtility.getFolderFiles(folder1)
    filenames2 = FileUtility.getFolderFiles(folder2)

    fnames1 = set(FileUtility.getFilenames(filenames1))
    fnames2 = set(FileUtility.getFilenames(filenames2))

    common_files = fnames1.intersection(fnames2)

    new_filenames1 = FileUtility.joins(common_files, folder1)
    new_filenames2 = FileUtility.joins(common_files, folder2)
      
    return new_filenames1,new_filenames2

  @staticmethod
  def joins(filenames,dst_path):
    result = []
    for filename in filenames:
      result.append(os.path.join(dst_path,filename))
      
    return result

  @staticmethod
  def exist_count(filenames):
    count = 0
    files_list = []
    for filename in filenames:
      if os.path.exists(filename):
        count = count + 1
        files_list.append(filename)

    return  files_list, count

  @staticmethod
  def copy_all_files(src_path,dst_path):
      src_files = FileUtility.getFoldersFiles(src_path)
      FileUtility.copyFiles2DstPath(src_files,src_path,dst_path)

  @staticmethod
  def copyFiles(src_path, dst_path, pattern_path,cut_flag=False,pair_extension = None):
    pattern_files = glob.glob(os.path.join(pattern_path, "*"))
    sub_folders = FileUtility.getSubfolders(src_path)
    # for pattern_file in pattern_files:
    for i in tqdm( range(len(pattern_files)),ncols=100):
      pattern_file = pattern_files[i]
      pat_fname = FileUtility.getFilename(pattern_file)

      if len(sub_folders) != 0:
        cur_src_filenames = FileUtility.joins(pat_fname,src_path,sub_folders)
        files_list,files_count = FileUtility.exist_count(cur_src_filenames)
        if files_count == 1:
          src_filename = files_list[0]
          dst_filename = os.path.join(dst_path,pat_fname)
          FileUtility.copyFile(src_filename,dst_filename)
          if cut_flag:
            os.remove(src_filename)


        if pair_extension :
           src_pair_filename = FileUtility.changeFileExt(src_filename,pair_extension)

           if os.path.exists(src_pair_filename):
             dst_pair_filename = FileUtility.changeFileExt(dst_filename,pair_extension)
             shutil.copyfile(src_pair_filename, dst_pair_filename)
             if cut_flag:
               os.remove(src_pair_filename)

  @staticmethod
  def copy_images_percent(src_path,dst_path,per = 1.0,random_copy =False):
      src_images = FileUtility.getFolderImageFiles(src_path)
      dst_images = FileUtility.getDstFilenames2(src_images,src_path,dst_path)
      total_count = len(src_images)
      count = int(per * total_count)
      indexs = Utility.getRandomList(len(src_images))[:count]
      s_images = Utility.getListByIndexs(src_images,indexs)
      d_images = Utility.getListByIndexs(dst_images,indexs)
      FileUtility.copyFilesByName(s_images,d_images)



  @staticmethod
  def copyFiles2(src_path, dst_path, pattern_path, cut_flag=False, pair_extension=None):
    pattern_files = FileUtility.getFoldersFiles(pattern_path)

    src_files = FileUtility.getDstFilenames2(pattern_files,src_path,pattern_path,pair_extension)
    dst_files = FileUtility.getDstFilenames2(pattern_files, dst_path,pattern_path,pair_extension)

    for i in tqdm(range(len(pattern_files)), ncols=100):
      if os.path.exists(src_files[i]):
         FileUtility.copyFile(src_files[i], dst_files[i])
      if cut_flag:
        if os.path.exists(src_files[i]):
          os.remove(src_files[i])

  @staticmethod
  def copyFileByName(src_filenames, dst_filenames):
    if os.path.exists(dst_filenames):
      return
    if src_filenames != dst_filenames and os.path.exists(src_filenames):
      shutil.copyfile(src_filenames, dst_filenames)

  @staticmethod
  def copyFilesByName(src_filenames,dst_filenames):
    def copy_file(src,dst):
      if os.path.exists(dst):
        return
      if src != dst and os.path.exists(src):
        shutil.copyfile(src, dst)
    with ThreadPoolExecutor() as executor:
      list(tqdm(executor.map(copy_file,src_filenames,dst_filenames),total=len(src_filenames), ncols=100, desc='Copy Files'))


  @staticmethod
  def copyFiles2DstPath(src_filenames,dst_path,src_path=None):
    dst_files = FileUtility.getDstFilenames2(src_filenames,dst_path,src_path)
    FileUtility.copyFilesByName(src_filenames,dst_files)

  @staticmethod
  def copyFilesByLabels(src_files, dst_path, labels):

    unique_values = Utility.getUniqueValues(labels)
    sub_folders = list(map(str,unique_values))
    FileUtility.createSubfolders(dst_path, sub_folders)

    for i, src_filename in enumerate(src_files):
      label = labels[i]
      dst_filename = FileUtility.getDstFilename(src_filename, dst_path, str(label))
      shutil.copyfile(src_filename, dst_filename)

  @staticmethod
  def getDstFilename(src_filename, dst_path,postfix ='', dst_branch=None):
    # path, name = os.path.split(src_filename)
    path,fname,ext = FileUtility.getFileTokens(src_filename)

    if dst_branch == None:
      return os.path.join(dst_path, fname+postfix+ext)
    else:
      return os.path.join(os.path.join(dst_path, dst_branch), fname+postfix+ext)

  @staticmethod
  def get_parent_path(files):
      common_dir = os.path.commonprefix(files)
      common_dir = os.path.dirname(common_dir)
      return common_dir

  @staticmethod
  def getDstFilename2(src_filename, dst_path,src_path = None, copy_to_root = False,dst_extension = None):
    if src_path is None:
      src_path = FileUtility.get_parent_path(src_filename)

    if copy_to_root:
      tokens = FileUtility.getFileTokens(src_filename)
      result =  os.path.join(dst_path,tokens[1]+tokens[2])
    else :

      fname= src_filename[len(src_path)+1:]
      result = os.path.join(dst_path,fname)
    if dst_extension is not None:
      result = FileUtility.changeFileExt(result,dst_extension)

    return result

  @staticmethod
  def getDstFilenamesToBranch(src_filenames,ids, dst_path,src_path = None, dst_extension = None):
    result = []
    for i,src_filename in enumerate(src_filenames):
      dst_filename = FileUtility.getDstFilename2(src_filename,dst_path,src_path,True,dst_extension)
      tokens = FileUtility.getFileTokens(dst_filename)
      cur_dst_path = os.path.join(tokens[0],str(ids[i]))
      dst_filename = os.path.join(cur_dst_path,tokens[1]+tokens[2])
      result.append(dst_filename)

    return result


  @staticmethod
  def getDstFilenames2(src_filenames, dst_path,src_path = None, dst_extension = None):
    result = []
    for src_filename in src_filenames:
      result.append(FileUtility.getDstFilename2(src_filename,dst_path,src_path,True,dst_extension))

    return result

  @staticmethod
  def getDstFilenames(src_files_name, dst_path,postfix =''):
    result = []
    for src_file_name in src_files_name:
      result.append(FileUtility.getDstFilename(src_file_name, dst_path,postfix))
    return result
  


  @staticmethod
  def getChangeFilesExt(files_name, dst_ext,jpeg_quality = 30):
    result = []
    for filename in files_name:
      result.append(FileUtility.getChangeFileExt(filename, dst_ext))
    return result




  @staticmethod
  def getFolders(src_path):
    paths =[name for name in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, name))]
    full_paths = []
    for path in paths:
      full_paths.append(os.path.join(src_path,path))

    return full_paths,paths

  @staticmethod
  def moveFiles(src_path,dst_path):
    all_output_files = glob.glob(src_path + "\*")
    for output_file in all_output_files:
      p, fname = os.path.split(output_file)
      dst_filename = os.path.join(dst_path, fname)
      shutil.move(output_file, dst_filename)

  @staticmethod
  def getFilePath(file_name):
        return os.path.split(file_name)[0]

  @staticmethod
  def renameFiles(path, prefix , start_counter= 0, remain_filename =False, replace_folder_name = False):
    if replace_folder_name :
      sub_folders = FileUtility.getSubfolders(path)
      for sub_folder in sub_folders:
        full_path = os.path.join(path,sub_folder)
        counter = start_counter
        image_files = FileUtility.getFolderImageFiles(full_path)
        for file in image_files:
          tokens = FileUtility.getFileTokens(file)
          if remain_filename:
            dst_fille = os.path.join(tokens[0],tokens[1]+'_'+ sub_folder + '_' + str(counter) + tokens[2])
          else : dst_fille = os.path.join( tokens[0], tokens[1]+'_'+ sub_folder+'_' + str(counter)+tokens[2])
          os.rename(file, dst_fille)
          counter += 1

    else :

        image_files =  FileUtility.getFolderImageFiles(path)
        counter = start_counter
        for file in image_files:
          tokens = FileUtility.getFileTokens(file)
          if remain_filename:
            dst_fille = os.path.join(tokens[0],tokens[1]+'_'+ prefix +'_'+ str(counter)+ tokens[2])
          else : dst_fille = os.path.join( tokens[0], prefix +'_'+ str(counter)+tokens[2])
          os.rename(file, dst_fille)
          counter += 1

  @staticmethod
  def renameFiles2(src_path, pre_fix = None, post_fix = None, ext = None):
    files = FileUtility.getFolderFiles(src_path,ext)

    for i in tqdm(range(len(files)), ncols=100):
      file = files[i]
      tokens = FileUtility.getFileTokens(file)
      name = ''
      if pre_fix != None:
        name += pre_fix
      name += tokens[1]
      if post_fix != None:
        name += post_fix
      dst_fille = os.path.join(tokens[0],name + tokens[2])
      os.rename(file, dst_fille)

  @staticmethod
  def getFileTokens(file_name):
    path, name = os.path.split(file_name)
    x = name.rfind(".")
    if x == -1:
      f_name = ''
      ext =''
    else :
      f_name = name[0:x]
      ext = name[x:]

    return path, f_name,  ext

  @staticmethod
  def addPostfix2File(file_name, token):
    tokens = FileUtility.getFileTokens(file_name)
    return os.path.join(tokens[0], tokens[1] + token + tokens[2])

  @staticmethod
  def addPostfix2Files(filenames,token):
      result_files = []
      for filename in filenames:
        result_files.append( FileUtility.addPostfix2File(filename,token))

      return result_files

  @staticmethod
  def makeFoldersBranch(dst_path, branchs, clear=False):
    if clear and os.path.exists(dst_path):
      shutil.rmtree(dst_path)
    os.mkdir(dst_path)
    for branch in branchs:
      os.mkdir(os.path.join(dst_path, branch))

  @staticmethod
  def makeFolders(folders):
    for folder in folders:
      os.makedirs(folder,exist_ok=True)


  @staticmethod
  def deleteImages(path):
    dst_files = FileUtility.getFolderImageFiles(path)
    for i in range(len(dst_files)):
      if os.path.exists(dst_files[i]):
        os.remove(dst_files[i])

  @staticmethod
  def deleteImages(path, pattern_path):
    pattern_files = FileUtility.getFolderImageFiles(pattern_path)
    dst_files = FileUtility.getDstFilenames(pattern_files, path)
    for i in range(len(dst_files)):
      if os.path.exists(dst_files[i]):
        os.remove(dst_files[i])


  @staticmethod
  def checkFileExt(filename,ext):
    ext_ = FileUtility.getFileExt(filename).lower()
    return ext_ == ext


  @staticmethod
  def getFileGroup(files_name):
    groups = defaultdict(list)
    for filename in files_name:
       tokens = FileUtility.getFileTokens(filename)
       groups[os.path.join(tokens[0],tokens[1])].append(tokens[2])

    return groups

  @staticmethod
  def _deleteFileGroupBy(group,items):
    if len(items) != 2:
      for item in items:
        filename = group + item
        os.remove(filename)

  @staticmethod
  def checkExtList(ext,ext_list):
    ext_ = ext.replace('.','')
    return ext in ext_list

  @staticmethod
  def checkImageExt(ext):
    ext_ = ext.replace('.', '')
    return ext_ in FileUtility.getImageExtensions()

  @staticmethod
  def removeUnpairFiles(path,exts1,exts2):
    files_name = FileUtility.getFolderFiles(path)
    groups = FileUtility.getFileGroup(files_name)
    for group, items in groups.items():

      if len(items) != 2:
         FileUtility._deleteFileGroupBy(group,items)
      else :
        if not(FileUtility.checkExt(items[0],exts1) and FileUtility.checkExt(items[1],exts2)):
          FileUtility._deleteFileGroupBy(group, items)

  @staticmethod
  def getUnpaireFiles(path):
    result = []
    files_name = FileUtility.getFolderFiles(path)
    groups = FileUtility.getFileGroup(files_name)
    for group, items in groups.items():
      if len(items) == 1:
        result.append(group+items[0])

    return result


  @staticmethod
  def getUnpaireImages(path):
    result = []
    files_name = FileUtility.getFolderFiles(path)
    groups = FileUtility.getFileGroup(files_name)
    for group, items in groups.items():
      if len(items) == 1 and FileUtility.checkImageExt(items[0]):
        result.append(group+items[0])

    return result

  @staticmethod
  def getUnpaireSamples(path):
    result = []
    files_name = FileUtility.getFolderFiles(path)
    groups = FileUtility.getFileGroup(files_name)
    for group, items in groups.items():
      if len(items) == 1 :
        result.append(group+items[0])

    return result
  
  @staticmethod
  def remove_unpair_images(src_path):
    single_samples = FileUtility.getUnpaireSamples(src_path)
    print("single image count:", len(single_samples))
    for single_image in single_samples:
      print(single_image)
    FileUtility.delete_files(single_samples)

  @staticmethod
  def getImageFiles(src_path,dst_path):
    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames(src_files,dst_path)
    return src_files,dst_files

  @staticmethod
  def getFiles(src_path,dst_path):
    src_files = FileUtility.getFolderFiles(src_path)
    dst_files = FileUtility.getDstFilenames(src_files,dst_path)
    return src_files,dst_files

  @staticmethod
  def getSubfolders(path):
    return [FileUtility.upFolderName(f.path) for f in os.scandir(path) if f.is_dir()]

  @staticmethod
  def createSubfolders(path,sub_folders):
    result = []
    for sub in sub_folders:
      cur_sub_folder =os.path.join(path,sub)
      result.append(cur_sub_folder)
      if not os.path.isdir(cur_sub_folder):
        os.makedirs(cur_sub_folder)

    return result

  @staticmethod
  def create_validation_subfolders(src_path,dst_path):
    FileUtility.copy_subfolders(src_path, dst_path)
    sub_folders = FileUtility.getSubfolders(src_path)
    for sub_folder in sub_folders:
      cur_dst_folder = os.path.join(dst_path, sub_folder)
      FileUtility.copy_subfolders(src_path, cur_dst_folder)

  @staticmethod
  def createSubfoldersByRange(path,low,up):
    return FileUtility.createSubfolders(path,Utility.range2StrList(low,up))
    

  @staticmethod
  def copy_subfolders(src_path,dst_path):
      sub_folders = FileUtility.getSubfolders(src_path)
      FileUtility.createSubfolders(dst_path,sub_folders)
      
      return sub_folders


  @staticmethod
  def copyFileByNewExtension(src_filename,dst_filename,ext):
    new_src_filename = FileUtility.getChangeFileExt(src_filename,ext)
    new_dst_filename = FileUtility.getChangeFileExt(dst_filename, ext)
    if os.path.exists(new_src_filename):
      copyfile(new_src_filename,new_dst_filename)

  @staticmethod
  def deleteFolderContents(path):
    if os.path.exists(path):
       shutil.rmtree(path, ignore_errors=True)
       while os.path.exists(path):
         pass
    os.mkdir(path)

  @staticmethod
  def delete_folder(path):
    if os.path.exists(path):
      shutil.rmtree(path)

  @staticmethod
  def upFolderName(path):
    if os.path.isfile(path):
      return os.path.basename(os.path.dirname(path))
    elif os.path.isdir(path):
      return os.path.basename(path)

  @staticmethod
  def getUpFolder(src_path):
    if os.path.isdir(src_path):
       return src_path[0:-len(FileUtility.upFolderName(src_path))-1]
    elif os.path.isfile(src_path):
       return FileUtility.removeFilename(src_path)
  @staticmethod
  def get_file_upfolder(src_filename):
    return Path(src_filename).parts[-2]

  @staticmethod
  def saveList2File(list,filename):
    f = open(filename,'w')
    for row in list:
      for i, item in enumerate(row) :
        if i > 0 :
          f.write(' | '+item)
        else : f.write(item)
      f.write('\n')


    f.close()

  @staticmethod
  def changeModifiedDate(filename,dst_date):
    date = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    modTime = time.mktime(date.timetuple())

    os.utime(filename, (modTime, modTime))

  @staticmethod
  def createClearFolder(path):
    if os.path.exists(path):
      FileUtility.deleteFolderContents(path)
    else : os.makedirs(path)

  @staticmethod
  def create_folder_if_not_exists(path):
    if not os.path.exists(path):
      os.makedirs(path)


  @staticmethod
  def file2Folder(filename):
    tokens = FileUtility.getFileTokens(filename)
    return os.path.join(tokens[0],tokens[1])

  @staticmethod
  def files2Folders(filenames):
    result = []
    for file in filenames:
      result.append(FileUtility.file2Folder(file))
    return result

  @staticmethod
  def file2DstFolder(filename,src_path,dst_path):
    dst_filename = FileUtility.getDstFilename2(filename,src_path,dst_path)
    return FileUtility.file2Folder(dst_filename)

  @staticmethod
  def files2DstFolders(filenames,src_path,dst_path):
    dst_filenames = []
    for filename in filenames :
      dst_filenames.append( FileUtility.file2DstFolder(filename,src_path,dst_path))

  @staticmethod
  def makeVideoFilesFolders(src_path,dst_path):
    video_src_files = FileUtility.getFolderVideoFiles(src_path)
    video_dst_files = FileUtility.getDstFilenames2( video_src_files,src_path, dst_path)
    folders = FileUtility.files2Folders(video_dst_files)

    FileUtility.makeFolders(folders)
    return folders

  @staticmethod
  def changeFileName(filename,new_name):
    tokens = FileUtility.getFileTokens(filename)
    return os.path.join(tokens[0], new_name + tokens[2])

  @staticmethod
  def changeFileNameEx(filename,new_name = "",pre_fix = "",post_fix=""):
    tokens = FileUtility.getFileTokens(filename)

    dst_name = ""
    if (pre_fix != ""):
      dst_name = pre_fix

    if (new_name != ""):
      dst_name = dst_name + new_name
    else : dst_name = dst_name + tokens[1]

    if (post_fix != ""):
      dst_name = dst_name+ post_fix

    return  os.path.join(tokens[0],dst_name+tokens[2])


  @staticmethod
  def changeFileExt(fileame,new_ext,check_exist = False):
    tokens = FileUtility.getFileTokens(fileame)
    filename = os.path.join(tokens[0],tokens[1]+'.'+new_ext)
    if (check_exist):
      if (not os.path.exists(filename)):
        filename = ""
    return filename

  @staticmethod
  def changeFilesExt(filesname,new_ext):
    result = []
    for filename in filesname:
      result.append(FileUtility.changeFileExt(filename,new_ext))

    return result

  @staticmethod
  def changeFilesExt2(filesname, new_ext, check_exist=False):
    org_filenames = []
    result = []
    for filename in filesname:
      new_filename = FileUtility.changeFileExt(filename, new_ext)
      if (new_filename != ""):
        org_filenames.append(filename)
        result.append(FileUtility.changeFileExt(filename, new_ext))

    return result, org_filenames

  @staticmethod
  def changeFilesExtPair(filesname,new_ext):
    src_filenames = []
    dst_filenames = []
    
    
    for filename in filesname:
        dst_file = FileUtility.changeFileExt(filename,new_ext)
        if os.path.exists(dst_file):
            src_filenames.append(filename)
            dst_filenames.append(dst_file)
            

    return src_filenames,dst_filenames

  @staticmethod
  def changeFilesname(files,prefix,start_counter,pad_count = 7):
    result = []
    counter = start_counter
    for file in files:
      tokens = FileUtility.getFileTokens(file)
      dst_filename = os.path.join( tokens[0] , prefix+Utility.paddingNumber(counter,pad_count)+tokens[2])
      result.append(dst_filename)
      counter += 1

    return result,counter

  @staticmethod
  def changeFilenamePostfix(filename,postfix):
    result = []

    tokens = FileUtility.getFileTokens(filename)
    dst_filename = os.path.join( tokens[0] , tokens[1]+postfix+tokens[2])

    return dst_filename

  @staticmethod
  def changeFilesnamePostfix(files,postfix):
    result = []

    for file in files:
      result.append(FileUtility.changeFilenamePostfix(file,postfix))

    return result



  @staticmethod
  def removePostfixFromFilename(filename,postfix):
      tokens = FileUtility.getFileTokens(filename)
      fname = tokens[1]
      if fname.endswith(postfix):
        fname = fname[:-len(postfix)]
      return os.path.join(tokens[0], fname +tokens[2])

  @staticmethod
  def removeFilenamesPostfix(filenames:List[str],postfix:str = None):
      new_filenames = []
      for filename in filenames:
        new_filenames.append(FileUtility.removeFilename(filename,postfix))

      return new_filenames

  @staticmethod
  def removeFilenamesPostfix(filenames: List[str], postfix: str = None):
    new_filenames = []
    for filename in filenames:
      new_filenames.append(FileUtility.removeFilenamesPostfix(filename, postfix))

    return new_filenames

  @staticmethod
  def removeLastPostfix(my_str:str):
      postfix = FileUtility.getFilenamePostfix(my_str)
      if postfix is not None:
         postfix = "_"+postfix
         my_str = FileUtility.removePostfixFromFilename(my_str, postfix)
      return my_str

  @staticmethod
  def removeListLastPostfix(list1:List[str]):
      result = []
      for my_str in tqdm(list1):
         result.append(FileUtility.removeFilenameLastPostfixStr(my_str))

      return result

  @staticmethod
  def removeFilenameLastPostfix(filename: str):
      new_filename = FileUtility.removeLastPostfix(filename)
      os.rename(filename,new_filename)

      return new_filename


  @staticmethod
  def removeFilenamesLastPostfix(filenames: List[str]):
    result = []
    for filename in tqdm(filenames):
      result.append(FileUtility.removeFilenameLastPostfix(filename))

    return result

  @staticmethod
  def removeDirLastPostfix(src_dir :str):

    filenames = FileUtility.getFolderImageFiles(src_dir)
    return FileUtility.removeFilenamesLastPostfix(filenames)





  @staticmethod
  def filenameToKey(filename:str,postfix:str = None):
    tokens = FileUtility.getFileTokens(filename)
    if postfix is not None and tokens[1].endswith(postfix):
      new_fname = tokens[1][:-len(postfix)]
    else:
      new_fname = tokens[1]
    return  new_fname

  @staticmethod
  def filenamesToKeys(filenames:List[str],postfix:str =None):
    keys = []
    for filename in filenames:
      keys.append(FileUtility.filenameToKey(filename,postfix))

    return keys


  @staticmethod
  def getKeySets(src_keys, dst_keys):
      src_set = set(src_keys)
      dst_set = set(dst_keys)
      shared_keys = src_set.intersection(dst_set)
      unique_src = src_set.difference(dst_set)
      unique_dst = dst_set.difference(src_set)
      return shared_keys, unique_src, unique_dst


  @staticmethod
  def changeFilesnamePrefix(files,pretfix):
    result = []

    for file in files:
      tokens = FileUtility.getFileTokens(file)
      dst_filename = os.path.join( tokens[0] ,pretfix+ tokens[1]+tokens[2])
      result.append(dst_filename)


    return result

  @staticmethod
  def add2Filename(filename,value):
    tokens = FileUtility.getFileTokens(filename)
    return  os.path.join( tokens[0],tokens[1]+value+tokens[2])

  @staticmethod
  def add2Filenames(filenames,value):
    result = []
    for filename in filenames:
      result.append(FileUtility.add2Filename(filename,value))

    return  result

  @staticmethod
  def addCounter2Filename(filename,counter):
    return FileUtility.add2Filename(filename,str(counter) )

  @staticmethod
  def addCounters2Filename(filename,start,end):
    result = []
    tokens = FileUtility.getFileTokens(filename)

    for i in range(start,start+end):
      result.append( os.path.join( tokens[0],tokens[1]+"_"+str(i)+tokens[2]))
    return  result
  
  
  @staticmethod
  def getFilenameWithoutExt(filename):
    tokens = FileUtility.getFileTokens(filename)
    return tokens[1].split('.')[0]


  @staticmethod
  def getFilename(filename):
    tokens = FileUtility.getFileTokens(filename)
    return tokens[1]+tokens[2]

  @staticmethod
  def getFilenames(filenames):
    result = []
    for filename in filenames:
      result.append(FileUtility.getFilename(filename))
    return result
  


  @staticmethod
  def copyFullSubFolders(src_path,dst_path,copy_files_as_folder = False):
      for item in os.listdir(src_path):
        s = os.path.join(src_path, item)
        d = os.path.join(dst_path, item)
        if os.path.isdir(s):
          os.makedirs(d,exist_ok=True)
          FileUtility.copyFullSubFolders(s, d,copy_files_as_folder)
        elif copy_files_as_folder :
          os.makedirs(FileUtility.file2Folder(d), exist_ok=True)

  @staticmethod
  def imagesExt():
    return ['jpg','png','bmp','tif','tiff']

  @staticmethod
  def getImagePair(filename):
     result = None
     for ext in FileUtility.imagesExt():
        image_filename = FileUtility.changeFileExt(filename,ext)
        if os.path.exists(image_filename):
          result = image_filename
          break
     return result

  @staticmethod
  def getImagePairs(filenames):
    result = []
    for file in filenames:
      result.append(FileUtility.getImagePair(file))
    return result


  @staticmethod
  def copyFile(src_filename,dst_filename):
    if os.path.exists(src_filename):
      shutil.copyfile(src_filename, dst_filename)
    
  @staticmethod
  def copyFile2Dst(src_path,dst_path, ext = None):
    src_files = FileUtility.getFolderFiles(src_path,ext)
    dst_files = FileUtility.getDstFilenames2(src_files,src_path,dst_path,copy_in_root)
    FileUtility.copyFilesByName(src_files,dst_files)

  @staticmethod
  def copyFiles2Dst(src_path,dst_path,copy_in_root = True,count = 0, ext = None):
     src_files = FileUtility.getFolderFiles(src_path,ext)

     if count == 0:
       dst_files = FileUtility.getDstFilename2(src_files, src_path, dst_path, True)
       FileUtility.copyFilesByName(src_file,dst_files)
     else :
          new_src_files = random.sample(src_files, count)
          dst_files = FileUtility.getDstFilenames2(new_src_files, src_path, dst_path, True)
          FileUtility.copyFilesByName(new_src_files, dst_files)

  @staticmethod
  def getFilesKeyInfo(src1_dir:str, src2_dir:str,src1_postfix:str = None, src2_postfix:str=None,exclude_keys =None):
    src1_files = FileUtility.getFolderImageFiles(src1_dir, src1_postfix)
    src2_files = FileUtility.getFolderImageFiles(src2_dir, src2_postfix)

    src1_keys = FileUtility.filenamesToKeys(src1_files, src1_postfix)
    src2_keys = FileUtility.filenamesToKeys(src2_files, src2_postfix)

    src1_dict = dict(zip(src1_keys, src1_files))
    src2_dict = dict(zip(src2_keys, src2_files))

    shared_keys, unique_src1, unique_src2 = FileUtility.getKeySets(src1_keys, src2_keys)

    if exclude_keys is not None:
      Utility.remove_items(shared_keys,exclude_keys)
      Utility.remove_items(unique_src1, exclude_keys)
      Utility.remove_items(unique_src2, exclude_keys)

    return src1_dict,src2_dict,shared_keys, unique_src1, unique_src2

  @staticmethod
  def copyImagePairs(src1_dir: str, src2_dir: str, dst1_dir: str, dst2_dir: str, count: int,
                     src1_postfix: str = None, src2_postfix: str = None):

      def copy_files(key, src1_dict, src2_dict, dst1_dir, dst2_dir):
          src1_filename = src1_dict.get(key)
          src2_filename = src2_dict.get(key)

          FileUtility.copy2Path(src1_filename, dst1_dir)
          FileUtility.copy2Path(src2_filename, dst2_dir)

      # Retrieve file info and keys
      (src1_dict, src2_dict, shared_keys, unique_src1,
       unique_src2) = FileUtility.getFilesKeyInfo(src1_dir, src2_dir, src1_postfix, src2_postfix)

      # Select a random subset of keys
      selected_keys = random.sample(shared_keys, count)

      # Clear destination directories
      FileUtility.createClearFolder(dst1_dir)
      FileUtility.createClearFolder(dst2_dir)

      # Define the maximum number of threads
      max_workers = 8  # Adjust based on your system's capabilities

      # Use ThreadPoolExecutor to copy files in parallel
      with ThreadPoolExecutor(max_workers=max_workers) as executor:
          futures = {executor.submit(copy_files, key, src1_dict, src2_dict, dst1_dir, dst2_dir): key for key in
                     selected_keys}

          # Use tqdm to show progress
          for _ in tqdm(as_completed(futures), total=len(futures)):
              pass

  @staticmethod
  def copyImagePairs2(src_dir: str, dst_dir: str,  count: int,separate_branch=True):
      src_branchs = FileUtility.getSubfolders(src_dir)
      if len(src_branchs) != 2:
        return 
      
      src1_dir = os.path.join(src_dir,src_branchs[0])
      src2_dir = os.path.join(src_dir, src_branchs[1])
      
      dst1_dir = os.path.join(dst_dir,src_branchs[0])
      dst2_dir = os.path.join(dst_dir, src_branchs[1])
      
      postfix1 = FileUtility.getFolderPostfix(src1_dir)
      postfix2 = FileUtility.getFolderPostfix(src2_dir)
      
      FileUtility.copyImagePairs(src1_dir,src2_dir,dst1_dir,dst2_dir,count,postfix1,postfix2)

  @staticmethod
  def get_files_by_shared_prefix(filenames1,filenames2,count=0):
      filenames1_key = FileUtility.getFilenamesFirstPostfix(filenames1)
      keys1 = list( filenames1_key.keys())
      filenames2_key = FileUtility.getFilenamesFirstPostfix(filenames2)
      keys2 = list(filenames2_key.keys())

      shared_keys = list(set(keys1) & set(keys2))
      if count != 0:
        shared_keys = shared_keys[:min(count,len(shared_keys))]

      filenames1 = []
      filenames2 = []

      for shared_key in shared_keys:
         filenames1.append(filenames1_key[shared_key][0])
         filenames2.append(filenames2_key[shared_key][0])
          
      return filenames1,filenames2



  @staticmethod
  def copyImagePairesToBranch(src_dir: str, dst_dir: str,branchs: List[str],count =0):
      branch_files = []
      for branch in branchs:
          branch_files.append( FileUtility.getFolderImageFiles(src_dir,branch))

      filenames1,filenames2 = FileUtility.get_files_by_shared_prefix(branch_files[0],branch_files[1],count)
      branch_files = [filenames1,filenames2]
          
      FileUtility.create_folder_if_not_exists(dst_dir)
      branch_titles = []
      for branch in branchs:
        branch_titles.append( branch.lstrip('_'))

      FileUtility.createSubfolders(dst_dir,branch_titles)
      for i,branch in enumerate(branch_titles):
        cur_dst_dir = os.path.join(dst_dir,branch)
        files = branch_files[i]
        if count != 0:
          files = files[:min(count,len(files))]
          
        FileUtility.copyFiles2DstPath(files,cur_dst_dir)
          
  @staticmethod
  def splitImage2GalleryProbe(src_dir: str, dst_dir: str,probes_end:List[str]):
      FileUtility.create_folder_if_not_exists(dst_dir)
      gallery_dir = os.path.join(dst_dir,'gallery')
      probe_dir = os.path.join(dst_dir, 'probe')

      FileUtility.createSubfolders(dst_dir,['gallery','probe'])


      src_filenames = FileUtility.getFolderImageFiles(src_dir)
      for src_filename in tqdm(src_filenames,desc = "split images"):
          last_postfix =  FileUtility.getFilenameLastPostfix(src_filename)
          if last_postfix in probes_end:
            cur_dst_dir = probe_dir
          else:
             cur_dst_dir = gallery_dir

          FileUtility.copy2Path(src_filename,cur_dst_dir)



      





  @staticmethod
  def checkRootFolder(src_path):
    for a , b, c in os.walk(src_path):
      return len(b)== 0


  @staticmethod
  def createDstBrach(dst_path,branchs,clear_dst):
      if clear_dst :
          FileUtility.deleteFolderContents(dst_path)

      if not os.path.exists(dst_path):
        os.mkdir(dst_path)

      for brach in branchs:
          train_path = os.path.join(dst_path,brach)
          if not os.path.exists(train_path):
              os.mkdir(train_path)

  @staticmethod
  def getFilenameFromURL(URL):
      a = urlparse(URL)
      return os.path.basename(a.path)




  @staticmethod
  def downloadURLExtract(URL, dst_path,clear_file = False):

    if not os.path.exists(dst_path):
      os.mkdir(dst_path)

    filename = FileUtility.getFilenameFromURL(URL)

    dst_filename = os.path.join(dst_path, filename)

    if not os.path.exists(dst_filename):
      urllib.request.urlretrieve(URL, dst_filename)


    if FileUtility.changeFileExt(dst_filename,'tar.gz') :
        tar = tarfile.open(dst_filename)
        tar.extractall(dst_path)
        tar.close()

    if clear_file :
      os.remove(dst_filename)

  @staticmethod
  def extractFile(cmp_filename,dst_path =''):

    if dst_path == '':
        dst_path = FileUtility.removeFilename(cmp_filename)

    if not os.path.exists(dst_path):
      os.makedirs(dst_path)


    ext = FileUtility.getFileExt(cmp_filename)

    if ext == 'zip'.lower():
      with zipfile.ZipFile(cmp_filename, "r") as zip_ref:
        zip_ref.extractall(dst_path)


    elif ext =='tar.gz' :
        tar = tarfile.open(cmp_filename)
        tar.extractall(dst_path)
        tar.close()

    return dst_path




  @staticmethod
  def compressFile(src_path, cmp_filename= ''):
    if cmp_filename == '':
      cmp_filename = os.path.join( FileUtility.getUpFolder(src_path),FileUtility.upFolderName(src_path)+'.zip')


    ext = FileUtility.getFileExt(cmp_filename).lower()



    if ext == 'zip':
      cmp_filename = FileUtility.removeFileExt(cmp_filename)

    f = FileUtility.upFolderName(cmp_filename)

    shutil.make_archive(cmp_filename, 'zip',FileUtility.getUpFolder(src_path),f)


  @staticmethod
  def getMediaInfo(media):
    if os.path.isdir(media):
      return MediaType.folder,None
    elif os.path.isfile(media):
      return MediaType.file,FileUtility.getFileExt(media)

  @staticmethod
  def isFolderEmpty(src_path):
    return len(os.listdir(src_path)) != 0


  @staticmethod
  def writeTextList(filename,src_list):
    with open(filename,'w') as f:
      for value in src_list:
        f.write(value +'\n')

      f.close()

  @staticmethod
  def read_text_list(filename):
    result = []
    with open(filename,'r') as f:
      for line in f.readlines():
        result.append(line.strip())
      f.close()
    return result


  @staticmethod
  def getFileCreationDate(path_to_file):

      if platform.system() == 'Windows':
          return os.path.getctime(path_to_file)
      else:
          stat = os.stat(path_to_file)
          try:
              return stat.st_birthtime
          except AttributeError:
              return stat.st_mtime


  @staticmethod
  def getLastFilename(src_path):
    p = os.path.join(src_path,'*')
    list_of_files = glob.glob(p)
    if len(list_of_files) == 0:
      return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

  @staticmethod
  def loadFilenamesLabels(src_path):
    filenames = []
    labels = []
    indexs = []
    counter = 0
    sub_folders = FileUtility.getSubfolders(src_path)
    for sub_folder in sub_folders:
      cur_path = os.path.join(src_path,sub_folder)
      cur_filenames = FileUtility.getFolderImageFiles(cur_path)
      filenames.extend(cur_filenames)
      labels.extend([sub_folder] * len(cur_filenames))
      indexs.extend([counter] * len(cur_filenames))
      counter += 1

    return filenames,labels


  @staticmethod
  def compareFolderFiles(src_path1,src_path2,copy_unpair = False,dst_path = None):
    files1 = FileUtility.getFolderAllFiles(src_path1)
    files2 = FileUtility.getFolderAllFiles(src_path2)

    pair_files = []
    unpair_files = []

    for i,filename1 in enumerate(files1):
      f_name1 = FileUtility.getFilename(filename1)
      filename2 = FileUtility.getDstFilename2(filename1,src_path1,src_path2)

      if os.path.exists(filename2):
        pair_files.append(filename1)
      else :unpair_files.append(filename1)

    if copy_unpair :
       dst_files = FileUtility.getDstFilenames2(unpair_files,src_path1,dst_path)
       FileUtility.copyFilesByName(unpair_files,dst_files)

    equal = len(unpair_files) == 0
    return  equal, pair_files,unpair_files

  @staticmethod
  def copy_file_to_folder(src_filename,dst_path,postfix = '',create_dst_branch = False):
    tokens = FileUtility.getFileTokens(src_filename)
    f_name = tokens[1]+postfix+tokens[2]

    if create_dst_branch:
      dst_branch = os.path.join(dst_path, tokens[1])
      FileUtility.createClearFolder(dst_branch)
      dst_filename = os.path.join(dst_branch, f_name)

    else:
      dst_filename = os.path.join(dst_path,f_name)

    FileUtility.copyFile(src_filename, dst_filename)

    if create_dst_branch:
      return dst_branch
    else: return None

  @staticmethod
  def copy_files_to_folder(src_filenames,dst_path,postfix = '',create_dst_branch = False):
    for src_filename in tqdm(src_filenames):
       FileUtility.copy_file_to_folder(src_filename,dst_path,postfix,create_dst_branch)

  @staticmethod
  def cvRect2Openvion(cv_rect):
    pass
  @staticmethod
  def openvinoRect2Cv(ov_rect):
    pass

  @staticmethod
  def remove_blank_subfolder(src_path):

    sub_folders = FileUtility.getSubfolders(src_path)
    for sub_folder in sub_folders:
      cur_path =os.path.join(src_path, sub_folder)
      files = FileUtility.getFolderImageFiles(cur_path)
      if len(files) == 0:
        os.removedirs(cur_path)

  @staticmethod
  def delete_files(files_name):
    for file_name in tqdm(files_name):
      if os.path.exists(file_name):
        os.remove(file_name)

  @staticmethod
  def delete_subfolder_extra_files(src_path,max_count):
    sub_folders = FileUtility.getSubfolders(images_path)
    l = len(sub_folders)
    for i in tqdm(range(l), ncols=100):
      sub_folder = sub_folders[i]
      cur_path = os.path.join(images_path, sub_folder)
      files = FileUtility.getFolderImageFiles(cur_path)
      len1 = len(files)
      if len1 > count:
        for i in range(count, len1):
          os.remove(files[i])

  @staticmethod
  def remove_token_from_filename(src_path, token_list_id,token_count = -1,sep='_'):

    files = FileUtility.getFolderImageFiles(src_path)
    for file in files:
      tokens = FileUtility.getFileTokens(file)
      f_tokens = tokens[1].split(sep)
      if token_count == -1 or( token_count !=-1 and len(f_tokens) == token_count):
        fname = ''
        for i, f_token in enumerate(f_tokens):
           if not( i in token_list_id):
             if fname != '':
               fname += sep
             fname += tokens[i]


        fname_ext = fname + tokens[2]
        dst_filename = os.path.join(tokens[0],fname_ext)
        if os.path.exists(dst_filename):
          for i in range(100):
            fname_ext = fname + sep + str(i + 1) + tokens[2]
            dst_filename = os.path.join(tokens[0],fname_ext)
            if not os.path.exists(dst_filename):
              break
        os.rename(file, dst_filename)

  @staticmethod
  def getFolderFilesCount(src_path):
    _, _, files = next(os.walk(src_path))
    return len(files)

  @staticmethod
  def get_folder_images_count(src_path):
    _, _, files = next(os.walk(src_path))
    images = []
    for filename in files:
      if FileUtility.checkIsImage(filename):
        images.append(filename)
    return len(images)

  @staticmethod
  def subfolders_exist(src_path,sub_folders):
    result = True
    for sub_folder in sub_folders:
      cur_folder = os.path.join(src_path,sub_folder)
      if not os.path.exists(cur_folder):
        result = False
        break
    return result

  @staticmethod
  def copy_branch_percent(src_path,dst_path,per = 1.0,random_copy = False):
    FileUtility.createClearFolder(dst_path)
    sub_folders = FileUtility.getSubfolders(src_path)
    for sub_folder in sub_folders:
      cur_src_folder = os.path.join(src_path,sub_folder )
      files_count = FileUtility.getFolderImagesCount(cur_src_folder)
      if files_count == 0:
        cur_dst_folder = os.path.join(dst_path,sub_folder)
        os.makedirs(cur_dst_folder)
        FileUtility.copy_branch_percent(cur_src_folder,cur_dst_folder,per,random_copy)
      else:
        cur_dst_folder = os.path.join(dst_path, sub_folder)
        FileUtility.copy_images_percent(cur_src_folder,cur_dst_folder,per,random_copy)


  @staticmethod
  def get_folder_imagefiles_group(src_path,branch = None):
    result = []
    sub_folders = FileUtility.getSubfolders(src_path)
    if branch != None and branch in sub_folders:
      cur_src_folder = os.path.join(src_path, branch)
      files = FileUtility.getFolderImageFiles(cur_src_folder)
      result.append(files)
    else:
      for sub_folder in sub_folders:
        cur_src_folder = os.path.join(src_path, sub_folder)
        files_count = FileUtility.get_folder_images_count(cur_src_folder)
        if files_count != 0:
          files = FileUtility.getFolderImageFiles(cur_src_folder)
          result.append(files)

    return result

  @staticmethod
  def get_folder_imagefiles_per(src_path,branch = 'train',train_per= 0.7):
    result = []
    if FileUtility.subfolders_exist(src_path,['train','test']):
       result = FileUtility.get_folder_imagefiles_group(src_path,branch)[0]
    else :
      files_list = FileUtility.get_folder_imagefiles_group(src_path)
      if len(files_list) == 0:
        files_list.append(FileUtility.getFolderImageFiles(src_path))
      if branch == 'train':
        for files in files_list:
          count = int(len(files) * train_per)
          result.extend(files[:count])
      elif branch == 'test':
        for files in files_list:
          count = int(len(files) * train_per)
          result.extend(files[count:])

    return result

  @staticmethod
  def duplicate_filenames(src_path,dst_path,count):
    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files,src_path,dst_path)
    result = []

    for i,src_file in enumerate(src_files):
      dst_file = dst_files[i]
      res = []
      for j in range(count):
           res.append( FileUtility.changeFilenamePostfix(dst_file, '_'+str(j)))
      result.append(res)

    return result


  @staticmethod
  def duplicate_images(src_path,dst_path,count):
    src_files = FileUtility.getFolderImageFiles(src_path)
    result = FileUtility.duplicate_filenames(src_path,dst_path,count)
    for i in tqdm(range(len(result)), ncols=100):
      res = result[i]
      src_file = src_files[i]
      for dst_filename in res:
         FileUtility.copyFile(src_file,dst_filename)

  @staticmethod
  def augment_filenames(filenames,aug_count= 1,shuffle = False):
    result = []
    for filename in filenames:
      result.append(filename)
      for i in range(aug_count):
         result.append(filename)
    random.shuffle(result)
    return result

  @staticmethod
  def replace_in_file(filename,find_str,replace_str):
    with open(filename) as file:
      s = file.read()

    s = s.replace(find_str, replace_str)
    with open(filepath, "w") as file:
      file.write(s)

  @staticmethod
  def replace_in_folder(folder,find_str,replace_str,exts):
    all_files = FileUtility.getFolderFiles(folder,exts)
    for i in tqdm(range(len(all_files)), ncols=100):
      FileUtility.replace_in_file(all_files[i],find_str,replace_str)

  @staticmethod
  def exists_nonzero(filename):
    return os.path.exists(filename) and os.path.getsize(filename) > 0

  @staticmethod
  def exists_nonzeros(filenames):
    result = True
    for filename in filenames:
      if not FileUtility.exists_nonzero(filename):
        result = False
        break
    return result

  @staticmethod
  def get_slash_path(path : str):
    result = path
    result = result.replace("\\\\", '/')
    result = result.replace('\\','/')
    return result

  @staticmethod
  def pop_by_filename(input_files,filename):
    files = []
    target_files = []
    for input_file in input_files:
      fname = FileUtility.getFilename(input_file)
      if fname == filename:
        target_files.append(input_file)
      else : files.append(input_file)

    return files,target_files

  @staticmethod
  def get_next_path(dst_path,branch_name):
    i = 0
    cur_path = os.path.join(dst_path,branch_name+str(i))
    while os.path.exists(cur_path):
      i +=1
      cur_path = os.path.join(dst_path, branch_name + str(i))

    return cur_path

  @staticmethod
  def copy_files_to_branchs(src_files,dst_path, branchs):
    FileUtility.createSubfolders(dst_path,branchs)
    for branch in branchs:
        branch_path = os.path.join(dst_path,branch)
        dst_filenames = FileUtility.getDstFilenames(src_files,branch_path)
        FileUtility.copyFilesByName(src_files,dst_filenames)

  @staticmethod
  def copy_files_to_branchsex(src_path,dst_path,count_in_branch):
    FileUtility.createClearFolder(dst_path)
    src_branchs = FileUtility.getSubfolders(src_path)
    for src_branch in src_branchs:
      dst_branch_path = os.path.join(dst_path,src_branch)
      FileUtility.createClearFolder(dst_branch_path)
      src_branch_path = os.path.join(src_path,src_branch)
      src_files = FileUtility.getFolderImageFiles(src_branch_path)
      min_count = min(count_in_branch,len(src_files))
      FileUtility.copyFiles2Dst(src_branch_path,dst_branch_path,True,min_count)
  @staticmethod
  def absPath(input_path):
    current_dir = os.getcwd()
    return  os.path.join(current_dir,input_path)

  @staticmethod
  def copy2Path(filename, dst_path):
    fname = FileUtility.getFilename(filename)
    dst_filename = os.path.join(dst_path,fname)
    FileUtility.copyFile(filename,dst_filename)


  @staticmethod
  def count_files_by_extension(folder_path):
    file_counts = defaultdict(int)
    for filename in os.listdir(folder_path):
      if os.path.isfile(os.path.join(folder_path, filename)):
        extension = os.path.splitext(filename)[1][1:].lower()
        file_counts[extension] += 1
    return file_counts

  @staticmethod
  def compare_extension_counts(folder_path, extensions):
    if not extensions or len(extensions) < 2:
      raise ValueError("At least two extensions must be provided for comparison.")

    file_counts = FileUtility.count_files_by_extension(folder_path)
    max_count = 0
    max_ext = None

    for ext in extensions:
      ext_count = file_counts.get(ext, 0)
      if ext_count > max_count:
        max_count = ext_count
        max_ext = ext
      elif ext_count == max_count:
        max_ext = None  # Tie, return None if counts are equal

    return max_ext

  @staticmethod
  def copy_gt_file(src_dir, dst_dir, ext, prefix=None, postfix=None):
    src_filenames = FileUtility.getFolderFiles(src_dir, [ext])
    dst_filenames = FileUtility.getDstFilenames2(src_filenames, src_dir, dst_dir)
    if prefix:
      dst_filenames = FileUtility.changeFilesnamePrefix(dst_filenames, prefix)

    if postfix:
      dst_filenames = FileUtility.changeFilesnamePostfix(dst_filenames, postfix)

    FileUtility.copyFilesByName(src_filenames, dst_filenames)

  @staticmethod
  def rename_replace_files(src_dir,find_str,replace_str):
    src_filenames = FileUtility.getFolderImageFiles(src_dir)
    for src_filename in tqdm(src_filenames):
      dst_filename = src_filename.replace(find_str, replace_str)
      os.rename(src_filename, dst_filename)
  
  @staticmethod
  def remove_postfix(src_path, postfix):
    src_files = FileUtility.getFolderFiles(src_path)
    for src_file in tqdm(src_files):
      new_src_file = src_file.replace(postfix, '')
      if os.path.exists(new_src_file):
        os.remove(src_file)
      else:
        os.rename(src_file, new_src_file)

  @staticmethod
  def has_folders_equal_files(dir1,dir2):
    files1 = FileUtility.getFoldersFiles(dir1)
    files1 = FileUtility.getFilenames(files1)

    files2 = FileUtility.getFoldersFiles(dir2)
    files2 = FileUtility.getFilenames(files2)
    return sorted(files1) == sorted(files2)

  @staticmethod
  def copy_images_from_paths(src_paths, dst_path,pair_files_exts = [], file_counter = 1):
    FileUtility.create_folder_if_not_exists(dst_path)


    # List to store destination file paths
    copied_files = []

    # Iterate through each source path
    for src_path in src_paths:
      src_files = FileUtility.getFolderImageFiles(src_path)

      # Copy each file with numbered filename
      for src_file in tqdm(src_files):
        # Get file extension
        tokens = FileUtility.getFileTokens(src_file)
        image_ext = tokens[2]

        # Create new filename with 7-digit zero-padded counter
        new_filename = f"{file_counter:07d}{image_ext}"
        dst_file = os.path.join(dst_path, new_filename)
        FileUtility.copyFile(src_file, dst_file)

        for ext in pair_files_exts:
            pair_filename = os.path.join(tokens[0],tokens[1]+'.'+ext)
            if os.path.exists(pair_filename):
              new_filename = f"{file_counter:07d}.{ext}"
              dst_file = os.path.join(dst_path, new_filename)
              FileUtility.copyFile(pair_filename, dst_file)



        # Add to copied files list
        copied_files.append(dst_file)

        # Increment counter
        file_counter += 1

    return copied_files










  













