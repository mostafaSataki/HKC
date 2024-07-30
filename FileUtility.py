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
  def getFolderImageFiles(path):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
        if FileUtility.checkIsImage(filename):
          result.append(os.path.join(dirpath, filename))

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

    src_files = FileUtility.getDstFilenames2(pattern_files,pattern_path,src_path,pair_extension)
    dst_files = FileUtility.getDstFilenames2(pattern_files, pattern_path, dst_path,pair_extension)

    for i in tqdm(range(len(pattern_files)), ncols=100):
      if os.path.exists(src_files[i]):
         FileUtility.copyFile(src_files[i], dst_files[i])
      if cut_flag:
        if os.path.exists(src_files[i]):
          os.remove(src_files[i])



  @staticmethod
  def copyFilesByName(src_filenames,dst_filenames):
    for i in tqdm(range(len(src_filenames)), ncols=100):
      if src_filenames[i] != dst_filenames[i]:
        if os.path.exists(src_filenames[i]):
          shutil.copyfile(src_filenames[i], dst_filenames[i])

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
  def getDstFilenames2(src_filenames, dst_path,src_path = None, copy_to_root = False,dst_extension = None):
    result = []
    for src_filename in src_filenames:
      result.append(FileUtility.getDstFilename2(src_filename,dst_path,src_path,copy_to_root,dst_extension))

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
  def copy2Path(filenamee,dst_path):
    fname = FileUtility.getfilename(filenamee)
    dst_filename = os.path.join(dst_path,fname)
    FileUtility.copyFile(filenamee,dst_filename)








  













