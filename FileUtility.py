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

class MediaType(Enum):
  folder = 1
  file = 2


class FileUtility:
  @staticmethod
  def getImageExtensions():
    return ['bmp', 'jpg',  'tif', 'tiff','png']

  @staticmethod
  def getVideoExtensions():
    return ['mp4', 'avi',  'mov']


  @staticmethod
  def removeFileExt(src_filename):
    return src_filename[0:-len(FileUtility.getFileExt(src_filename))-1]

  @staticmethod
  def removeFilename(src_filename):
     p,_,_ =  FileUtility.getFileTokens(src_filename)
     return p


  @staticmethod
  def checkIsImage(filename):
    ext = FileUtility.getFileExt(filename).lower()
    return  ext in FileUtility.getImageExtensions()



  @staticmethod
  def checkIsVideo(filename):
    ext = FileUtility.getFileExt(filename).lower()
    return  ext in FileUtility.getVideoExtensions()


  def getTifExtensions():
    return ['tif','tiff']

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

  @staticmethod
  def getFolderFiles(path, ext = None, has_path = True, has_ext = True):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
          fname = FileUtility.getFilenameWithoutExt(filename)
          cur_ext = FileUtility.getFileExt(filename)
          if ext != None and cur_ext != ext:
            continue

          if has_path :
            name = os.path.join(dirpath,fname)
          else :name = fname

          if has_ext:
            name = name+'.'+cur_ext
          result.append(name)
          # result.append(os.path.join(dirpath,filename))

    return  result

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
  def copyFiles(src_path, dst_path, pattern_path,cut_flag=False,pair_extension = None):
    pattern_files = glob.glob(os.path.join(pattern_path, "*"))

    # for pattern_file in pattern_files:
    for i in tqdm( range(len(pattern_files)),ncols=100):
      pattern_file = pattern_files[i]
      file_name = os.path.split(pattern_file)[-1]
      src_filename = os.path.join(src_path, file_name)
      dst_filename = os.path.join(dst_path, file_name)
      if os.path.exists(src_filename):
        shutil.copyfile(src_filename, dst_filename)
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
  def copyFilesByName(src_filenames,dst_filenames):
    for i in tqdm(range(len(src_filenames)), ncols=100):
      if src_filenames[i] != dst_filenames[i]:
        shutil.copyfile(src_filenames[i], dst_filenames[i])

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
  def getDstFilename2(src_filename, src_path, dst_path, copy_to_root = False):
    if copy_to_root:
      tokens = FileUtility.getFileTokens(src_filename)
      return os.path.join(dst_path,tokens[1]+tokens[2]) 
    else :
      filename = src_filename
      filename = filename.replace(src_path,'')
      return dst_path+filename

  @staticmethod
  def getDstFilenames2(src_filenames, src_path, dst_path, copy_to_root = False):
    result = []
    for src_filename in src_filenames:
      result.append(FileUtility.getDstFilename2(src_filename,src_path,dst_path,copy_to_root))

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
  def renameFiles(path, prefix , start_counter= 0,replace_folder_name = False):
    if replace_folder_name :
      sub_folders = FileUtility.getSubfolders(path)
      for sub_folder in sub_folders:
        full_path = os.path.join(path,sub_folder)
        counter = start_counter
        image_files = FileUtility.getFolderImageFiles(full_path)
        for file in image_files:
          tokens = FileUtility.getFileTokens(file)
          dst_fille = os.path.join( tokens[0], sub_folder+'_' + str(counter)+tokens[2])
          os.rename(file, dst_fille)
          counter += 1

    else :

        image_files =  FileUtility.getFolderImageFiles(path)
        counter = start_counter
        for file in image_files:
          tokens = FileUtility.getFileTokens(file)
          dst_fille = os.path.join( tokens[0], prefix + str(counter)+tokens[2])
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
  def changeFileExt(fileame,new_ext):
    tokens = FileUtility.getFileTokens(fileame)
    return os.path.join(tokens[0],tokens[1]+'.'+new_ext)

  @staticmethod
  def changeFilesExt(filesname,new_ext):
    result = []
    for filename in filesname:
      result.append(FileUtility.changeFileExt(filename,new_ext))

    return result

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
  def changeFilesnamePostfix(files,postfix):
    result = []

    for file in files:
      tokens = FileUtility.getFileTokens(file)
      dst_filename = os.path.join( tokens[0] , tokens[1]+postfix+tokens[2])
      result.append(dst_filename)


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
    return tokens[1]

  @staticmethod
  def getFilename(filename):
    tokens = FileUtility.getFileTokens(filename)
    return tokens[1]+tokens[2]


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
  def copyFile(src_filename,dst_filename):
    shutil.copyfile(src_filename, dst_filename)
    
  @staticmethod
  def copyFile2Dst(src_path,dst_path, ext = None):
    src_files = FileUtility.getFolderFiles(src_path,ext)
    dst_files = FileUtility.getDstFilenames2(src_files,src_path,dst_path,copy_in_root)
    FileUtility.copyFilesByName(src_files,dst_files)

  @staticmethod
  def copyFiles2Dst(src_path,dst_path,copy_in_root = True,ext = None):
     src_files = FileUtility.getFolderFiles(src_path,ext)
     dst_files = FileUtility.getDstFilename2(src_files,src_path,dst_path,True)
     FileUtility.copyFiles(src_file,dst_files)

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







