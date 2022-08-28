from .MatUtility import *
import  random
import  os
from enum import Enum
from urllib.parse import urlparse
import sys
import datetime
import numpy as np
from operator import itemgetter
import threading

class IndexType(Enum):
  begin_total = 1
  end_total = 2
  begin_branch = 3
  end_branch = 4
  random = 5

class Utility:

  @staticmethod
  def splitList(list,lables):

    counts = MatUtility.percentsCount(lables.values(), len(list))
    pos = 0
    result = []
    cur_list = []
    cdf_counts = MatUtility.getCDF(counts)
    for i, l in enumerate(list):
      if i < cdf_counts[pos]:
        cur_list.append(list[l])
      else:
        result.append(cur_list)
        cur_list = [list[l]]
        pos += 1

    return result


  @staticmethod
  def getDictValues(dict):
    result = []
    for key, values in dict.items():
      for value in values:
        result.append(value)

    return result

  @staticmethod
  def getUniqueValues(list):
    result = []
    for l in list:
      if l not in result:
        result.append(l)
    return result

  @staticmethod
  def saveList(list,filename):
    f = open(filename, 'w')
    for value in list:
      f.write(value + '\n')

    f.close()

  @staticmethod
  def saveDictUniqueValues2File(dict, filename):
    values = Utility.getDictValues(dict)
    unique_values = Utility.getUniqueValues(values)
    unique_values.sort()
    Utility.saveList(unique_values,filename)

  @staticmethod
  def saveDict2File(dict,filename):
    f = open(filename,'w')
    for key,values in dict.items():
       f.write(key + '\n')
       for value in values:
          f.write('\t'+value+'\n')

    f.close()

  @staticmethod
  def getRandomList(total_count):
    list1 = list(range(total_count))
    random.shuffle(list1)
    return list1

  @staticmethod
  def str2Int(str):
    try:
      return [int(str), True]
    except ValueError:
      return [0, False]

  @staticmethod
  def range2StrList(lower,upper):
    return list(map(str,list(range(lower,upper))))

  @staticmethod
  def paddingNumber(number, pad_count=8):
    str1 = '{0:0' + str(pad_count) + 'd}'
    return str1.format(number)
  


  @staticmethod
  def getListByIndexs(src_list,indexs_list):
      result = []
      for i,index in enumerate( indexs_list) :
        result.append(src_list[indexs_list[i]])
      return result


  @staticmethod
  def copyImagesAsPer(src_path, dst_path, per=1.0, copy_to_root=False, select_type=IndexType.random, clear_dst=False):
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

        else:
          dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames, src_cur_branch, dst_cur_branch,
                                                             copy_to_root)


        FileUtility.copyFilesByName(src_image_filenames, dst_image_filenames)



    else:
      image_filenames, gt_filenames = GTUtilityDET.getGtFiles(src_path)
      indexs, _ = GTUtility.getGTIndexs(len(image_filenames), per, select_type)

      src_image_filenames, _ = GTUtilityDET.getGTFiles(image_filenames, gt_filenames, indexs)

      dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames, src_path, dst_path, copy_to_root)


      FileUtility.copyFilesByName(src_image_filenames, dst_image_filenames)

  @staticmethod
  def getIndex(label,labels):
    try:
      index_value = labels.index(label)
    except ValueError:
      index_value = -1
    return index_value

  @staticmethod
  def readField(str,field_value):
     pos = str.find(field_value)

     flag = False
     result = ""

     if pos != -1 :
       result =  str[pos + len(field_value):-1]
       flag = True

     return flag,result

  @staticmethod
  def getPlatform():
    platforms = {
      'linux1': 'Linux',
      'linux2': 'Linux',
      'darwin': 'OS X',
      'win32': 'Windows'
    }
    if sys.platform not in platforms:
      return sys.platform

    return platforms[sys.platform]

  @staticmethod
  def isWindows():
     return Utility.getPlatform() == 'Windows'

  @staticmethod
  def isLinux():
    return Utility.getPlatform() == 'Linux'

  @staticmethod
  def add2PythonPath(env_list):
    pass

  @staticmethod
  def lowerStrList(items):
      return [x.lower() for x in items]

  @staticmethod
  def upperStrList(items):
    return [x.lower() for x in items]

  @staticmethod
  def matchLists(items1, items2):
    if len(items1) != len(items2):
      return False

    list_1 = Utility.lowerStrList(items1)
    list_2 = Utility.lowerStrList(items2)

    res = set(list_1) & set(list_2)

    return len(res) == len(items1)

  @staticmethod
  def sortListByIndexs(listA, indexs):
    return [x for _, x in sorted(zip(indexs, listA))]

  @staticmethod
  def strList2Indexs(listA):
    result = []
    for l in listA:
      result.append(int(l))
    return result

  @staticmethod
  def getNowStr():
     now = datetime.datetime.now()
     return now.strftime("D-%Y-%m-%d-T-%H-%M-%S")

  @staticmethod
  def getNumericLabels(str_labels):
      unique_labels = list(set(str_labels))
      unique_labels = sorted(unique_labels)

      labels_tuple = {}
      labels_tuple_inv = {}
      for i,u in enumerate( unique_labels):
        labels_tuple[i] = u
        labels_tuple_inv[u] = i

      result = []
      for str_label in str_labels:
          result.append( labels_tuple_inv[str_label])

      return result,labels_tuple

  @staticmethod
  def breakList(src_list,n):
    l2 = np.array_split(range(len(src_list)), n)
    result = []
    for i in range(n):
       l3 = l2[i].tolist()
       result.append( itemgetter(*l3)(src_list))
    return result





