from .MatUtility import *
import  random
import  os
from enum import Enum


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




