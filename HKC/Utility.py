from .MatUtility import *
import  random
import  os
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
