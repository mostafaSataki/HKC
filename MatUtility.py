import random

class MatUtility:
  @staticmethod
  def percentsCount(percents,count):
     return [ int(count * per) for per in percents]

  @staticmethod
  def getCDF(counts):
    if len(counts) == 0:
      return None
    result = [counts[0]]
    for i in range(1,len(counts)):
      result.append(counts[i-1] + counts[i])

    return result

  @staticmethod
  def getRandomIndexs(count,per = 1.0):
    list1 = list(range(count))
    random.shuffle(list1)
    count = int(len(list1) * per)
    return list1[:count]
  
  
  
  



