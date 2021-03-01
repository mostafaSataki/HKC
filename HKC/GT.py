from .GTClassification import *
from  .GTDetection import *

import  random

class GTUtility:
    @staticmethod
    def getIndexs(count,index_type = IndexType.random):
        if index_type == IndexType.end_total or index_type == IndexType.end_branch:
            list1 = list(range(count - 1, -1, -1))
        else:
            list1 = list(range(count))
            if index_type == IndexType.random :
                random.shuffle(list1)
        return list1

    @staticmethod
    def getGTIndexs(count,per = 1.0, index_type = IndexType.random):
        total_indexs = GTUtility.getIndexs(count,index_type)
        train_count = int(count * per)
        train_indexs = total_indexs[:train_count]
        test_indexs = total_indexs[train_count:]
        return train_indexs, test_indexs
