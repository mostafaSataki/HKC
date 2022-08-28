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
    def getGTIndexs(count,valid_per = 0.2,test_per = 0.2, index_type = IndexType.random):
        total_indexs = GTUtility.getIndexs(count,index_type)
        train_per = 1 - valid_per - test_per
        train_indexs = total_indexs[0:int(train_per*count)]
        valid_indexs = total_indexs[int(train_per*count):int((train_per+valid_per)*count)]
        test_indexs = total_indexs[int((train_per+valid_per)*count):]

        return train_indexs, valid_indexs, test_indexs
