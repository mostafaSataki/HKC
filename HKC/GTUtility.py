from .FileUtility import *
from .CvUtility import *
from .MatUtility import *

class GTUtility:
    @staticmethod
    def getGTRandomIndexs(count, train_per=0.8):
        total_indexs = MatUtility.getRandomIndexs(count)
        train_count = int(count * train_per)
        train_indexs = total_indexs[:train_count]
        test_indexs = total_indexs[train_count:]
        return train_indexs, test_indexs