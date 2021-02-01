import abc


class GTItem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

class GTData(metaclass=abc.ABCMeta):
    def __init__(self):
        self.objects_ = []
        self.filename_ = ""
        self.size_ = (0,0,0)

    # @abc.abstractmethod
    # def add(self,region,label_id):
    #     self.objects_.append(region,label_id)

    def clear(self):
        self.objects_.clear()

    def delete(self,index):
        self.objects_.remove(index)


