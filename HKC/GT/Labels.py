import os
class Labels:
    def __init__(self):
        self.items_ = []

    def add(self,value):
        if not value in self.items_ :
          self.items_.append(value)

    def getIndex(self,value):
        return self.items_.index(value)
    def getLabel(self,index):
        return self.items_[index]

    def load(self,save_path):
        filename = os.path.join(save_path, 'classes.txt')
        with open(filename,'r') as file:
            lines = file.readlines()
            for line in lines:
                self.items_.append(line.rstrip())
            file.close()


    def save(self,save_path):
        filename = os.path.join(save_path,'classes.txt')
        with open(filename,'w') as file:
            for item in self.items_:
                file.write(item+'\n')
            file.close()
