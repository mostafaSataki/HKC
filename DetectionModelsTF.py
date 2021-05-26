
import enum

class DetectorType(enum.Enum):
    MobileNet2 = 1


class DetectionModels:
    @staticmethod
    def getModel(type, size, classes_count):
        if type == DetectorType.MobileNet2:
            pass
           # return ClsModels.getSimpleModel(size,classes_count)

