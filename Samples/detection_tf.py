from HKC.DetectionTF import *
from HKC.GTDetection import *
from HKC.InstallDetectionTF import *


 #In Windows download the appropriate version from https://github.com/protocolbuffers/protobuf/releases and extract it.
#Copy the protoc.exe path below.
protoc_filename = r''
object_detection_path = r''

deploy_path = r'E:\deep_video\Compressed\Hande-State\deploy2'
object_detection_path = r'D:\library\tf_object_detection'

size = (320,320)

def train():
    org_images_path = r'E:\deep_video\Compressed\Hande-State\Final_DB'
    labels = ['hand']
    train_per = 0.8


    DetectionTF.addModel('model1','')

    detector = DetectionTF(object_detection_path, deploy_path)

    detector.selectModel('model1')
    detector.createTFRecord(org_images_path,labels, size,train_per)
    detector.train()


def test():
    detector = DetectionTF(object_detection_path,deploy_path)
    detector.test(r'',r'')


def freeze():
    pass


if __name__ == '__main__':
    # InstallDetectionTF.install(object_detection_path,protoc_filename)
    train()
    # test()
    # freeze()

