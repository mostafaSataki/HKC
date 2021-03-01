from HKC.DetectionTF import *
from HKC.GTDetection import *

deploy_path = r'E:\deep_video\Compressed\Hande-State\deploy2'
size = (320,320)

def train():
    org_images_path = r'E:\deep_video\Compressed\Hande-State\Final_DB'
    labels = ['hand']
    train_per = 0.8

    detector = DetectionTF(deploy_path)

    detector.createTFRecord(org_images_path,labels, size,train_per)
    # detector.train()


def test():
    pass

def freeze():
    pass


if __name__ == '__main__':
    train()
    # test()
    # freeze()

