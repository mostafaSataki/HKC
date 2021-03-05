from HKC.ClassificationTF import *
from HKC.ClassificationModelsTF import *
from keras.optimizers import Adam


deploy_path = r'E:\deep_video\Compressed\Hande-State\deploy'
org_classes = ['Fist', 'Like', 'One', 'Palm', 'Two']
delimiter = ','
size = (32, 32)
norm = True
gray = True
float_type = True

def train():
    org_images_path = r'E:\deep_video\Compressed\Hande-State\crop_db_resize'
    train_per = 0.8


    classifier = ClassificationTF(deploy_path,org_classes, delimiter)

    classifier.resizeBatch(org_images_path, size)

    classifier.createGtFiles(train_per)

    classifier.loadDataset(True,True,size=size)
    classifier.selectModel(ClassifierType.Simple1)
    classifier.train(Adam(lr=0.0001))


def test():
    classifier = ClassificationTF(deploy_path, org_classes)
    org_images_path = r'E:\deep_video\Compressed\Hande-State\crop_db_resize'
    classifier.inferenceBatch(org_images_path,size,org_classes, norm,gray,float_type)


def freeze():
    classifier = ClassificationTF(deploy_path, org_classes)
    classifier.freezeModel(deploy_path)


if __name__ == '__main__':
    train()
    # test()
    # freeze()
