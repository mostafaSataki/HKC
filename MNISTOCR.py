import  os
import struct
import sys
import numpy as np
from array import array
import  cv2
from  HKC import  FileUtility
from tqdm import tqdm
class MNISTOCR:


    def read(self, src_path, part,dst_size = None):
        if part is "training":
            if os.path.exists(os.path.join(src_path, "train-images.idx3-ubyte")):
                fname_img = os.path.join(src_path, "train-images.idx3-ubyte")
                fname_lbl = os.path.join(src_path, "train-labels.idx1-ubyte")
            else:
                fname_img = os.path.join(src_path, "train-images-idx3-ubyte")
                fname_lbl = os.path.join(src_path, "train-labels-idx1-ubyte")


        elif part is "testing":
            if os.path.exists(os.path.join(src_path, "t10k-images.idx3-ubyte")):
                fname_img = os.path.join(src_path, "t10k-images.idx3-ubyte")
                fname_lbl = os.path.join(src_path, "t10k-labels.idx1-ubyte")
            else:
                fname_img = os.path.join(src_path, "t10k-images-idx3-ubyte")
                fname_lbl = os.path.join(src_path, "t10k-labels-idx1-ubyte")

        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        flbl = open(fname_lbl, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = array("b", flbl.read())
        flbl.close()


        count = 1
        fimg = open(fname_img, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))

        image_bytes = fimg.read(784)
        labels = lbl.tolist()
        images = []
        # print(len(labels))
        for i in tqdm(range(len(labels)), ncols=100):
        # while image_bytes:
            image = np.zeros((28, 28, 1), np.uint8)
            image_unsigned_char = struct.unpack("=784B", image_bytes)
            for i in range(784):
                image.itemset(i, image_unsigned_char[i])
            # image_save_path = r"%s\%d.png" % (images_save_folder, count)
            if dst_size != None:
                image = cv2.resize(image, dst_size)
            images.append(image)
            # cv2.imshow("view", image)
            # cv2.waitKey(0)

            # cv2.imwrite(image_save_path, image)
            # print(count)
            image_bytes = fimg.read(784)
            count += 1
        fimg.close()



        return images,labels
    def saveImages(self,images,labels,dst_path):
        counter = 0
        for i in tqdm(range(len(images)), ncols=100):
            image = images[i]
            label = labels[i]
            filename = os.path.join(os.path.join(dst_path,str(label)),str(counter)+'.jpg')


            cv2.imwrite(filename,image,[int(cv2.IMWRITE_JPEG_QUALITY), 40])
            # print(filename)
            counter += 1

    def readTrainPart(self, src_path,dst_size = None):
        return self.read(src_path, "training",dst_size)

    def readTestPart(self, src_path,dst_size = None):
        return self.read(src_path,  "testing",dst_size)

    def save(self,images,labels,dst_path,part):
        part_path = os.path.join(dst_path, part)
        FileUtility.makeFoldersBranch(part_path, [str(x) for x in range(10)], True)
        self.saveImages(images,labels,part_path)

    def saveTrainPart(self,images,labels,dst_path):
        self.save(images,labels,dst_path,"training")

    def saveTestPart(self, images, labels, dst_path):
            self.save(images, labels, dst_path, "testing")


    def readSave(self,src_path,dst_path,dst_size = None):
        FileUtility.createClearFolder(dst_path)
        images,labels = self.readTrainPart(src_path,dst_size)
        self.saveTrainPart(images,labels,dst_path)

        images,labels = self.readTestPart(src_path,dst_size)
        self.saveTestPart(images,labels,dst_path)

