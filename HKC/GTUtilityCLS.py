from .FileUtility import *
from .CvUtility import *
from .GTUtility import *
from .TrainUtility import *
import os
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from .ClsModels import *
from time import time
from  keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tqdm import tqdm
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

class GTUtilityClS:
    def __init__(self,deploy_path,org_classes ,delimiter = ','):
        self._deply_path = deploy_path
        self._delimiter = delimiter
        self._org_classes = org_classes

    def _join(self, path):
        return os.path.join(self._deply_path,path)

    def trainGtFilename(self):
        return os.path.join(self.gtPath(),'train.txt')

    def testGtFilename(self):
        return os.path.join(self.gtPath(),'test.txt')


    def gtPath(self):
        return self._join('GT')

    @staticmethod
    def getFilesStrLabels_(src_path):
        if FileUtility.checkRootFolder(src_path):
            print("There are not any classes in the folder.")
            return None,None

        filenames = []
        labels = []

        sub_folders = FileUtility.getSubfolders(src_path)
        for sub_folder in sub_folders:
            cur_sub_folder = os.path.join(src_path,sub_folder)
            cur_filenames = FileUtility.getFolderImageFiles(cur_sub_folder)
            filenames.extend(cur_filenames )
            labels.extend([sub_folder] * len(cur_filenames))


        return filenames, labels

    @staticmethod
    def getFilesIntLabels_(images_path,org_classes):
        filesname,str_labels = GTUtilityClS.getFilesStrLabels_(images_path)
        if filesname == None :
            return None,None

        str_classes = Utility.getUniqueValues(str_labels)
        if not Utility.matchLists(org_classes, str_classes):
            print('Destination and source classes are different.')
            return None, None

        int_labels = []
        for label in str_labels:
            int_labels.append(org_classes.index(label))

        return  filesname,int_labels


    def getFilesIntLabels(self):
        return GTUtilityClS.getFilesIntLabels_(self._join('images'),self._org_classes)

    @staticmethod
    def getSplitList_(images_path,org_classes, train_per=0.8):
        files, labels = GTUtilityClS.getFilesIntLabels_(images_path,org_classes)

        indexs = MatUtility.getRandomIndexs(len(files))

        files = Utility.sortListByIndexs(files, indexs)
        labels = Utility.sortListByIndexs(labels, indexs)

        train_count = int(train_per * len(files))

        return files[:train_count], labels[:train_count], files[train_count:], labels[train_count:]

    def getSplitList(self, train_per = 0.8):
        return GTUtilityClS.getSplitList_(self._join('images'),self._org_classes, train_per)

    @staticmethod
    def writeGtFile_(csv_filename, filenames, labels,delimiter = ','):
        header = ['filename', 'label']
        with open(csv_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter= delimiter)
            writer.writerow(header)
            for i, csv_filename in enumerate(filenames):
                row = []

                row.append(csv_filename)
                row.append(labels[i])

                writer.writerow(row)

    @staticmethod
    def readGtFile_(csv_filename,delimiter = ','):
        filenames = []
        labels = []
        with open(csv_filename, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=delimiter)
            for i, row in enumerate(reader):
                if i == 0:
                    if len(row) != 2:
                        return filenames, labels
                else:
                    filenames.append(row[0])
                    labels.append(row[1])

        return filenames, labels

    @staticmethod
    def loadCSVFile_(csv_filename,org_classes,  size = None,norm = False,gray = False,float_type = True, use_per=1.0,delimiter = ','):

        filenames,labels = GTUtilityClS.readGtFile_(csv_filename)

        indexs = MatUtility.getRandomIndexs(len(filenames), use_per)

        filenames = Utility.getListByIndexs(filenames, indexs)
        labels = Utility.getListByIndexs(labels, indexs)


        images = CvUtility.readImages(filenames,size,norm,gray,float_type)
        images = np.array(images)
        labels = Utility.strList2Indexs(labels)
        labels = TrainUtility.convertOneHot(labels,len(org_classes))

        return images,labels,filenames


    def loadCSVFile(self, csv_filename, size=None, norm=False, gray=False, float_type=True, use_per=1.0,delimiter = ','):
        return GTUtilityClS.loadCSVFile_(csv_filename, size, norm, gray, float_type, use_per,delimiter)

    @staticmethod
    def loadFolderImages(src_path, size = None,norm = False,gray = False,float_type = True, use_per=1.0):
        filenames = FileUtility.getFolderImageFiles(src_path)

        indexs = MatUtility.getRandomIndexs(len(filenames), use_per)
        filenames = Utility.getListByIndexs(filenames, indexs)

        images = CvUtility.readImages(filenames, size, norm, gray, float_type)
        images = np.array(images)
        return  images ,filenames

    def resizeBatch(self, src_path,size):
        self._size = size
        dst_path =self._join('images')
        FileUtility.createClearFolder(dst_path)
        FileUtility.copyFullSubFolders(src_path,dst_path)
        CvUtility.resizeBatch(src_path,self._join('images'),size)

    @staticmethod
    def createGtFiles_(images_path,train_gt_filename,test_gt_filename,org_classes, train_per=0.8):
        train_files, train_labels, test_files, test_labels = GTUtilityClS.getSplitList_(images_path,org_classes, train_per)

        GTUtilityClS.writeGtFile_(train_gt_filename,train_files,train_labels)
        GTUtilityClS.writeGtFile_(test_gt_filename, test_files, test_labels)

    def createGtFiles(self, train_per = 0.8):
        FileUtility.createClearFolder(self.gtPath())
        GTUtilityClS.createGtFiles_(self._join('images'),self.trainGtFilename(),self.testGtFilename(),self._org_classes, train_per)


    @staticmethod
    def loadDataset_(train_gt_filename,test_gt_filename,org_classes,norm = False,gray = False,float_type = True, use_per=1.0,size = None,delimiter = ','):

        train_X,train_y,_ = GTUtilityClS.loadCSVFile_(train_gt_filename,org_classes,size,norm,gray,float_type,use_per,delimiter)
        test_X,test_y,_ = GTUtilityClS.loadCSVFile_(test_gt_filename,org_classes,size,norm,gray,float_type,use_per,delimiter)

        return  train_X,train_y ,test_X,test_y



    def loadDataset(self, norm = False,gray = False,float_type = True, use_per=1.0,size = None):
        self._norm = norm
        self._gray = gray
        self._float_type = float_type
        self._use_per = use_per
        if size :
            self._size = size

        self._train_X, self._train_y, self._test_X, self._test_y = GTUtilityClS.loadDataset_(
            self.trainGtFilename(),self.testGtFilename(),self._org_classes, norm,gray,float_type,use_per,self._size)



    def selectModel(self, type):
        self._model = ClsModels.getModel(type,self._size,len(self._org_classes))

    
    def train(self, optimizer,loss = 'categorical_crossentropy',metrics = 'accuracy',tl_checkpoint = None, epochs=1000):

        self._model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        if not os.path.exists(self._join('checkpoint')):
           os.mkdir(self._join('checkpoint'))

        self._last_checkpoint = Utility.getNowStr()
        tensorboard = TensorBoard(log_dir=self._join('tensorboard/{}').format(time()))
        last_checkpoint_path = os.path.join(self._join('checkpoint'),self._last_checkpoint)
        os.mkdir(last_checkpoint_path)
        ck_path =os.path.join(last_checkpoint_path, 'model-{epoch:03d}.ckpt')
        checkpoint = ModelCheckpoint(ck_path, verbose=1,
                                     monitor='val_loss', save_best_only=True,  mode='auto')

        train_datagen = ImageDataGenerator(rescale=1. / 1, zoom_range=0.2, rotation_range=15,
                                           width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                                           horizontal_flip=True, fill_mode='nearest')
        train_generator = train_datagen.flow(self._train_X, self._train_y, batch_size=30)

        # self._model.fit(self._train_X,self._train_y, validation_data=(self._test_X,self._test_y), epochs=epochs, verbose=1,
        #       callbacks=[tensorboard, checkpoint], shuffle=True)

        train_steps_per_epoch = self._train_X.shape[0] // 30
        val_steps_per_epoch = self._test_X.shape[0] // 20

        val_datagen = ImageDataGenerator(rescale=1. / 1)


        val_generator = val_datagen.flow(self._test_X, self._test_y, batch_size=20)


        history = self._model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=100,
                                  validation_data=val_generator, validation_steps=val_steps_per_epoch, verbose=1,
                                  callbacks=[checkpoint,tensorboard])

    def getLastCheckpoint(self):
        return  FileUtility.getLastFilename(FileUtility.getLastFilename(self._join('checkpoint')))

    def inference(self,filename):
        pass


    def saveImages(self,filenames, labels,scores):
        test_path = self._join('test')
        FileUtility.createClearFolder(test_path)
        FileUtility.createSubfolders(test_path, self._org_classes)

        for i in tqdm(range(len(filenames)), ncols=100):
            filename = filenames[i]
        # for i,filename in enumerate(filenames) :
            label = labels[i]
            score = str(int(scores[i] * 100))

            tokens = FileUtility.getFileTokens(filename)
            fname = tokens[1] + tokens[2]
            # src_filename = os.path.join(self._temp_path,fname)

            label_str = self._org_classes[label]
            dst_filename = os.path.join(os.path.join(test_path,label_str),score+ "__" + fname)

            FileUtility.copyFile(filename,dst_filename)



    def inferenceBatch(self,src_path,size,org_classes, norm = False,gray = False,float_type = True,checkpoint = '__last__',use_per = 1.0):
        self._org_classes = org_classes
        if checkpoint =='__last__':
            checkpoint = self.getLastCheckpoint()
            if checkpoint == None:
                print('checkpoint is invalid.')
                return
        else :
            if not os.path.exists(checkpoint):
                print('checkpoint is invalid.')
                return

        model = load_model(checkpoint)
        self._temp_path = tempfile.mkdtemp()
        CvUtility.resizeBatch(src_path,self._temp_path, size)



        X_test ,filenames = GTUtilityClS.loadFolderImages(self._temp_path,size,norm,gray,float_type,use_per)
        res = model.predict(X_test)
        labels = np.argmax(res, axis=1)
        scores = np.max(res, axis=1)

        self.saveImages(filenames,labels,scores)

        FileUtility.createClearFolder(self._temp_path)

    def savePbFormat(self, checkpoint='__last__'):
        if checkpoint == '__last__':
            checkpoint = self.getLastCheckpoint()
            if checkpoint == None:
                print('checkpoint is invalid.')
                return
        else:
            if not os.path.exists(checkpoint):
                print('checkpoint is invalid.')
                return

        model = load_model(checkpoint)
        tf.saved_model.save(model, r'E:\deep_video\Compressed\Hande-State\deploy\p')


    def saveCheckPoint(self, deploy_path, graphdef_filename, checkpoint='__last__'):
        self._deply_path = deploy_path
        if checkpoint == '__last__':
            checkpoint = self.getLastCheckpoint()
            if checkpoint == None:
                print('checkpoint is invalid.')
                return
        else:
            if not os.path.exists(checkpoint):
                print('checkpoint is invalid.')
                return

        network = load_model(checkpoint)
        print("input:", network.input.name)
        print("output:", network.output.name)


        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: network(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(network.inputs[0].shape, network.inputs[0].dtype))

        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()


        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

        # Save frozen graph from frozen ConcreteFunction to hard drive
        frozen_models_path = os.path.join( self._join('frozen_models'), Utility.getNowStr())
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=frozen_models_path,
                          name="frozen_graph.pb",
                          as_text=False)










