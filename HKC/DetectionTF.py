from .Utility import *
from .FileUtility import *
from .GTDetection import *
from object_detection.utils import dataset_util
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from .GTDetection import *
from .DetectionModelsTF import *
import tensorflow as tf
import time
import cv2
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

class DetectionTF:
    def __init__( self,object_detection_path, deploy_path):

        self._object_detection_path = object_detection_path
        self._deploy_path = deploy_path

        self._createPaths()

    def getDatasetPath(self):
        return self._join('Dataset')

    def getLablemapFilename(self):
        return self._join('Dataset/labelemap.txt')

        # def getCheckpointPath(self):
        #     return self._join('Checkpoint')

    def getTrainPath(self):
        return self._join("Train")

    def getModelsPath(self):
        return os.path.join(self._object_detection_path,'models')



    def _createPaths(self):
        if not os.path.exists(self._deploy_path):
            os.mkdir(self._deploy_path)

        if not os.path.exists(self.getDatasetPath()):
            os.mkdir(self.getDatasetPath())

         if not os.path.exists(self.getTrainPath()):
            os.mkdir(self.getTrainPath())

         if not os.path.exists(self.getModelsPath()):
            os.mkdir(self.getModelsPath())



    def resize(self,src_path,size,jpeg_quality=30, interpolation=None):
        self._size = size

        FileUtility.createClearFolder(self.getImagesPath())
        FileUtility.copyFullSubFolders(src_path, self.getImagesPath())

        GTDetection.resizeBatch(src_path,self.getImagesPath(),size,jpeg_quality,interpolation)

    def _join(self, path):
        return os.path.join(self._deploy_path,path)

    @staticmethod
    def getIndex(label ,labels):
        index =  Utility.getIndex(label, labels)
        if index >= 0 :
            index += 1
        return index

    @staticmethod
    def createTFExample(images_path, csv_data, tf_rec_filename, labels, branch):
        writer = tf.io.TFRecordWriter(tf_rec_filename)
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        class_ = []
        width = 0
        height = 0
        cur_full_filename = ''
        encoded_jpg = None
        image_format = b'jpg'
        filename = ''

        def addSample():
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(cur_full_filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(cur_full_filename.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes)}))
            writer.write(tf_example.SerializeToString())

        cur_filename = ""
        new_file = False
        for i in tqdm(range(len(csv_data)), ncols=100):
            filename = csv_data['filename'][i]
            xmin = csv_data['xmin'][i]
            xmax = csv_data['xmax'][i]
            ymin = csv_data['ymin'][i]
            ymax = csv_data['ymax'][i]
            class_ = csv_data['class'][i]
            width = csv_data['width'][i]
            height = csv_data['height'][i]

            save_flag = False
            if not cur_filename:
                cur_filename = filename
                new_file = True
            elif filename != cur_filename:
                cur_filename = filename
                save_flag = True
                new_file = True
            else:
                new_file = False

            if save_flag:
                addSample()

            if new_file:
                cur_full_filename = os.path.join(os.path.join(images_path, branch), cur_filename)
                with tf.io.gfile.GFile(cur_full_filename, 'rb') as fid:
                    encoded_jpg = fid.read()

                # encoded_jpg_io = io.BytesIO(encoded_jpg)
                # image = Image.open(encoded_jpg_io)
                # width, height = image.size

                # filename = group.filename.encode('utf8')
                xmins = []
                xmaxs = []
                ymins = []
                ymaxs = []
                classes_text = []
                classes = []

            xmins.append(xmin / width)
            xmaxs.append(xmax / width)
            ymins.append(ymin / height)
            ymaxs.append(ymax / height)
            classes_text.append(class_.encode('utf8'))
            classes.append(DetectionTF.getIndex(class_, labels))

        if save_flag:
            addSample()
        writer.close()


    @staticmethod
    def extractCSVLabels(csv_filename):
        all_labels = pd.read_csv(csv_filename, sep=',', usecols=['class'])

        return list(set(all_labels['class']))


    @staticmethod
    def csv2TFRec(images_path, csv_filename, tf_rec_filename, branch, labels):
        # labels = GTUtilityDET.extractCSVLabels(csv_filename)
        writer = tf.io.TFRecordWriter(tf_rec_filename)
        csv_data = pd.read_csv(csv_filename)
        DetectionTF.createTFExample(images_path, csv_data, tf_rec_filename, labels, branch)
        writer.close()


    @staticmethod
    def csv2TFRecBranchs(images_path, csv_path, dst_path, labels):
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        branchs = FileUtility.getFolderFiles(csv_path, 'csv', False, False)
        for branch in branchs:
            DetectionTF.csv2TFRec(images_path, os.path.join(csv_path, branch + '.csv'),
                                   os.path.join(dst_path, branch + '.record'), branch, labels)


    @staticmethod
    def convertImages2TFRec(src_media, dst_path, lablemap_filename, labels,size, train_per=0.8,
                            clear_dst=True,jpeg_quality=30, interpolation=None):
        temp_path = tempfile.mkdtemp()

        GTDetection.copySplitGT2(src_media, temp_path, train_per, True, clear_dst=clear_dst)
        GTDetection.resizeBatch(temp_path,temp_path,size,jpeg_quality,interpolation)
        GTDetection.GT2CsvBranchs(temp_path, temp_path)

        lbls = DetectionTF.getLabelMap(lablemap_filename, labels)
        DetectionTF.csv2TFRecBranchs(temp_path, temp_path, dst_path, lbls)

        FileUtility.createClearFolder(temp_path)


    @staticmethod
    def createLabelMap(dst_filename, labels):
        with open(dst_filename, "w") as file:
            for i, label in enumerate(labels):
                str = 'item {{\n\tid: {0}\n\tname: {1}\n}}\n'.format(i + 1, label)
                file.write(str)


    @staticmethod
    def readLabelMap(filename):
        result = []
        with open(filename, 'r') as file:
            lines = file.readlines()

        for line in lines:
            flag, res = Utility.readField(line, 'name: ')
            if flag:
                print(res)
                result.append(res)

        return result


    @staticmethod
    def loadTensorboardInColab(trained_path):
        subprocess.call(['load_ext', 'tensorboard'])
        subprocess.call(['tensorboard', '--logdir', trained_path])


    @staticmethod
    def initalizeColabForObjDetection(trained_path):
        DetectionTF.mountGDriveInColb()
        DetectionTF.installObjectDetectionInColab()
        DetectionTF.loadTensorboardInColab(trained_path)


    @staticmethod
    def getLabelMap(filename, input_labels=None):
        result = []
        if os.path.exists(filename):
            result = DetectionTF.readLabelMap(filename)
        else:
            DetectionTF.createLabelMap(filename, input_labels)
            result = input_labels

        return result


    @staticmethod
    def prepareImages(src_path, dst_path=None, dst_size=None, jpeg_quality=30, gray=False, flip_horz=False,
                      flip_postfix="_fh", interpolation=None):
        if dst_path == '':
            dst_path = src_path
        else:
            FileUtility.createClearFolder(dst_path)

        dst_flag = False
        if dst_size:
            DetectionTF.resizeBatch(src_path, dst_path, dst_size, jpeg_quality, interpolation)
            dst_flag = True

        if flip_horz:
            if dst_flag:
                DetectionTF.flipHorzBatch(dst_path, dst_path, post_fix=flip_postfix, jpeg_quality=jpeg_quality)
            else:
                DetectionTF.flipHorzBatch(src_path, dst_path, post_fix=flip_postfix, jpeg_quality=jpeg_quality)
                dst_flag = True
        if gray:
            if dst_flag:
                DetectionTF.toGray(dst_path, dst_path)
            else:
                DetectionTF.toGray(src_path, dst_path)




    def getTrainTFRec(self):
        return os.path.join( self.getDatasetPath(),'train.record')

    def getTestTFRec(self):
        return os.path.join( self.getDatasetPath(),'test.record')

    def getModelConfigFile(self):
        pass

    def createTrainSession(self):
        session_path = os.path.join( self.getTrainPath(), self._current_model_name  + '_' + Utility.getNowStr())
        os.mkdir(session_path)
        return session_path



    def setConfigFile(self):
        self._session_path = self.createTrainSession()
        self._session_config_filename = os.path.join(self._session_path,'pipline.config')
        FileUtility.copyFile(self.getConfigFilename(),self._session_config_filename)

        # labelmap, size = (320, 320), num_classes = 1, batch_size = 20, num_steps = 200000
        input_filename = r'J:\flash\inference_graph\New folder\pipeline.config'
        output_filename = r'd:\pipeline.config'

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

        with tf.gfile.GFile(input_filename, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)

        pipeline_config.model.ssd.num_classes = num_classes
        pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.height = size[0]
        pipeline_config.model.ssd.image_resizer.fixed_shape_resizer.width = size[1]

        pipeline_config.train_config.batch_size = batch_size
        pipeline_config.train_config.fine_tune_checkpoint_type: "detection"
        pipeline_config.train_config.num_steps = num_steps

        pipeline_config.train_input_reader.label_map_path = labelmap
        pipeline_config.train_input_reader.tf_record_input_reader.input_path = train_tf

        pipeline_config.eval_input_reader.label_map_path = labelmap
        pipeline_config.eval_input_reader.tf_record_input_reader.input_path = val_tf

        config_text = text_format.MessageToString(pipeline_config)
        with tf.gfile.Open(output_filename, "wb") as f:
            f.write(config_text)


    def _getTrainMethodFilename(self):
        ver = tf.__version__.split('.')[0]
        if ver == '1':
            train_filename = 'model_main.py'
        elif ver == '2':
            train_filename = 'model_main_tf2.py'

        return os.path.join(self._object_detection_path, train_filename)

    def getConfigFilename(self):
        self._join("Models")
        return self._join()

    def selectModel(self,model_name):
        self._current_model_name = model_name



    def getCurrentCheckpoint(self):
         return os.path.join( self.getCheckpointPath(),Utility.getNowStr())

    def getCurrentModelsPath(self):
        return os.path.join(self.getTrainPath(), self._current_model)

    def getConfigFilename(self):
        return os.path.join( self.getCurrentModelsPath(),self._config_filename)


    def train(self,  batch_size=20, num_steps=200000):
        self._batch_size =  batch_size
        self._num_steps = num_steps

        subprocess.call(['python', self._getTrainMethodFilename(), '--pipeline_config_path', self.getConfigFilename(), '--model_dir',
                         self.getCurrentCheckpoint()])

    def test(self,src_path,dst_path,model_path ='__last__',per = 1.0, color=(0,255,0),thickness = 1):
        src_filenames = FileUtility.getFolderImageFiles(src_path)
        dst_filenames = FileUtility.getDstFilenames2(src_filenames,src_path,dst_path)


        tf.keras.backend.clear_session()
        detect_fn = tf.saved_model.load('/root/datalab/my_model/saved_model/')

        model_category_index = label_map_util.create_category_index_from_labelmap(MODEL_LABELS,
                                                                                  use_display_name=True)
        for i in tqdm(range(int(len(src_filenames) * per)), ncols=100):
            src_filename = src_filenames[i]
            dst_filename = dst_filenames[i]

            src_image = cv2.imread(src_filename)
            detections = detect_fn(src_image)


            label_id_offset = 1

            viz_utils.visualize_boxes_and_labels_on_image_array(
                src_image,
                detections['detection_boxes'][0].numpy(),
                detections['detection_classes'][0].numpy().astype(np.int32),
                detections['detection_scores'][0].numpy(),
                model_category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.35,
                agnostic_mode=False)

            cv2.imwrite(dst_filename,src_image)



    def createTFRecord(self,src_path,labels,size, train_per = 0.8,jpeg_quality=30, interpolation=None):
        self._size = size
        self._num_classes = len(Labels)
        DetectionTF.createLabelMap(self.getLablemapFilename(),labels)
        DetectionTF.convertImages2TFRec(src_path,self.getDatasetPath(),self.getLablemapFilename(),labels,size,
                                        train_per,True,jpeg_quality,interpolation)

    @staticmethod
    def addModel(host_name,pretrained_models_path, url):
        if not os.path.exists(pretrained_models_path) :
            os.mkdir(pretrained_models_path)

        if not os.path.exists(os.path.join(pretrained_models_path,host_name)) :
            host_full_name = os.path.join(host_name,pretrained_models_path)
            print('Downloading model ...')
            urllib.request.urlretrieve(url, host_full_name)
            print('Extracting model ...')
            FileUtility.extractFile(host_full_name, pretrained_models_path)









































