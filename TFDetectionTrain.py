


from .FileUtility import *
from .GTDetection import *
import tensorflow as tf
from .Utility import *
import  numpy as np
from .DetectionTF import *
import  os
from object_detection.utils import config_util
from object_detection import model_lib_v2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
class ModelData:
    def __init__(self,name,url,size,config_filename):
        self.name = name
        self.url = url
        self.size = size
        self.config_filename = config_filename


class TFDetectionTrain:

    def __init__(self, project_name, model_name='ssd_mobilenet_v2_fpnlite_320x320', models_root_path=r'E:\Models',
                 db_root_path = r'E:\Database',
                 object_detection_path = r'E:\Library\Tensorflow2\models\research\object_detection'
                  ):
        self._model_name = None
        self._models_root_path = None
        self._db_root_path = None
        self._tf_config_path = None

        self._object_detection_path = object_detection_path
        self.tf_config_path = os.path.join(self._object_detection_path,r'configs\tf2')
        self._get_pretrained_models()

        self._project_name = project_name

        self.model_name = model_name
        self.models_root_path = models_root_path
        self.db_root_path = db_root_path


        self._labelmap_fname = 'label_map.pbtxt'
        self._train_tfname = 'train.tfrecord'
        self._eval_tfname = 'eval.tfrecord'
        self.jpeg_quality = 40
        self.train_per = 0.8
        self.interpolation = cv2.INTER_CUBIC
        self.batch_size = 128

    def _get_pretrained_models(self):
        self._models = {}
        self._models['ssd_mobilenet_v2_fpnlite_320x320'] = ModelData('ssd_mobilenet_v2_fpnlite_320x320',
                                                                     'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
                                                                     (320, 320),
                                                                     'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config')

        self._models['ssd_mobilenet_v2_fpnlite_640x640'] = ModelData('ssd_mobilenet_v2_fpnlite_640x640',
                                                                     'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz',
                                                                     (640, 640),
                                                                     'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config')

    def create_labelmap(self):
        labelmap_filename =  self._get_labelmap_filename()
        if FileUtility.exists_nonzero(labelmap_filename):
            return
        GTDetection.create_labelmap_from_voc_folder(self._db_voc_path, labelmap_filename)

    def _get_cur_db_voc_images_path(self):
        return os.path.join( self._get_cur_db_voc_path(), 'images')

    def resize_dataset(self, recreate=False):

        voc_original_images_path = self._get_voc_original_images_path()
        if os.path.exists(self._db_voc_path) and os.path.exists(voc_original_images_path):
            cur_db_voc_images_path = self._get_cur_db_voc_images_path()

            do_resize = False
            if recreate:
                do_resize = True
            else:
                equal,_,_ = FileUtility.compareFolderFiles(voc_original_images_path, cur_db_voc_images_path)
                if not equal:
                    do_resize = True

            if do_resize:

                if os.path.exists(cur_db_voc_images_path):
                    shutil.rmtree(cur_db_voc_images_path)

                os.makedirs(cur_db_voc_images_path)

                GTDetection.resizeBatch(voc_original_images_path, cur_db_voc_images_path, self._get_model_size(),
                                    self.jpeg_quality, cv2.INTER_LINEAR, recreate)


        self._create_tfrecord(recreate)

    def _get_checkpoint_last_item(self,resume_checkpoint_path):
        files = FileUtility.getFolderFiles(resume_checkpoint_path,['index'])
        max_value = -1
        for file in files:
            value = int(FileUtility.getFilenameWithoutExt(file).split('-')[-1])
            if value > max_value:
                max_value = value
        return os.path.join(resume_checkpoint_path, 'ckpt-' + str(max_value))



    def _get_resume_checkpoint_path(self,resume_checkpoint=0):
        if resume_checkpoint == 0:
            resume_checkpoint_path = self._get_model_pretrained_checkpoint_path()
        elif resume_checkpoint == -1:
            _, resume_checkpoint_path = self._get_last_checkpoint()
        elif resume_checkpoint > 0:
            checkpoint = self._find_checkpoint_by_index(resume_checkpoint)
            if checkpoint is None:
                _, resume_checkpoint = self._get_last_checkpoint()
            else:
                resume_checkpoint_path = checkpoint
        return resume_checkpoint_path

    def _get_resume_checkpoint_item_path(self, resume_checkpoint=0):
        return self._get_checkpoint_last_item(self._get_resume_checkpoint_path())

    def add_quotes(self,string):
        return '"' + string + '"'

    def train(self,resume_checkpoint=0,bacth_size = 4):
        model_checkpoints_path =  self._get_cur_model_checkpoints_path()
        model_tensorboard_path = self._get_cur_model_tensorboard_path()

        if not os.path.exists(model_checkpoints_path):
            os.makedirs(model_checkpoints_path)
        if not os.path.exists(model_tensorboard_path):
            os.makedirs(model_tensorboard_path)

        self._resume_checkpoint_path = self._get_resume_checkpoint_item_path(resume_checkpoint)

        self._edit_config_file()

        next_checkpoint_path = os.path.join(self._get_cur_model_checkpoints_path(), self.get_next_checkpoint_path())
        if not os.path.exists(next_checkpoint_path):
            os.makedirs(next_checkpoint_path)

        model_main_tf2_path = self.add_quotes( os.path.join(self._object_detection_path, 'model_main_tf2.py'))
        cmd_str = 'python {} --pipeline_config_path={} --model_dir={} --alsologtostderr --num_train_steps=300000 --num_eval_steps=1000'.format(
           model_main_tf2_path,self.add_quotes(self._get_dst_config_filename()),self.add_quotes(next_checkpoint_path))
        print(cmd_str)
        # os.system(cmd_str)

    def export(self,checkpoint_index):
        exporter_main_v2_path = self.add_quotes(os.path.join(self._object_detection_path, 'exporter_main_v2.py'))
        checkpoint_path = self.add_quotes(self._get_resume_checkpoint_path(checkpoint_index))
        model_path = self.add_quotes(os.path.join(checkpoint_path,'model'))

        cmd_str = 'python {} - -input_type image_tensor --trained_checkpoint_dir {} --pipeline_config_path={} --output_directory={} '.format(
            exporter_main_v2_path,checkpoint_path, self.add_quotes(self._get_dst_config_filename()),model_path)
        print(cmd_str)
        os.system(cmd_str)

    def load_image_into_numpy_array(self,path):
        return np.array(Image.open(path))

    #use object detection tensorflow 2 pre trained model

    def test_by_model(self, images_path, checkpoint_index):
        model_path = os.path.join( self._get_resume_checkpoint_path(checkpoint_index),r'model\saved_model')
        detect_fn = tf.saved_model.load(model_path)
        files = FileUtility.getFolderImageFiles(images_path)
        category_index = label_map_util.create_category_index_from_labelmap(self._get_labelmap_filename(),  use_display_name=True)

        for file in files:
            image_np = self.load_image_into_numpy_array(file)

            # Things to try:
            # Flip horizontally
            # image_np = np.fliplr(image_np).copy()

            # Convert image to grayscale
            # image_np = np.tile(
            #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image_np)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

            # input_tensor = np.expand_dims(image_np, 0)
            detections = detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.10,
                agnostic_mode=False)

            plt.figure()
            plt.imshow(image_np_with_detections)
            print('Done')
            plt.show()

    def test_by_checkpoint(selff, images_path, checkpoint_index):
        pass





    def _get_tfrecord_filenames(self):
        tfrecord_path = self._get_cur_db_tfrecord_path()
        if tfrecord_path is None:
            return None


        return [os.path.join(tfrecord_path, self._train_tfname),
               os.path.join(tfrecord_path, self._eval_tfname)]


    def _create_model_path(self):
        self._model_path = os.path.join(self._project_model_path, self._model_name)
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        if value in self._models:
            self._model_name = value
            self._set_models_pretrained_path()
            self._set_project_model_path()
            self._download_extract_model()

        else:
            raise Exception('Model not found')


    @property
    def models_root_path(self):
        return self._models_root_path

    @models_root_path.setter
    def models_root_path(self, value):
        if self._models_root_path != value:
            self._models_root_path = value
            self._set_project_model_path()

            self._set_models_pretrained_path()
            self._download_extract_model()



    @property
    def db_root_path(self):
        return self._db_path

    @db_root_path.setter
    def db_root_path(self, value):
        if self._db_root_path != value:
            self._db_root_path = value
            self._set_project_db_path()
            self._set_project_voc_path()


    @property
    def tf_config_path(self):
        return self._tf_config_path
    @tf_config_path.setter
    def tf_config_path(self, value):
        if self._tf_config_path != value and os.path.exists(value):
            self._tf_config_path = value


    def _get_model(self):
        return self._models[self._model_name]

    def _get_model_size(self):
        return self._models[self._model_name].size


    def _set_models_pretrained_path(self):
        if self._models_root_path is None :
            return
        self._models_pretrained_path = os.path.join(self._models_root_path, 'pretrained')

        if not os.path.exists(self._models_pretrained_path):
            os.makedirs(self._models_pretrained_path)

    def _set_project_model_path(self):
        if self._models_root_path is None or self._project_name is None:
            return

        self._project_model_path = os.path.join(self._models_root_path, self._project_name)
        if not os.path.exists(self._project_model_path):
            os.makedirs(self._project_model_path)


    def _set_src_model_path(self):
        if self._models_pretrained_path is None or self._model_name is None:
            return None
        self._src_model_path = os.path.join(self._models_pretrained_path, self._model_name)
        if not os.path.exists(self._src_model_path):
            return None




    def _set_project_db_path(self):
        if self._db_root_path is None or self._project_name is None:
            return
        self._project_db_path = os.path.join(self._db_root_path, self._project_name)
        if not os.path.exists(self._project_db_path):
            os.makedirs(self._project_db_path)


    def _get_labelmap_filename(self):
        if self._db_voc_path is None:
            return None
        return os.path.join(self._db_voc_path, self._labelmap_fname)

    def _get_voc_original_images_path(self):
        if self._db_voc_path is None:
            return None
        return os.path.join(self._db_voc_path, 'original')

    def _set_project_voc_path(self):
        if self._project_db_path is None :
            return

        self._db_voc_path = os.path.join(self._project_db_path, 'voc')
        if not os.path.exists(self._db_voc_path):
            os.makedirs(self._db_voc_path)

        self._model_voc_path = os.path.join(self._project_model_path, 'voc')
        if not os.path.exists(self._model_voc_path):
            os.makedirs(self._model_voc_path)


        voc_original_images_path = self._get_voc_original_images_path()
        if not os.path.exists(voc_original_images_path):
            os.makedirs(voc_original_images_path)







    def _get_model_voc_name(self):
        model = self._get_model()
        if model:
            return 'voc' + str(model.size[0])
        return None


    def _get_cur_db_voc_path(self):
        if self._db_voc_path is None or self._model_name is None:
            return

        model = self._get_model()
        if model :
            cur_db_voc_path = os.path.join(self._db_voc_path, self._get_model_voc_name())
            if not os.path.exists(cur_db_voc_path):
                os.makedirs(cur_db_voc_path)
            return cur_db_voc_path
        return None

    def _get_cur_model_voc_path(self):
        if self._model_voc_path is None or self._model_name is None:
            return

        model = self._get_model()
        if model :
            cur_model_voc_path = os.path.join(self._model_voc_path, self._get_model_voc_name())
            if not os.path.exists(cur_model_voc_path):
                os.makedirs(cur_model_voc_path)
            return cur_model_voc_path
        return None

    def _get_cur_model_checkpoints_path(self):
        if self._get_cur_model_voc_path() is None:
            return None
        return os.path.join(self._get_cur_model_voc_path(), 'checkpoints')

    def _get_cur_model_tensorboard_path(self):
        if self._get_cur_model_voc_path() is None:
            return None
        return os.path.join(self._get_cur_model_voc_path(), 'tensorboard')







    def _get_src_config_filename(self):
        model = self._get_model()
        if model is None:
            return None
        return os.path.join(self._tf_config_path,model.config_filename)

    def _get_dst_config_filename(self):
        return os.path.join(self._get_cur_model_voc_path(),'pipeline.config')


    def _get_model_images_path(self):
        return os.path.join(self._project_voc_model_path,'images')

    def _get_cur_db_tfrecord_path(self):
        return os.path.join(self._get_cur_db_voc_path(),'tfrecord')



    def _create_tfrecord(self,recreate =False):
        labelemap_filename = self._get_labelmap_filename()
        if labelemap_filename is None:
            return False

        if not FileUtility.exists_nonzero(labelemap_filename):
            return False

        cur_db_voc_path  = self._get_cur_db_voc_path()
        if cur_db_voc_path is None or not os.path.exists(cur_db_voc_path):
                return False

        images_path = self._get_cur_db_voc_images_path()
        tfrecord_path = self._get_cur_db_tfrecord_path()

        if not os.path.exists(images_path) :
            return False

        if not os.path.exists(tfrecord_path) :
            os.makedirs(tfrecord_path)


        tf_records_filenames = self._get_tfrecord_filenames()
        if recreate == False and FileUtility.exists_nonzeros(tf_records_filenames):
            return True


        print('Creating tfrecord files...')
        labels = DetectionTF.get_labels_from_labelmap(labelemap_filename)

        temp_path = tempfile.mkdtemp()

        GTDetection.copySplitGT2(images_path, temp_path,self.train_per , True, clear_dst=True)
        GTDetection.GT2CsvBranchs(temp_path, temp_path)
        DetectionTF.csv2TFRecBranchs(temp_path, temp_path, tfrecord_path, labels)
        FileUtility.createClearFolder(temp_path)

        return True

    def _get_model_pretrained_filename(self):
        model = self._get_model()
        if model is None or self._models_pretrained_path is None:
            return None

        model_filename = FileUtility.getFilename(model.url)
        return os.path.join(self._models_pretrained_path,model_filename)

    def _get_model_pretrained_path(self):
        model = self._get_model()
        if model is None or self._models_pretrained_path is None:
            return None
        return  os.path.join(self._models_pretrained_path, FileUtility.getFilenameWithoutExt(FileUtility.getFilename(model.url)))



    def _get_model_pretrained_checkpoint_path(self):
        model = self._get_model()
        if model is None or self._models_pretrained_path is None:
            return None

        return os.path.join(self._get_model_pretrained_path(),'checkpoint')



    def _download_model(self):
        model = self._get_model()
        if model is None:
            return False

        model_pretrained_filename =  self._get_model_pretrained_filename()

        if model_pretrained_filename:
            if os.path.exists(model_pretrained_filename):
                print('Downloading model...')

            else:
                print('Downloading model...')
                urllib.request.urlretrieve(model.url, model_pretrained_filename)

            return True
        else: return False

    def _extract_model(self):
        model_pretrained_filename = self._get_model_pretrained_filename()
        if os.path.exists(model_pretrained_filename):
            print('extracting model ...')
            tar = tarfile.open(model_pretrained_filename)
            tar.extractall(path=self._models_pretrained_path)
            tar.close()

    def _download_extract_model(self):
        if self._models_root_path is None or self._project_name is None:
            return

        if self._download_model():
            self._extract_model()
            return True
        return False

    def _edit_mb2_config_file(self):
        labelmap_filename = self._get_labelmap_filename()
        labels_count = len(DetectionTF.get_labels_from_labelmap(labelmap_filename))
        src_config_filename = self._get_src_config_filename()
        dst_config_filename = self._get_dst_config_filename()

        FileUtility.copyFile(src_config_filename, dst_config_filename)

        pipeline_config_dict = config_util.get_configs_from_pipeline_file(dst_config_filename)

        pipeline_config_dict['model'].ssd.num_classes = labels_count

        pipeline_config_dict['train_config'].fine_tune_checkpoint = self._resume_checkpoint_path
        pipeline_config_dict['train_config'].fine_tune_checkpoint_type = "detection"
        pipeline_config_dict['train_config'].batch_size = self.batch_size


        tfrec_filenames = self._get_tfrecord_filenames()

        pipeline_config_dict['train_input_config'].tf_record_input_reader.input_path[0] = tfrec_filenames[0]
        pipeline_config_dict['train_input_config'].label_map_path = labelmap_filename

        pipeline_config_dict['eval_input_config'].tf_record_input_reader.input_path[0] = tfrec_filenames[1]
        pipeline_config_dict['eval_input_config'].label_map_path = labelmap_filename

        pipeline_config = config_util.create_pipeline_proto_from_configs(pipeline_config_dict)
        config_util.save_pipeline_config(pipeline_config, self._get_cur_model_voc_path())

    def _is_mb2_model(self):
        return self._model_name =='ssd_mobilenet_v2_fpnlite_320x320' or self._model_name == 'ssd_mobilenet_v2_fpnlite_640x640'

    def _edit_config_file(self):
        if self._is_mb2_model():
            self._edit_mb2_config_file()


    def extract_number(self,string):
        for i in range(len(string)):
            if string[i] == '[':
                j = i + 1
                while string[j] != ']':
                    return int(string[j])

        return 0

    def _get_last_checkpoint(self):
        model_checkpoint_path = self._get_cur_model_checkpoints_path()
        sub_folders =  FileUtility.getSubfolders(model_checkpoint_path)

        #process sub_folders by extract_numbers and find max value and max item
        max_value = 0
        max_item = ''
        for item in sub_folders:
            value = self.extract_number(item)
            if value > max_value:
                max_value = value
                max_item = item
        return max_value,os.path.join(model_checkpoint_path,max_item)

    def _find_checkpoint_by_index(self,index):
        model_checkpoint_path = self._get_cur_model_checkpoints_path()
        sub_folders =  FileUtility.getSubfolders(model_checkpoint_path)

        for item in sub_folders:
            value = self.extract_number(item)
            if value == index:
                return os.path.join(model_checkpoint_path,item)
        return None



    def get_next_checkpoint_path(self):
        max_value,_ = self._get_last_checkpoint()
        return '[{}]{}'.format(str(max_value+1),datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
