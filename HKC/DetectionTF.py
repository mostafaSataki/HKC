from .Utility import *
from .FileUtility import *
from .GTDetection import *
from object_detection.utils import dataset_util
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

class DetectionTF:
    @staticmethod
    def getIndex(label ,labels):
        index =  Utility.getIndex(label, labels)
        if index >= 0 :
            index += 1
        return index

    @staticmethod
    def createTFExample(images_path ,groups ,tf_rec_filename ,labels, branch):
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
        for i in tqdm(range(len(groups)), ncols=100):
            filename = groups['filename'][i]
            xmin = groups['xmin'][i]
            xmax = groups['xmax'][i]
            ymin = groups['ymin'][i]
            ymax = groups['ymax'][i]
            class_ = groups['class'][i]
            width = groups['width'][i]
            height = groups['height'][i]

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
        grouped = pd.read_csv(csv_filename)
        DetectionTF.createTFExample(images_path, grouped, tf_rec_filename, labels, branch)
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
    def convertImages2TFRec(src_media, dst_path, lablemap_filename, labels, train_per=0.8, clear_dst=True):
        temp_path = tempfile.mkdtemp()

        GTDetection.copySplitGT2(src_media, temp_path, train_per, True, clear_dst=clear_dst)
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
    def installObjectDetectionInColab(od_path):
        models_path = os.path.join(od_path, 'models')
        research_path = os.path.join(models_path, 'research')
        slim_path = os.path.join(research_path, 'slim')

        os.chdir(od_path)
        subprocess.call(['git', 'clone', '--quiet', 'https://github.com/tensorflow/models.git'])
        os.chdir(models_path)
        if Utility.isLinux():
            subprocess.call(['apt-get', 'install', '-qq', 'protobuf-compiler', 'python-tk'])

        subprocess.call(['pip', 'install', '-q', 'Cython', 'contextlib2', 'pillow', 'lxml', 'matplotlib', 'PyDrive'])
        subprocess.call(['pip', 'install', '-q', 'pycocotools'])
        os.chdir(research_path)
        # subprocess.call(['cd', '~/models/research'])
        subprocess.call(['protoc', 'object_detection/protos/*.proto', '--python_out', '.'])

        subprocess.call(['pip', 'install', 'pascal_voc_writer'])
        subprocess.call(['pip', 'install', 'imgaug'])
        subprocess.call(['pip', 'install', 'selenium'])
        subprocess.call(['pip', 'install', 'tf-models-official'])

        subprocess.call(['pip', 'install', 'tf_slim'])

        os.environ['PYTHONPATH'] += ':/root/models/research/:/root/models/research/slim/'

        subprocess.call(['python', os.path.join(research_path, 'object_detection/builders/model_builder_test.py')])

        os.chdir(research_path)
        # subprocess.call(['cd', '/root/models/research'])
        subprocess.call(['python', 'setup.py', 'build'])
        subprocess.call(['sudo', 'python', 'setup.py', 'install'])


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


    @staticmethod
    def downloadModel(URL, model_name, train_path, pretrained_path, model_ext='.tar.gz', clear_file=False):
        pretrained_full_path = os.path.join(pretrained_path, model_name)

        model_filename = model_name + model_ext
        model_host_filename = os.path.join(pretrained_full_path, model_filename)
        model_repo_filename = os.path.join(URL, model_filename)

        if not os.path.exists(model_host_filename):
            urllib.request.urlretrieve(model_repo_filename, model_host_filename)
        FileUtility.extractFile(model_host_filename, pretrained_full_path)


    @staticmethod
    def customizeConfigFile(labelmap, train_tf, val_tf, size=(320, 320), num_classes=1, batch_size=20, num_steps=200000):
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


    @staticmethod
    def trainObjDetection(od_path, models_path, training_path):
        return
        config_file = 'pipeline.config'
        checkpoints_path = os.path.join(models_path, 'checkpoints')

        if not os.path.exists(checkpoints_path):
            os.mkdir(checkpoints_path)

        ver = tf.__version__.split('.')[0]
        if ver == '1':
            model_main = 'model_main.py'
        elif ver == '2':
            model_main = 'model_main_tf2.py'

        model_main_full = os.path.join(od_path, model_main)
        config_filename = os.path.join(model_path, config_file)

        subprocess.call(['python', model_main_full, '--pipeline_config_path', config_filename, '--model_dir',
                         '/content/drive/MyDrive/Hand/checkpoint '])








































