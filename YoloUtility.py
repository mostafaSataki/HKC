import os
from .FileUtility import *
from utility import  *
from .GTDetection import *
from .LableMeJson2YOLO import *
from .Voc2YOLO import *
from typing import List
from .YoloInference import *

class YoloUtility:
    def __init__(self,action_type = ActionType.segmentation ,model_size =ModelSize.nano):
        self.action_type = action_type
        self.model_size = model_size


    def convert(self,src_dir,yolo_dir):
        ext = FileUtility.compare_extension_counts(src_dir,'json','xml')
        if ext == None:
            return

        FileUtility.create_folder_if_not_exists(yolo_dir)
        FileUtility.remove_unpair_images(src_dir)
        src_image_filenames = FileUtility.getFolderImageFiles(src_dir)
        dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames,yolo_dir,src_dir)
        gt_filenames = FileUtility.changeFilesExt(src_image_filenames, ext)

        yolo_filenames = FileUtility.getDstFilenames2(gt_filenames,yolo_dir,src_dir)
        yolo_filenames = FileUtility.changeFilesExt(yolo_filenames,'txt')


        yolo_label_filename = os.path.join(yolo_dir, "classes.txt")
        labels = save_yolo_label(gt_filenames,yolo_label_filename)

        if ext == 'json':
            lableme_to_yolo = LableMeJson2YOLO(labels, self.action_type)
            lableme_to_yolo.convert_files(gt_filenames, yolo_filenames)
        elif ext == 'xml':
            voc_to_yolo = Voc2YOLO(labels, self.action_type)
            voc_to_yolo.convert_files(gt_filenames, yolo_filenames)

        FileUtility.copyFilesByName(src_image_filenames,dst_image_filenames)


        
    def generate_yolo_session(self, src_dir, yolo_dir, val_per, test_per):
        ext = FileUtility.compare_extension_counts(src_dir, 'json', 'xml')
        if ext == None:
            return

        train_per = 1 - (val_per + test_per)
        FileUtility.create_folder_if_not_exists(yolo_dir)

        FileUtility.remove_unpair_images(src_dir)


        json_image_filenames = FileUtility.getFolderImageFiles(src_dir)
        train_yolo_image_files, val_yolo_image_files, test_yolo_image_files, train_gt_image_files, val_gt_image_files, test_gt_image_files = \
            split_data_filename(json_image_filenames, yolo_dir,  train_per, val_per, test_per)

        train_yolo_files = FileUtility.changeFilesExt(train_yolo_image_files,'txt')
        val_yolo_files = FileUtility.changeFilesExt(val_yolo_image_files,'txt')
        test_yolo_files = FileUtility.changeFilesExt(test_yolo_image_files,'txt')
        train_json_files = FileUtility.changeFilesExt(train_gt_image_files, ext)
        val_json_files =  FileUtility.changeFilesExt(val_gt_image_files, ext)
        test_json_files =   FileUtility.changeFilesExt(test_gt_image_files, ext)


        all_json_files = train_json_files + val_json_files + test_json_files

        yolo_label_filename = os.path.join(yolo_dir, "classes.txt")

        labels = save_yolo_label(all_json_files,yolo_label_filename)

        if ext == 'json':
            lableme_to_yolo = LableMeJson2YOLO(labels, self.action_type)
            lableme_to_yolo.convert_files(train_json_files, train_yolo_files)
            lableme_to_yolo.convert_files(val_json_files, val_yolo_files)
            lableme_to_yolo.convert_files(test_json_files, test_yolo_files)
        elif ext == 'xml':
            voc_to_yolo = Voc2YOLO(labels, self.action_type)
            voc_to_yolo.convert_files(train_json_files, train_yolo_files)
            voc_to_yolo.convert_files(val_json_files, val_yolo_files)
            voc_to_yolo.convert_files(test_json_files, test_yolo_files)


        #copy image files
        FileUtility.copyFilesByName(train_gt_image_files, train_yolo_image_files)
        FileUtility.copyFilesByName(val_gt_image_files, val_yolo_image_files)
        FileUtility.copyFilesByName(test_gt_image_files, test_yolo_image_files)


        self.write_dataset_ymal_proc(yolo_dir,labels)

        self.create_train_batchfile(yolo_dir)

        self.copy_pretrain_model(yolo_dir)

        self.save_export_file(yolo_dir)
        self.create_segment_export_batchfile(yolo_dir)


    def generate_yolo_session_from_VOC(self, voc_dir, yolo_dir, val_per, test_per):
        train_per = 1 - (val_per + test_per)
        FileUtility.create_folder_if_not_exists(yolo_dir)

        remove_unpair_images(voc_dir)


        json_image_filenames = FileUtility.getFolderImageFiles(voc_dir)
        train_yolo_image_files, val_yolo_image_files, test_yolo_image_files, train_voc_image_files, val_voc_image_files, test_voc_image_files = \
            split_data_filename(json_image_filenames, yolo_dir,  train_per, val_per, test_per)

        train_yolo_files = FileUtility.changeFilesExt(train_yolo_image_files,'txt')
        val_yolo_files = FileUtility.changeFilesExt(val_yolo_image_files,'txt')
        test_yolo_files = FileUtility.changeFilesExt(test_yolo_image_files,'txt')
        train_voc_files = FileUtility.changeFilesExt(train_voc_image_files, 'xml')
        val_voc_files =  FileUtility.changeFilesExt(val_voc_image_files, 'xml')
        test_voc_files =   FileUtility.changeFilesExt(test_voc_image_files, 'xml')


        all_voc_files = train_voc_files + val_voc_files + test_voc_files

        yolo_label_filename = os.path.join(yolo_dir, "classes.txt")

        labels = save_yolo_label(all_voc_files, yolo_label_filename)

        lableme_to_yolo = LableMeJsonYOLO(labels, self.action_type)


        lableme_to_yolo.convert_files(train_voc_files, train_yolo_files)
        lableme_to_yolo.convert_files(val_voc_files, val_yolo_files)
        lableme_to_yolo.convert_files(train_voc_files, train_yolo_files)

        #copy image files
        FileUtility.copyFilesByName(train_voc_image_files, train_yolo_image_files)
        FileUtility.copyFilesByName(val_voc_image_files, val_yolo_image_files)
        FileUtility.copyFilesByName(test_voc_image_files, test_yolo_image_files)


        self.write_dataset_ymal_proc(yolo_dir,labels)

        self.create_train_batchfile(yolo_dir)

        self.copy_pretrain_model(yolo_dir)

        self.save_export_file(yolo_dir)
        self.create_segment_export_batchfile(yolo_dir)


    def write_dataset_ymal_proc(self,yolo_dir, labels):
        yaml_filename = os.path.join(yolo_dir, "dataset.yaml")
        branchs = ['train', 'val', 'test']
        branchs_path = []
        for branch in branchs:
            branchs_path.append(os.path.join(yolo_dir, branch))

        write_dataset_ymal(yaml_filename, branchs, branchs_path, labels)

    def _get_segment_model_name(self):
        if self.model_size == ModelSize.nano:
            return 'yolov8n-seg'
        elif self.model_size == ModelSize.small:
            return 'yolov8s-seg'
        elif self.model_size == ModelSize.medium:
            return 'yolov8m-seg'
        elif self.model_size == ModelSize.large:
            return 'yolov8l-seg'
        elif self.model_size == ModelSize.xtra_large:
            return 'yolov8x-seg'

    def _get_detection_model_name(self):
        if self.model_size == ModelSize.nano:
            return 'yolov8n'
        elif self.model_size == ModelSize.small:
            return 'yolov8s'
        elif self.model_size == ModelSize.medium:
            return 'yolov8m'
        elif self.model_size == ModelSize.large:
            return 'yolov8l'
        elif self.model_size == ModelSize.xtra_large:
            return 'yolov8x'

    def _get_pose_model_name(self):
        if self.model_size == ModelSize.nano:
            return 'YOLOv8n-pose'
        elif self.model_size == ModelSize.small:
            return 'yolov8s-pose'
        elif self.model_size == ModelSize.medium:
            return 'yolov8m-pose'
        elif self.model_size == ModelSize.large:
            return 'yolov8l-pose'
        elif self.model_size == ModelSize.xtra_large:
            return 'yolov8x-pose'

    def _get_classification_model_name(self):
        if self.model_size == ModelSize.nano:
            return 'YOLOv8n-cls'
        elif self.model_size == ModelSize.small:
            return 'yolov8s-cls'
        elif self.model_size == ModelSize.medium:
            return 'yolov8m-cls'
        elif self.model_size == ModelSize.large:
            return 'yolov8l-cls'
        elif self.model_size == ModelSize.xtra_large:
            return 'yolov8x-cls'

    def _get_yolo_model_name(self):
        if self.action_type ==   ActionType.segmentation:
            return self._get_segment_model_name()
        elif self.action_type == ActionType.detection:
            return self._get_detection_model_name()
        elif self.action_type == ActionType.pose_estimation:
            return self._get_pose_model_name()
        elif self.action_type == ActionType.classification:
            return self._get_classification_model_name()


    def copy_pretrain_model(self, yolo_dir):
        FileUtility.create_folder_if_not_exists(os.path.join(yolo_dir, "models"))

        model_filename = self._get_yolo_model_name()
        fname = r'models\{}.pt'.format(model_filename)

        filename = os.path.basename(fname)
        src_filename = os.path.join(os.path.join(os.getcwd(),"Run"), fname)
        dst_filename = os.path.join(yolo_dir, filename)
        FileUtility.copyFile(src_filename, dst_filename)


    def create_train_batchfile(self, yolo_dir):

        model_filename = r'models\{}.pt'.format(self._get_yolo_model_name())


        if self.action_type == ActionType.segmentation :
            batch_filename = os.path.join(yolo_dir, "train_segmentation.bat")
        
        elif self.action_type == ActionType.detection:
            batch_filename = os.path.join(yolo_dir, "train_detection.bat")
        
        elif self.action_type == ActionType.pose_estimation:
            batch_filename = os.path.join(yolo_dir, "train_pose_estimation.bat")
        

        yaml_filename = 'dataset.yaml'
        with open(batch_filename, "w") as fw:
            if self.action_type == ActionType.segmentation :
                cmd = 'yolo  task=segment   mode=train  epochs=500   data ="{}"  model="{}"  imgsz=640  batch=8'.format(
                yaml_filename, model_filename)
            elif self.action_type == ActionType.detection :
                cmd = 'yolo  task=detect   mode=train  epochs=500   data ="{}"  model="{}"  imgsz=640  batch=8'.format(
                yaml_filename, model_filename)
            elif self.action_type == ActionType.pose_estimation:
                cmd = 'yolo  task=pose   mode=pose  epochs=500   data ="{}"  model="{}"  imgsz=640  batch=8'.format(
                    yaml_filename, model_filename)



            fw.write(cmd + '\n')



    def save_export_file(self,yolo_dir):
        fname = "export.py"
        with open(os.path.join(yolo_dir, fname), "w") as f:
            f.write("import sys\n")
            f.write("from ultralytics import YOLO\n")
            f.write("model_filename = sys.argv[1]\n")
            f.write("model = YOLO(model_filename)\n")
            f.write("model.export(format=\"onnx\", opset=12)\n")


    def create_segment_export_batchfile(self,yolo_dir):
        fname = "export.bat"
        if self.action_type == ActionType.segmentation :
            model_filename = os.path.join(yolo_dir, r"runs\segment\train\weights\best.pt")
        elif self.action_type == ActionType.detection:
            model_filename = os.path.join(yolo_dir, r"runs\detect\train\weights\best.pt")
        elif self.action_type == ActionType.pose_estimation:
            model_filename = os.path.join(yolo_dir, r"runs\pose\train\weights\best.pt")

        with open(os.path.join(yolo_dir, fname), "w") as fw:
            cmd = "python export.py {}".format(model_filename)
            fw.write(cmd)

    @staticmethod
    def clear_image_data_from_jsonfile(self, json_file):
        LableMeJson2YOLO.clear_image_data_from_jsonfile(json_file)

    @staticmethod
    def clear_image_data_from_dir(self, json_dir):
        LableMeJson2YOLO.clear_image_data_from_dir(json_dir)

    @staticmethod
    def replace_fname_with_imagepath(json_filename, image_filename):
        LableMeJson2YOLO.replace_fname_with_imagepath(json_filename, image_filename)

    @staticmethod
    def replace_fname_with_imagepath_batch(src_dir):
        LableMeJson2YOLO.replace_fname_with_imagepath_batch(src_dir)

    @staticmethod
    def inference_segment_single(model_filename: str, labels_file: str, src_filename: str, dst_filename: str):
        inf = YoloInference(model_filename, labels_file)
        inf.inference(src_filename,dst_filename)
    
    @staticmethod
    def inference_segment_dir(  model_filename: str, labels_file: str, src_dir: str, dst_dir: str,
                                draw = True,save_json = False,is_rect_contour = False, min_area_cofi = None,crop_dir = None, crop_size = None):
        inf = YoloInference(model_filename, labels_file,draw,save_json,is_rect_contour,min_area_cofi,crop_dir, crop_size)
        inf.inference_dir(src_dir,dst_dir)
        
        


            








