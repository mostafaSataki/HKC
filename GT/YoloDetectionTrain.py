from HKC.FileUtility import *
import os
from HKC.GT.YOLO import *
from HKC.CvUtility import *
import  cv2
from HKC.GTDetection import *
class YoloDetectionTrain:
    def __init__(self, project_name, model_name='ssd_mobilenet_v2_fpnlite_320x320',
                       models_root_path=r'd:\Models',
                       db_root_path=r'd:\Database'
                     ):
            self._model_name = None
            self._models_root_path = None
            self._db_root_path = None


            self._project_name = project_name
            self._db_root_path = db_root_path
            self._db_project_path = os.path.join(self._db_root_path,project_name)
            self._db_yolo_path = os.path.join(self._db_project_path, 'yolo')
            self._db_project_yolo_path = os.path.join(self._db_yolo_path, 'DB')
            self._db_break_project_yolo_path = os.path.join(self._db_yolo_path, 'DB_break')
            self._db_project_original_path = os.path.join(self._db_yolo_path, 'original')
            self._models_root_path = models_root_path




    def _create_db_folders(self):
        train_path = os.path.join(self._db_project_yolo_path, 'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        valid_path = os.path.join(self._db_project_yolo_path, 'valid')
        if not os.path.exists(valid_path):
            os.makedirs(valid_path)
        test_path = os.path.join(self._db_project_yolo_path, 'test')
        if not os.path.exists(test_path):
            os.makedirs(test_path)

    def _create_db_break_folders(self):
        train_path = os.path.join(self._db_break_project_yolo_path, 'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        valid_path = os.path.join(self._db_break_project_yolo_path, 'valid')
        if not os.path.exists(valid_path):
            os.makedirs(valid_path)
        test_path = os.path.join(self._db_break_project_yolo_path, 'test')
        if not os.path.exists(test_path):
            os.makedirs(test_path)

    def _read_labels(self,label_filename):
        labels = []
        with open(label_filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                labels.append(line.rstrip())
            file.close()
        return labels

    def _get_labels_list(self,labels):
        result = '[]'
        if len(labels):
            result = '['
            for label in labels:
                result += '\'' + label + '\','
            result = result[:-1] + ']'
        return result


    def _create_data_file(self):
        labels = self._read_labels(os.path.join( self._db_project_original_path,'labels.txt'))
        data_filename = os.path.join(self._db_project_yolo_path, 'data.yaml')
        with open(data_filename, 'w') as file:

            file.write('train: ' + FileUtility.get_slash_path( os.path.join(self._db_project_yolo_path, 'train')) + '\n')
            file.write('valid: ' + FileUtility.get_slash_path(os.path.join(self._db_project_yolo_path, 'valid')) + '\n')
            file.write('test: ' + FileUtility.get_slash_path(os.path.join(self._db_project_yolo_path, 'test')) + '\n')
            file.write('nc: ' +str(len(labels)) + '\n')
            file.write('names: ' + self._get_labels_list(labels) + '\n')

            file.close()


    def splitGT_copy(self, valid_per=0.2, test_per=0.2):
        train_image_filenames, train_gt_filenames, valid_image_filenames, valid_gt_filenames, test_image_filenames, test_gt_filenames = \
            GTDetection.splitGT(self._db_project_original_path, valid_per, test_per)

        train_path = os.path.join(self._db_project_yolo_path, 'train')
        valid_path = os.path.join(self._db_project_yolo_path, 'valid')
        test_path = os.path.join(self._db_project_yolo_path, 'test')

        FileUtility.copyFiles2DstPath(train_image_filenames,self._db_project_original_path, train_path)
        FileUtility.copyFiles2DstPath(train_gt_filenames,self._db_project_original_path, train_path)
        FileUtility.copyFiles2DstPath(valid_image_filenames,self._db_project_original_path, valid_path)
        FileUtility.copyFiles2DstPath(valid_gt_filenames,self._db_project_original_path, valid_path)
        FileUtility.copyFiles2DstPath(test_image_filenames,self._db_project_original_path, test_path)
        FileUtility.copyFiles2DstPath(test_gt_filenames,self._db_project_original_path, test_path)
        return train_image_filenames, train_gt_filenames, valid_image_filenames, valid_gt_filenames, test_image_filenames, test_gt_filenames

    def split_dataset(self,valid_per=0.2,test_per=0.2,recreate =False):
        train_per = 1 - valid_per - test_per

        if recreate:
            FileUtility.deleteFolderContents(self._db_project_yolo_path)

        if not os.path.exists(self._db_project_yolo_path):
            os.makedirs(self._db_project_yolo_path)

        self._create_db_folders()

        original_path = os.path.join(self._db_root_path, self._project_name, 'original')
        self.splitGT_copy( valid_per,test_per)

        self._create_data_file()


    def _break_image_yolo(self, image_filename, gt_filenames, src_path, dst_path, grid_cols, grid_rows, conflict_cols=0.0,
                         conflict_rows=0.0):
        yolo = YOLO()
        images = []
        gts = []
        yolo.load(gt_filenames)

        image = cv2.imread(image_filename, 1)
        height, width, _ = image.shape
        result = CvUtility.break_image(image, grid_cols, grid_rows, conflict_cols, conflict_rows)

        gt_dst_filename = FileUtility.getDstFilename2(gt_filenames, src_path, dst_path)
        image_dst_filename = FileUtility.getDstFilename2(image_filename, src_path, dst_path)

        for i, res in enumerate(result):
            cur_yolo = yolo.clone()
            rct = CvUtility.rect2Yolorect(res[1], (width, height))
            cur_yolo.filter_by_region(rct)
            cur_yolo.data_.size_ = (int(rct[2] * width), int(rct[3] * height), 3)

            cur_image_dst_filename = FileUtility.changeFileNameEx(image_dst_filename, "", "", "_" + str(i))
            cur_gt_dst_filename = FileUtility.changeFileNameEx(gt_dst_filename, "", "", "_" + str(i))

            cv2.imwrite(cur_image_dst_filename, res[0])
            cur_yolo.save(cur_gt_dst_filename)


    def create_db_break(self, grid_cols, grid_rows, conflict_cols=0.0, conflict_rows=0.0):
        src_path = self._db_project_yolo_path
        dst_path = self._db_break_project_yolo_path

        if os.path.exists(dst_path):
            os.makedirs(dst_path)


        FileUtility.copyFullSubFolders(src_path, dst_path)

        all_src_gt_files = FileUtility.getFolderFiles(src_path, FileUtility.getTexExtensions())
        all_src_gt_files, label_src_files = FileUtility.pop_by_filename(all_src_gt_files, 'classes.txt')
        label_dst_files = FileUtility.getDstFilenames2(label_src_files, src_path, dst_path)
        FileUtility.copyFilesByName(label_src_files, label_dst_files)

        all_src_image_files = FileUtility.getImagePairs(all_src_gt_files)

        for i in tqdm(range(len(all_src_gt_files)), ncols=100):
            gt_src_filename = all_src_gt_files[i]
            image_src_filename = all_src_image_files[i]

            self._break_image_yolo(image_src_filename, gt_src_filename, src_path, dst_path, 1, 2, 0.0, 0.1)










