import enum
import os
from  .GTUtility import *
from .GT.VOC import *
from .GT.YOLO import *


class GTFormat(enum.Enum):
    YOLO = 1
    VOC = 2


class CopyMethod(enum.Enum):
    stretch = 1;
    valid = 2;
    expand = 3


class GTDetection:

    @staticmethod
    def getGtFolderFormat(gt_path):
        image_filesname = FileUtility.getFolderImageFiles(gt_path)
        if len(image_filesname) == 0:
            return None

        image_filename = image_filesname[0]
        voc_filename = FileUtility.changeFileExt(image_filename, 'xml')
        if os.path.exists(voc_filename):
            return GTFormat.VOC

        yolo_filename = FileUtility.changeFileExt(image_filename, 'txt')
        if os.path.exists(yolo_filename):
            return GTFormat.YOLO

        return None

    @staticmethod
    def getGtFileFormat(gt_filename):
        result = None
        ext = FileUtility.getFileExt(gt_filename)
        if ext == 'txt':
            result = GTFormat.YOLO
        elif ext == 'voc':
            result = GTFormat.VOC
        return result

    @staticmethod
    def getGtExt(format):
        if format == GTFormat.YOLO:
            return 'txt'
        elif format == GTFormat.VOC:
            return 'xml'

    @staticmethod
    def allGTFormats():
        return [GTFormat.YOLO, GTFormat.VOC]

    @staticmethod
    def getImageGtPair(image_filename, gt_format=None):
        tokens = FileUtility.getFileTokens(image_filename)
        if gt_format == None:
            all_formats = GTDetection.allGTFormats()

            for format in all_formats:
                gt_filename = os.path.join(tokens[0], tokens[1] + '.' + GTDetection.getGtExt(format))
                if os.path.exists(gt_filename):
                    return gt_filename, format

        else:
            return os.path.join(tokens[0], tokens[1] + '.' + GTDetection.getGtExt(gt_format)), gt_format

    @staticmethod
    def _convertYolo2VocPath(src_path, dst_path):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        src_image_filesname = FileUtility.getFolderImageFiles(src_path)

        src_yolo = YOLO()
        src_yolo.labels_.load(src_path)

        dst_voc = VOC()

        for i in tqdm(range(len(src_image_filesname)), ncols=100):
            src_image_filename = src_image_filesname[i]
            src_gt_filename = FileUtility.changeFileExt(src_image_filename, 'txt')

            dst_image_filename = FileUtility.getDstFilename2(src_image_filename, src_path, dst_path)

            FileUtility.copyFile(src_image_filename, dst_image_filename)
            src_yolo.load(src_gt_filename)
            dst_voc.new(dst_image_filename)

            for obj in src_yolo.data_.objects_:
                label = src_yolo.labels_.getLabel(obj.label_id_)
                dst_voc.add(obj.region_.getCvRect(), label)

            dst_voc.save()

    @staticmethod
    def convertYolo2Voc(src_path, dst_path):
        if FileUtility.checkRootFolder(src_path):
            GTDetection._convertYolo2VocPath(src_path, dst_path)
        else:
            sub_folders = FileUtility.getSubfolders(src_path)
            for sub_folder in sub_folders:
                GTDetection.convertYolo2Voc(os.path.join(src_path, sub_folder), os.path.join(dst_path, sub_folder))

    @staticmethod
    def convertVoc2Yolo(src_path, dst_path):
        src_image_filesname = FileUtility.getFolderImageFiles(src_path)

        src_voc = VOC()
        # src_voc.labels_.load(src_path)

        dst_yolo = YOLO()

        for i, src_image_filename in enumerate(src_image_filesname):
            src_gt_filename = FileUtility.changeFileExt(src_image_filename, 'xml')

            dst_image_filename = FileUtility.getDstFilename2(src_image_filename, src_path, dst_path)

            FileUtility.copyFile(src_image_filename, dst_image_filename)
            src_voc.load(src_gt_filename)
            dst_yolo.new(dst_image_filename)

            for obj in src_voc.data_.objects_:
                dst_yolo.addByLable(obj.region_.getCvRect(), obj.name_)

            dst_yolo.save()

        dst_yolo.labels_.save(dst_path)

    @staticmethod
    def convertGt(src_path, dst_path, src_format=None, dst_format=None):
        if src_format == None:
            src_format = getGtFolderFormat(src_path)
        if dst_format == None:
            dst_format = getGtFolderFormat(dst_path)

        if src_format == None or dst_format == None:
            return

        if src_format == GTFormat.YOLO and dst_format == GTFormat.VOC:
            GTDetection.convertYolo2Voc(src_path, dst_path)
        elif src_format == GTFormat.VOC and dst_format == GTFormat.YOLO:
            GTDetection.convertVoc2Yolo(src_path, dst_path)

    @staticmethod
    def getObjectsCount(gt_filename):
        result = 0
        gt_format = GTDetection.getGtFileFormat(gt_filename)
        if gt_format == None:
            return result

        if gt_format == GTFormat.YOLO:
            gt = YOLO()
        elif gt_format == GTFormat.VOC:
            gt = VOC()
        gt.load(gt_filename)
        return gt.getObjectsCount()

    @staticmethod
    def getObjectsRegions(gt_filename):
        result = []
        gt_format = GTDetection.getGtFileFormat(gt_filename)
        if gt_format == None:
            return result

        if gt_format == GTFormat.YOLO:
            gt = YOLO()
        elif gt_format == GTFormat.VOC:
            gt = VOC()
        gt.load(gt_filename)
        return gt.getObjectsRegions()

    @staticmethod
    def removeBlankGT(path):
        gt_format = GTDetection.getGtFolderFormat(path)
        image_filesname = FileUtility.getFolderImageFiles(path)

        empty_count = 0
        for i in tqdm(range(len(image_filesname)), ncols=100):
            image_filename = image_filesname[i]
            gt_filename, _ = GTDetection.getImageGtPair(image_filename, gt_format)

            if not os.path.exists(gt_filename) or GTDetection.getObjectsCount(gt_filename) == 0:
                if os.path.exists(gt_filename):
                    os.remove(gt_filename)
                os.remove(image_filename)
                empty_count += 1

        print("empty count:", empty_count)

    @staticmethod
    def getGtFolderFormat(gt_path):
        image_filesname = FileUtility.getFolderImageFiles(gt_path)
        if len(image_filesname) == 0:
            return None

        image_filename = image_filesname[0]
        voc_filename = FileUtility.changeFileExt(image_filename, 'xml')
        if os.path.exists(voc_filename):
            return GTFormat.VOC

        yolo_filename = FileUtility.changeFileExt(image_filename, 'txt')
        if os.path.exists(yolo_filename):
            return GTFormat.YOLO

        return None

    @staticmethod
    def getGtFileFormat(gt_filename):
        result = None
        ext = FileUtility.getFileExt(gt_filename)
        if ext == 'txt':
            result = GTFormat.YOLO
        elif ext == 'voc':
            result = GTFormat.VOC
        return result

    @staticmethod
    def getGtExt(format):
        if format == GTFormat.YOLO:
            return 'txt'
        elif format == GTFormat.VOC:
            return 'xml'

    @staticmethod
    def allGTFormats():
        return [GTFormat.YOLO, GTFormat.VOC]

    @staticmethod
    def getImageGtPair(image_filename, gt_format=None):
        tokens = FileUtility.getFileTokens(image_filename)
        if gt_format == None:
            all_formats = GTDetection.allGTFormats()

            for format in all_formats:
                gt_filename = os.path.join(tokens[0], tokens[1] + '.' + GTDetection.getGtExt(format))
                if os.path.exists(gt_filename):
                    return gt_filename, format

        else:
            return os.path.join(tokens[0], tokens[1] + '.' + GTDetection.getGtExt(gt_format)), gt_format

    @staticmethod
    def _convertYolo2VocPath(src_path, dst_path):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        src_image_filesname = FileUtility.getFolderImageFiles(src_path)

        src_yolo = YOLO()
        src_yolo.labels_.load(src_path)

        dst_voc = VOC()

        for i in tqdm(range(1, len(src_image_filesname)), ncols=100):
            src_image_filename = src_image_filesname[i]
            src_gt_filename = FileUtility.changeFileExt(src_image_filename, 'txt')

            dst_image_filename = FileUtility.getDstFilename2(src_image_filename, src_path, dst_path)

            FileUtility.copyFile(src_image_filename, dst_image_filename)
            src_yolo.load(src_gt_filename)
            dst_voc.new(dst_image_filename)

            for obj in src_yolo.data_.objects_:
                label = src_yolo.labels_.getLabel(obj.label_id_)
                dst_voc.add(obj.region_.getCvRect(), label)

            dst_voc.save()

    @staticmethod
    def convertYolo2Voc(src_path, dst_path):
        if FileUtility.checkRootFolder(src_path):
            GTDetection._convertYolo2VocPath(src_path, dst_path)
        else:
            sub_folders = FileUtility.getSubfolders(src_path)
            for sub_folder in sub_folders:
                GTDetection.convertYolo2Voc(os.path.join(src_path, sub_folder), os.path.join(dst_path, sub_folder))

    @staticmethod
    def convertVoc2Yolo(src_path, dst_path):
        src_image_filesname = FileUtility.getFolderImageFiles(src_path)

        src_voc = VOC()
        # src_voc.labels_.load(src_path)

        dst_yolo = YOLO()

        for i, src_image_filename in enumerate(src_image_filesname):
            src_gt_filename = FileUtility.changeFileExt(src_image_filename, 'xml')

            dst_image_filename = FileUtility.getDstFilename2(src_image_filename, src_path, dst_path)

            FileUtility.copyFile(src_image_filename, dst_image_filename)
            src_voc.load(src_gt_filename)
            dst_yolo.new(dst_image_filename)

            for obj in src_voc.data_.objects_:
                dst_yolo.addByLable(obj.region_.getCvRect(), obj.name_)

            dst_yolo.save()

        dst_yolo.labels_.save(dst_path)

    @staticmethod
    def convertGt(src_path, dst_path, src_format=None, dst_format=None):
        if src_format == None:
            src_format = getGtFolderFormat(src_path)
        if dst_format == None:
            dst_format = getGtFolderFormat(dst_path)

        if src_format == None or dst_format == None:
            return

        if src_format == GTFormat.YOLO and dst_format == GTFormat.VOC:
            GTDetection.convertYolo2Voc(src_path, dst_path)
        elif src_format == GTFormat.VOC and dst_format == GTFormat.YOLO:
            GTDetection.convertVoc2Yolo(src_path, dst_path)

    @staticmethod
    def getObjectsCount(gt_filename):
        result = 0
        gt_format = GTDetection.getGtFileFormat(gt_filename)
        if gt_format == None:
            return result

        if gt_format == GTFormat.YOLO:
            gt = YOLO()
        elif gt_format == GTFormat.VOC:
            gt = VOC()
        gt.load(gt_filename)
        return gt.getObjectsCount()

    @staticmethod
    def getObjectsRegions(gt_filename):
        result = []
        gt_format = GTDetection.getGtFileFormat(gt_filename)
        if gt_format == None:
            return result

        if gt_format == GTFormat.YOLO:
            gt = YOLO()
        elif gt_format == GTFormat.VOC:
            gt = VOC()
        gt.load(gt_filename)
        return gt.getObjectsRegions()

    @staticmethod
    def removeBlankGT(path):
        gt_format = GTDetection.getGtFolderFormat(path)
        image_filesname = FileUtility.getFolderImageFiles(path)

        empty_count = 0
        for i in tqdm(range(len(image_filesname)), ncols=100):
            image_filename = image_filesname[i]
            gt_filename, _ = GTDetection.getImageGtPair(image_filename, gt_format)

            if not os.path.exists(gt_filename) or GTDetection.getObjectsCount(gt_filename) == 0:
                if os.path.exists(gt_filename):
                    os.remove(gt_filename)
                os.remove(image_filename)
                empty_count += 1

        print("empty count:", empty_count)


    @staticmethod
    def mergeAllGTFiles(zip_path, dst_path, prefix, start_counter=0, remove_blank_gt=True, recompress_qulaity=30,
                        pad_count=7,
                        clear_similar_frames=True, psnr_thresh=19, distance_thresh=100, area_thresh=100):
        if clear_similar_frames:
            remove_blank_gt = True
        zip_files = FileUtility.getFolderFiles(zip_path, 'zip')

        FileUtility.copyFullSubFolders(zip_path, dst_path, True)

        counter = 0
        for i in tqdm(range(len(zip_files)), ncols=100):
            zip_file = zip_files[i]
            temp_path = tempfile.mkdtemp()

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_path)

            gt_format = GTDetection.getGtFolderFormat(temp_path)
            gt_ext = GTDetection.getGtExt(gt_format)

            if remove_blank_gt:
                GTDetection.removeBlankGT(temp_path)

            if recompress_qulaity != None:
                CvUtility.recompressImages(temp_path)

            if clear_similar_frames:
                GTDetection.clearSimilarGTFrames(temp_path, psnr_thresh, distance_thresh, area_thresh)
                GTDetection.removeBlankGT(temp_path)

            src_image_files = FileUtility.getFolderImageFiles(temp_path)
            src_gt_files = FileUtility.changeFilesExt(src_image_files, gt_ext)

            dst_image_files, _ = FileUtility.changeFilesname(src_image_files, prefix, counter, pad_count)
            dst_gt_files, _ = FileUtility.changeFilesname(src_gt_files, prefix, counter, pad_count)

            cur_dst_path = FileUtility.file2Folder(FileUtility.getDstFilename2(zip_file, zip_path, dst_path))

            dst_image_files = FileUtility.getDstFilenames2(dst_image_files, temp_path, cur_dst_path, True)
            dst_gt_files = FileUtility.getDstFilenames2(dst_gt_files, temp_path, cur_dst_path, True)

            FileUtility.copyFilesByName(src_image_files, dst_image_files)
            FileUtility.copyFilesByName(src_gt_files, dst_gt_files)

            counter += len(dst_gt_files)

            FileUtility.deleteFolderContents(temp_path)

            # shutil.rmtree(temp_path)


    @staticmethod
    def getGTData(image_filename):
        gt_filename, _ = GTDetection.getImageGtPair(image_filename)
        image = cv2.imread(image_filename)
        regions = GTDetection.getObjectsRegions(gt_filename)
        return image, regions


    @staticmethod
    def similartGT(image1_filename, image2_filename, psnr_thresh=24, distance_thresh=100, area_thresh=100):
        image1, regions1 = GTDetection.getGTData(image1_filename)
        image2, regions2 = GTDetection.getGTData(image2_filename)
        if len(regions1) == 0 or len(regions2) == 0:
            return
        return CvUtility.similariy(image1, regions1[0], image2, regions2[0])


    @staticmethod
    def clearSimilarGTFrames(src_path, psnr_thresh=19, distance_thresh=100, area_thresh=100):
        image_filesname = FileUtility.getFolderImageFiles(src_path)
        ref_id = 0
        ref_image_filename = image_filesname[ref_id]

        ref_img, ref_regions = GTDetection.getGTData(ref_image_filename)

        for i in tqdm(range(len(image_filesname)), ncols=100):
            cur_image_filename = image_filesname[i]
            cur_img, cur_regions = GTDetection.getGTData(cur_image_filename)
            is_similar = CvUtility.similariy(ref_img, ref_regions[0], cur_img, cur_regions[0])
            if not is_similar:
                ref_image_filename = cur_image_filename
                ref_img = cur_img
                ref_regions = cur_regions
            else:
                os.remove(image_filesname[i])

    @staticmethod
    def loadGT(image_filename):
        gt_filename, gt_format = GTDetection.getImageGtPair(image_filename)
        if gt_format == GTFormat.YOLO:
            gt = YOLO()
        elif gt_format == GTFormat.VOC:
            gt = VOC()

        gt.load(gt_filename)

        return gt, gt_filename

    @staticmethod
    def correctnessYoloLableFile(src_path, labels):
        classess_fname = 'classes.txt'
        if FileUtility.checkRootFolder(src_path):
            classess_filename = os.path.join(src_path, classess_fname)
            if not os.path.exists(classess_filename):
                FileUtility.writeTextList(classess_filename, labels)
        else:
            sub_folders = FileUtility.getSubfolders(src_path)
            for sub_folder in sub_folders:
                GTDetection.correctnessYoloLableFile(os.path.join(src_path, sub_folder), labels)

    @staticmethod
    def resize(image_filename, size, jpeg_quality=30, interpolation=None):
        gt, gt_filename = GTDetection.loadGT(image_filename)
        src_image = cv2.imread(image_filename)
        dst_image, scale = CvUtility.resizeImage(src_image, size, interpolation)
        for obj in gt.data_.objects_:
            obj.region_.setCvRect(RectUtility.resize(obj.region_.getCvRect(), scale))

        cv2.imwrite(image_filename, dst_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        gt.save(gt_filename)

    @staticmethod
    def resizeBatch(src_path, dst_path, size, jpeg_quality=30, interpolation=None):
        FileUtility.copyFullSubFolders(src_path, dst_path)

        src_image_filesname, src_gt_filesname = GTDetection.getGtFiles(src_path)

        dst_image_filesname = FileUtility.getDstFilenames2(src_image_filesname, src_path, dst_path)
        dst_gt_filesname = FileUtility.getDstFilenames2(src_gt_filesname, src_path, dst_path)

        FileUtility.copyFilesByName(src_image_filesname, dst_image_filesname)
        FileUtility.copyFilesByName(src_gt_filesname, dst_gt_filesname)

        for i in tqdm(range(len(dst_image_filesname)), ncols=100):
            dst_image_filename = dst_image_filesname[i]
            GTDetection.resize(dst_image_filename, size, jpeg_quality, interpolation)

    @staticmethod
    def flipHorz(image_filename, jpeg_quality=30):
        gt, gt_filename = GTDetection.loadGT(image_filename)

        image = cv2.imread(image_filename)

        for obj in gt.data_.objects_:
            obj.region_.setCvRect(RectUtility.flipHorzRect(obj.region_.getCvRect(), image.shape))

        image = cv2.flip(image, 1)
        cv2.imwrite(image_filename, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        gt.save(gt_filename)

    @staticmethod
    def flipHorzBatch(src_path, dst_path, post_fix="_fh", jpeg_quality=30):

        if src_path != dst_path:
            FileUtility.copyFullSubFolders(src_path, dst_path)

        src_image_filesname, src_gt_filesname = GTDetection.getGtFiles(src_path)

        dst_image_filesname = FileUtility.getDstFilenames2(src_image_filesname, src_path, dst_path)
        dst_gt_filesname = FileUtility.getDstFilenames2(src_gt_filesname, src_path, dst_path)

        dst_image_filesname = FileUtility.changeFilesnamePostfix(dst_image_filesname, post_fix)
        dst_gt_filesname = FileUtility.changeFilesnamePostfix(dst_gt_filesname, post_fix)

        FileUtility.copyFilesByName(src_image_filesname, dst_image_filesname)
        FileUtility.copyFilesByName(src_gt_filesname, dst_gt_filesname)

        for i in tqdm(range(len(dst_image_filesname)), ncols=100):
            dst_image_filename = dst_image_filesname[i]
            GTDetection.flipHorz(dst_image_filename, jpeg_quality)

    @staticmethod
    def createGt(image, regions, label, format):
        if format == GTFormat.YOLO:
            gt = YOLO()
        elif format == GTFormat.VOC:
            gt = VOC()

    @staticmethod
    def cropGT(src_path, dst_path):
        src_files = FileUtility.getFolderImageFiles(src_path)
        dst_files = FileUtility.getDstFilenames2(src_files, src_path, dst_path)

        FileUtility.copyFullSubFolders(src_path, dst_path)

        gt_format = GTDetection.getGtFolderFormat(src_path)

        for i in tqdm(range(len(src_files)), ncols=100):
            src_image_file = src_files[i]
            dst_image_file = dst_files[i]

            # src_gt_filename = GTUtility.getImageGtPair(src_file,gt_format)
            image, regions = GTDetection.getGTData(src_image_file)

            for j, region in enumerate(regions):
                croped_image = CvUtility.imageROI(image, region)
                cur_dst_image_filename = dst_image_file
                if j > 0:
                    cur_dst_image_filename = FileUtility.changeFilesnamePostfix(dst_image_file)
                cv2.imwrite(cur_dst_image_filename, croped_image)

    @staticmethod
    def getGtFiles(src_path):
        gt_format = GTDetection.getGtFolderFormat(src_path)
        gt_ext = GTDetection.getGtExt(gt_format)

        image_filenames = FileUtility.getFolderImageFiles(src_path)
        gt_filenames = FileUtility.changeFilesExt(image_filenames, gt_ext)

        return image_filenames, gt_filenames

    @staticmethod
    def getGTFiles(image_filenames, gt_filenames, indexs):
        image_filenames = Utility.getListByIndexs(image_filenames, indexs)
        gt_filenames = Utility.getListByIndexs(gt_filenames, indexs)
        return image_filenames, gt_filenames

    @staticmethod
    def splitGT(src_path, train_per=0.8):
        image_filenames, gt_filenames = GTDetection.getGtFiles(src_path)
        train_indexs, test_indexs = GTUtility.getGTIndexs(len(image_filenames), train_per, IndexType.random)

        train_image_filenames, train_gt_filenames = GTDetection.getGTFiles(image_filenames, gt_filenames, train_indexs)
        test_image_filenames = Utility.getListByIndexs(image_filenames, test_indexs)
        test_gt_filenames = Utility.getListByIndexs(gt_filenames, test_indexs)

        return train_image_filenames, train_gt_filenames, test_image_filenames, test_gt_filenames

    @staticmethod
    def GT2Csv(src_path, csv_filename):
        image_filenames, gt_filenames = GTDetection.getGtFiles(src_path)
        gt_list = []

        for i in tqdm(range(len(image_filenames)), ncols=100):
            image_filename = image_filenames[i]
            gt_filename = gt_filenames[i]
            gt, _ = GTDetection.loadGT(image_filename)

            for obj in gt.data_.objects_:
                r = obj.region_.get2Points()

                value = (gt.data_.filename_,
                         gt.data_.size_[0], gt.data_.size_[1],
                         obj.name_, r[0], r[1], r[2], r[3])
                gt_list.append(value)

        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(gt_list, columns=column_name)

        xml_df.to_csv(csv_filename, index=None)

    @staticmethod
    def GT2CsvBranchs(src_path, dst_path, clear_dst=False):
        if clear_dst:
            FileUtility.createClearFolder(dst_path)

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        sub_folders = FileUtility.getSubfolders(src_path)
        for sub_folder in sub_folders:
            cur_folder = os.path.join(src_path, sub_folder)
            GTDetection.GT2Csv(cur_folder, os.path.join(dst_path, sub_folder + ".csv"))

    @staticmethod
    def copySplitGT2(src_media, dst_path, train_per=0.8, copy_to_root=False, select_type=IndexType.random,
                     clear_dst=False, branchs=['train', 'test']):
        media = src_media
        media_type, ext = FileUtility.getMediaInfo(src_media)
        if media_type == MediaType.file and ext == 'zip':
            media = FileUtility.extractFile(src_media)

        FileUtility.createDstBrach(dst_path, branchs, clear_dst)

        branch_state = False
        if not FileUtility.checkRootFolder(media):
            branch_state = True
            if copy_to_root == False:
                for branch in branchs:
                    FileUtility.copyFullSubFolders(media, os.path.join(dst_path, branch))

        dst_path_train = os.path.join(dst_path, branchs[0])
        dst_path_test = os.path.join(dst_path, branchs[1])

        if branch_state and (select_type == IndexType.begin_branch or select_type == IndexType.end_branch):
            sub_folders = FileUtility.getSubfolders(media)
            for sub_folder in sub_folders:
                src_cur_branch = os.path.join(media, sub_folder)
                dst_cur_branch_train = os.path.join(dst_path_train, sub_folder)
                dst_cur_branch_test = os.path.join(dst_path_test, sub_folder)

                image_filenames, gt_filenames = GTDetection.getGtFiles(src_cur_branch)
                train_indexs, test_indexs = GTUtility.getGTIndexs(len(image_filenames), train_per, select_type)

                src_train_image_filenames, src_train_gt_filenames = GTDetection.getGTFiles(image_filenames,
                                                                                           gt_filenames, train_indexs)
                src_test_image_filenames, src_test_gt_filenames = GTDetection.getGTFiles(image_filenames, gt_filenames,
                                                                                         test_indexs)

                if copy_to_root:
                    dst_train_image_filenames = FileUtility.getDstFilenames2(src_train_image_filenames, src_cur_branch,
                                                                             dst_path_train,
                                                                             copy_to_root)
                    dst_train_gt_filenames = FileUtility.getDstFilenames2(src_train_gt_filenames, src_cur_branch,
                                                                          dst_path_train,
                                                                          copy_to_root)

                    dst_test_image_filenames = FileUtility.getDstFilenames2(src_test_image_filenames, src_cur_branch,
                                                                            dst_path_test,
                                                                            copy_to_root)
                    dst_test_gt_filenames = FileUtility.getDstFilenames2(src_test_gt_filenames, src_cur_branch,
                                                                         dst_path_test,
                                                                         copy_to_root)


                else:
                    dst_train_image_filenames = FileUtility.getDstFilenames2(src_train_image_filenames, src_cur_branch,
                                                                             dst_cur_branch_train, copy_to_root)
                    dst_train_gt_filenames = FileUtility.getDstFilenames2(src_train_gt_filenames, src_cur_branch,
                                                                          dst_cur_branch_train,
                                                                          copy_to_root)

                    dst_test_image_filenames = FileUtility.getDstFilenames2(src_test_image_filenames, src_cur_branch,
                                                                            dst_cur_branch_test, copy_to_root)
                    dst_test_gt_filenames = FileUtility.getDstFilenames2(src_test_gt_filenames, src_cur_branch,
                                                                         dst_cur_branch_test,
                                                                         copy_to_root)

                FileUtility.copyFilesByName(src_train_image_filenames, dst_train_image_filenames)
                FileUtility.copyFilesByName(src_train_gt_filenames, dst_train_gt_filenames)
                FileUtility.copyFilesByName(src_test_image_filenames, dst_test_image_filenames)
                FileUtility.copyFilesByName(src_test_gt_filenames, dst_test_gt_filenames)


        else:
            image_filenames, gt_filenames = GTDetection.getGtFiles(media)
            train_indexs, test_indexs = GTUtility.getGTIndexs(len(image_filenames), train_per, select_type)

            src_train_image_filenames, src_train_gt_filenames = GTDetection.getGTFiles(image_filenames, gt_filenames,
                                                                                       train_indexs)
            src_test_image_filenames, src_test_gt_filenames = GTDetection.getGTFiles(image_filenames, gt_filenames,
                                                                                     test_indexs)

            dst_path_train = os.path.join(dst_path, branchs[0])
            dst_path_test = os.path.join(dst_path, branchs[1])

            dst_train_image_filenames = FileUtility.getDstFilenames2(src_train_image_filenames, media, dst_path_train,
                                                                     copy_to_root)
            dst_train_gt_filenames = FileUtility.getDstFilenames2(src_train_gt_filenames, media, dst_path_train,
                                                                  copy_to_root)

            dst_test_image_filenames = FileUtility.getDstFilenames2(src_test_image_filenames, media, dst_path_test,
                                                                    copy_to_root)
            dst_test_gt_filenames = FileUtility.getDstFilenames2(src_test_gt_filenames, media, dst_path_test,
                                                                 copy_to_root)

            FileUtility.copyFilesByName(src_train_image_filenames, dst_train_image_filenames)
            FileUtility.copyFilesByName(src_train_gt_filenames, dst_train_gt_filenames)
            FileUtility.copyFilesByName(src_test_image_filenames, dst_test_image_filenames)
            FileUtility.copyFilesByName(src_test_gt_filenames, dst_test_gt_filenames)

        if media_type == MediaType.file and ext == "zip":
            shutil.rmtree(media)

    @staticmethod
    def copyGTAsPer(src_path, dst_path, per=1.0, copy_to_root=False, select_type=IndexType.random, clear_dst=False):
        if clear_dst:
            FileUtility.createClearFolder(dst_path)

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        branch_state = False
        if not FileUtility.checkRootFolder(src_path):
            branch_state = True
            if not copy_to_root:
                FileUtility.copyFullSubFolders(src_path, dst_path)

        if branch_state and (select_type == IndexType.begin_branch or select_type == IndexType.end_branch):
            sub_folders = FileUtility.getSubfolders(src_path)
            for sub_folder in sub_folders:
                src_cur_branch = os.path.join(src_path, sub_folder)
                dst_cur_branch = os.path.join(dst_path, sub_folder)

                image_filenames, gt_filenames = GTDetection.getGtFiles(src_cur_branch)
                indexs, _ = GTUtility.getGTIndexs(len(image_filenames), per, select_type)
                src_image_filenames, src_gt_filenames = GTDetection.getGTFiles(image_filenames, gt_filenames, indexs)

                if copy_to_root:
                    dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames, src_cur_branch, dst_path,
                                                                       copy_to_root)
                    dst_gt_filenames = FileUtility.getDstFilenames2(src_gt_filenames, src_cur_branch, dst_path,
                                                                    copy_to_root)
                else:
                    dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames, src_cur_branch,
                                                                       dst_cur_branch,
                                                                       copy_to_root)
                    dst_gt_filenames = FileUtility.getDstFilenames2(src_gt_filenames, src_cur_branch, dst_cur_branch,
                                                                    copy_to_root)

                FileUtility.copyFilesByName(src_image_filenames, dst_image_filenames)
                FileUtility.copyFilesByName(src_gt_filenames, dst_gt_filenames)


        else:
            image_filenames, gt_filenames = GTDetection.getGtFiles(src_path)
            indexs, _ = GTUtility.getGTIndexs(len(image_filenames), per, select_type)

            src_image_filenames, src_gt_filenames = GTDetection.getGTFiles(image_filenames, gt_filenames, indexs)

            dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames, src_path, dst_path, copy_to_root)
            dst_gt_filenames = FileUtility.getDstFilenames2(src_gt_filenames, src_path, dst_path, copy_to_root)

            FileUtility.copyFilesByName(src_image_filenames, dst_image_filenames)
            FileUtility.copyFilesByName(src_gt_filenames, dst_gt_filenames)




