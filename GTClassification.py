import os
from  .FileUtility import *
from .TrainUtility import *
from .CvUtility import *

class GTClassification:

    @staticmethod
    def getFilesStrLabels_(src_path):
        if FileUtility.checkRootFolder(src_path):
            print("There are not any classes in the folder.")
            return None, None

        filenames = []
        labels = []

        sub_folders = FileUtility.getSubfolders(src_path)
        for sub_folder in sub_folders:
            cur_sub_folder = os.path.join(src_path, sub_folder)
            cur_filenames = FileUtility.getFolderImageFiles(cur_sub_folder)
            filenames.extend(cur_filenames)
            labels.extend([sub_folder] * len(cur_filenames))

        return filenames, labels

    @staticmethod
    def getFilesIntLabels_(images_path, org_classes):
        filesname, str_labels = GTClassification.getFilesStrLabels_(images_path)
        if filesname == None:
            return None, None

        str_classes = Utility.getUniqueValues(str_labels)
        if not Utility.matchLists(org_classes, str_classes):
            print('Destination and source classes are different.')
            return None, None

        int_labels = []
        for label in str_labels:
            int_labels.append(org_classes.index(label))

        return filesname, int_labels

    def getFilesIntLabels(self):
        return GTClassification.getFilesIntLabels_(self._join('images'), self._org_classes)

    @staticmethod
    def getSplitList_(images_path, org_classes, train_per=0.8):
        files, labels = GTClassification.getFilesIntLabels_(images_path, org_classes)

        indexs = MatUtility.getRandomIndexs(len(files))

        files = Utility.sortListByIndexs(files, indexs)
        labels = Utility.sortListByIndexs(labels, indexs)

        train_count = int(train_per * len(files))

        return files[:train_count], labels[:train_count], files[train_count:], labels[train_count:]

    def getSplitList(self, train_per=0.8):
        return GTClassification.getSplitList_(self._join('images'), self._org_classes, train_per)

    @staticmethod
    def writeGtFile_(csv_filename, filenames, labels, delimiter=','):
        header = ['filename', 'label']
        with open(csv_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=delimiter)
            writer.writerow(header)
            for i, csv_filename in enumerate(filenames):
                row = []

                row.append(csv_filename)
                row.append(labels[i])

                writer.writerow(row)

    @staticmethod
    def readGtFile_(csv_filename, delimiter=','):
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
    def loadCSVFile_(csv_filename, org_classes, size=None, norm=False, gray=False, float_type=True, use_per=1.0,
                     delimiter=','):

        filenames, labels = GTClassification.readGtFile_(csv_filename)

        indexs = MatUtility.getRandomIndexs(len(filenames), use_per)

        filenames = Utility.getListByIndexs(filenames, indexs)
        labels = Utility.getListByIndexs(labels, indexs)

        images = GTClassification.readImages(filenames, size, norm, gray, float_type)
        images = np.array(images)
        labels = Utility.strList2Indexs(labels)
        labels = TrainUtility.convertOneHot(labels, len(org_classes))

        return images, labels, filenames

    def loadCSVFile(self, csv_filename, size=None, norm=False, gray=False, float_type=True, use_per=1.0, delimiter=','):
        return GTClassification.loadCSVFile_(csv_filename, size, norm, gray, float_type, use_per, delimiter)

    @staticmethod
    def loadFolderImages(src_path, size=None, norm=False, gray=False, float_type=True, use_per=1.0):
        filenames = FileUtility.getFolderImageFiles(src_path)

        indexs = MatUtility.getRandomIndexs(len(filenames), use_per)
        filenames = Utility.getListByIndexs(filenames, indexs)

        images = GTClassification.readImages(filenames, size, norm, gray, float_type)
        images = np.array(images)
        return images, filenames



    @staticmethod
    def createGtFiles_(images_path, train_gt_filename, test_gt_filename, org_classes, train_per=0.8):
        train_files, train_labels, test_files, test_labels = GTClassification.getSplitList_(images_path, org_classes,
                                                                                        train_per)

        GTClassification.writeGtFile_(train_gt_filename, train_files, train_labels)
        GTClassification.writeGtFile_(test_gt_filename, test_files, test_labels)

    def createGtFiles(self, train_per=0.8):
        FileUtility.createClearFolder(self.gtPath())
        GTClassification.createGtFiles_(self._join('images'), self.trainGtFilename(), self.testGtFilename(),
                                    self._org_classes, train_per)

    @staticmethod
    def loadDataset_(train_gt_filename, test_gt_filename, org_classes, norm=False, gray=False, float_type=True,
                     use_per=1.0, size=None, delimiter=','):

        train_X, train_y, _ = GTClassification.loadCSVFile_(train_gt_filename, org_classes, size, norm, gray, float_type,
                                                        use_per, delimiter)
        test_X, test_y, _ = GTClassification.loadCSVFile_(test_gt_filename, org_classes, size, norm, gray, float_type,
                                                      use_per, delimiter)

        return train_X, train_y, test_X, test_y

