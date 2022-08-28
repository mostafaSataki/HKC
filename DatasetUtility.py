import os
from .FileUtility import  *

class DatasetUtility:
    @staticmethod
    def separate_files(files,sep='_'):
        part1 = []
        part2 = []
        for file in files:
            filename_tokens = FileUtility.getFileTokens(file)
            fname_tokens = filename_tokens[1].split(sep)
            part1.append( os.path.join(filename_tokens[0],fname_tokens[0]))
            tokens2 = fname_tokens[1].split('.')
            part2.append(tokens2[0])
        return part1,part2

    @staticmethod
    def extract_imposter_genuine_from_files(src_files,sep='_'):
        parts1, parts2 = DatasetUtility.separate_files(src_files, sep)
        genuine1 = []
        genuine2 = []

        imposter1 = []
        imposter2 = []


        for i in tqdm(range(len(src_files)),ncols=100):
            src_file = src_files[i]
            file1_part1 = parts1[i]
            file1_part2 = parts2[i]
            for j in range(i + 1, len(src_files)):
                file2_part1 = parts1[j]
                file2_part2 = parts2[j]
                if file1_part1 == file2_part1:
                    genuine1.append(i)
                    genuine2.append(j)
                else:
                    imposter1.append(i)
                    imposter2.append(j)

        genuine = list(zip(genuine1, genuine2))
        imposter = list(zip(imposter1, imposter2))

        return genuine, imposter

    @staticmethod
    def extract_imposter_genuine_from_path(src_path, sep='_',recursive=True):
        if recursive :
           sub_folders = FileUtility.getSubfolders(src_path)
           paths = []
           for sub_folder in sub_folders:
               paths.append(os.path.join(src_path,sub_folder))
           return DatasetUtility.extract_imposter_genuine_from_paths(paths,sep)
        else :
            src_files = FileUtility.getFolderFiles(src_path,sep)
            return src_files, DatasetUtility.extract_imposter_genuine_from_files(src_files,sep)


    @staticmethod
    def extract_imposter_genuine_from_paths(src_paths,sep='_'):
        src_files = FileUtility.getFoldersFiles(src_paths,['npy'])
        return src_files, FileUtility.extract_imposter_genuine_from_files(src_files,sep)

