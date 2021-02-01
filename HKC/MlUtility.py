from .FileUtility import  *
from .Utility import  *
from shutil import copyfile

class MlUtility:

  @staticmethod
  def _totalJoin(group_files,dst_path):
      result = []
      for group in group_files :
         g = []
         for file in group:
           filename = os.path.join(dst_path,file)
           g.append(filename)

      result.append(g)

      return  result

  @staticmethod
  def copyFiles(files, groups, src_path, dst_path,dst_labels,src_label = ''):
       for i, group in enumerate(groups):
         dst_label = dst_labels[i]
         for file_index in group :
            filename = files[file_index]

            if src_label == '':
              src_filename = os.path.join(src_path, filename)
              dst_filename = os.path.join(os.path.join(dst_path,dst_label),filename)
            else :
              src_filename = os.path.join(os.path.join(src_path,src_label), filename)
              dst_filename = os.path.join(os.path.join(os.path.join(dst_path,dst_label),src_label),filename)
            copyfile(src_filename,dst_filename)
            FileUtility.copyFileByNewExtension(src_filename,dst_filename,'xml')


  @staticmethod
  def partitionFiles(src_path, dst_path, lables):
      FileUtility.deleteFolderContents(dst_path)
      dst_lables ,dst_percents = Utility.getDictKeysValues(lables)
      src_subfolders = FileUtility.getSubfolders(src_path)

      if len(src_subfolders) == 0 :
        FileUtility.createSubfolders(dst_path, dst_lables)
        src_files = FileUtility.getFolderImageFilesWithoutPath(src_path)
        list1 = list(range(len(src_files)))
        np.random.shuffle(list1)
        groups = Utility.splitList(list1, lables)
        MlUtility.copyFiles(src_files, groups, src_path, dst_path,dst_lables)
      else :
        FileUtility.createSubfolders(dst_path, dst_lables)
        for src_label in src_subfolders:
          for dst_label in dst_lables :
             FileUtility.createSubfolders(os.path.join( os.path.join( dst_path,dst_label)),src_label)


          src_files = FileUtility.getFolderImageFilesWithoutPath(os.path.join(src_path,src_label))
          list1 = list(range(len(src_files)))
          np.random.shuffle(list1)
          groups = Utility.splitList(list1, lables)
          MlUtility.copyFiles(src_files,groups,src_path,dst_path, dst_lables,src_label)





