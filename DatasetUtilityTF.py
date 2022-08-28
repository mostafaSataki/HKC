import os
from .FileUtility import  *

class DatasetUtilityTF:
  @staticmethod
  def loadDatasetFromFolder(src_path, gray = False, numeric_label = False, train_per = 0.8, total_per = 1.0):
     files ,lbls = FileUtility.loadFilenamesLabels(src_path )
     total_count = len(files)
     per_count = int(total_count * total_per)

     train_count = int(per_count * train_per)

     c = list(zip(files, lbls))
     random.shuffle(c)
     files, lbls =     [list(a) for a in zip(*c)]


     files = files[:per_count]
     lbls = lbls[:per_count]

     images = CvUtility.loadImages(files,gray)
     if numeric_label:
       labels,labels_tuple = Utility.getNumericLabels(lbls)
     else :
       labels = lbls
       labels_tuple ={}


     train_images_dataset = tf.data.Dataset.from_tensor_slices(images[:train_count])
     train_labels_dataset = tf.data.Dataset.from_tensor_slices(labels[:train_count])

     test_images_dataset = tf.data.Dataset.from_tensor_slices(images[train_count:])
     test_labels_dataset = tf.data.Dataset.from_tensor_slices(labels[train_count:])

     return tf.data.Dataset.zip((train_images_dataset, train_labels_dataset)),tf.data.Dataset.zip((test_images_dataset, test_labels_dataset)),labels_tuple
