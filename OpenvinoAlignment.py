import  cv2
import numpy as np
from .open_vino.utils import cut_roi, resize_input
from .open_vino.ie_module import Module

class OpenvinoAlignment:
  def __init__(self):
      # self._input_shape = input_shape

      self._REFERENCE_LANDMARKS = [
            (30.2946 / 96, 51.6963 / 112), # left eye
            (65.5318 / 96, 51.5014 / 112), # right eye
            (48.0252 / 96, 71.7366 / 112), # nose tip
            (33.5493 / 96, 92.3655 / 112), # left lip corner
            (62.7299 / 96, 92.2041 / 112)] # right lip corner

      self._input_shape = (1, 3, 128, 128)

  @staticmethod
  def normalize(array, axis):
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std

  @staticmethod
  def get_transform(src, dst):
      assert np.array_equal(src.shape, dst.shape) and len(src.shape) == 2, \
        '2d input arrays are expected, got {}'.format(src.shape)

      src_col_mean, src_col_std = OpenvinoAlignment.normalize(src, axis=0)
      dst_col_mean, dst_col_std = OpenvinoAlignment.normalize(dst, axis=0)

      u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
      r = np.matmul(u, vt).T

      transform = np.empty((2, 3))
      transform[:, 0:2] = r * (dst_col_std / src_col_std)
      transform[:, 2] = dst_col_mean.T - np.matmul(transform[:, 0:2], src_col_mean.T)
      return transform



  def extract(self, frame, roi, landmark):
      return self._preprocess(frame,roi,landmark)



  def _preprocess(self, frame, roi, landmark):
      
      image = frame.copy()
      input = cut_roi(image, roi)
      

      self._align_roi(input, landmark)
      # input = resize_input(input, self._input_shape)
      return input


  def _align_roi(self,image,landmark):
      scale = np.array((image.shape[1], image.shape[0]))
      desired_landmark = np.array(self._REFERENCE_LANDMARKS, dtype=np.float64) * scale
      landmark1 = landmark * scale

      transform = OpenvinoAlignment.get_transform(desired_landmark, landmark1)
      cv2.warpAffine(image, transform, tuple(scale), image, flags=cv2.WARP_INVERSE_MAP)

