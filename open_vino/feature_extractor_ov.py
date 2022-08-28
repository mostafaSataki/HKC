

import numpy as np

from .ie_module import Module
from .utils import resize_input
# from openvino.inference_engine import IECore
from  ..FaceUtility import  *
from HKC.CvUtility import *

class FaceExtractorOV(Module):
   def __init__(self,backbone):

       self.max_requests = 1
       ie = IECore()


       model_filename = r'F:\dataset\openvino\intel\face-reidentification-retail-0095\FP32\face-reidentification-retail-0095.xml'
       super(FaceExtractorOV, self).__init__(ie, model_filename)
       self._backbone = backbone
       if self._backbone == FaceFeatureType.OPENVINO:
           # self.exec_net = ie.load_network(self.model, "CPU", config=plugin_config, num_requests=max_requests)
           self._match_threshold =  0.5
           self.input_blob = next(iter(self.model.input_info))
           self.output_blob = next(iter(self.model.outputs))
           self.input_shape = self.model.input_info[self.input_blob].input_data.shape
           output_shape = self.model.outputs[self.output_blob].shape

   def extarct(self,bgr_image):
       if self._backbone == FaceFeatureType.OPENVINO:
           tensor = CvUtility.image_to_tensor(bgr_image,self.input_shape)
           super(FaceExtractorOV, self).enqueue({self.input_blob: tensor})
           result = [out[self.output_blob].buffer.flatten() for out in self.get_outputs()]
           return result[0]
    
