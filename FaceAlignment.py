from enum import  Enum
import dlib
from .FaceUtility import *
from .align.face_align import *
from .OpenvinoAlignment import *

class FaceAlignment:
    def __init__(self,all_models = None):
        if all_models == None:
            all_models = [AlignmentType.DLIB,AlignmentType.MTCNN,AlignmentType.SF]
            # , AlignmentType.OPENVINO

        # if AlignmentType.OPENVINO in all_models:
        #    self._load_openvino()

        if AlignmentType.SF in all_models:
            self._load_SF()

    def extract(self,bgr_image,rgb_image,region, landmark_data,type):
        if type == AlignmentType.DLIB:
           return dlib.get_face_chip(rgb_image, landmark_data)
        elif type == AlignmentType.MTCNN:
            crop_size = 112
            scale = crop_size / 112.

            reference = get_reference_facial_points(default_square=True) * scale

            return warp_and_crop_face(rgb_image,landmark_data, reference, crop_size=(crop_size, crop_size))

        # elif type == AlignmentType.OPENVINO:
        #     # return self._openvino_aligner.extract(bgr_image,FaceUtility.cvrect_to_openvino(region), landmark_data)
        #     return self._openvino_aligner.extract(bgr_image,FaceUtility.cvrect_to_openvino(region), landmark_data)

        elif type == AlignmentType.SF:
            return self._sf_aligner.alignCrop(bgr_image, region)





    def _load_openvino(self):
        self._openvino_aligner = OpenvinoAlignment()

    def _load_SF(self):
        self._sf_aligner = cv2.FaceRecognizerSF_create(r"E:\Models\face_recognizer_fast.onnx", "")
