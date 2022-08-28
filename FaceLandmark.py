from enum import Enum
import dlib
from .align.detector import MTCNNFace_LandmarkDetection
from .FaceUtility import *
from HKC.mtcnn.detector import *
# from .open_vino.landmarks_detector import *
# from openvino.inference_engine import IECore

class FaceLandmark:
    def __init__(self,all_models = None):
        if all_models == None:
            all_models =[LandmarkType.DLIB,LandmarkType.MTCNN,LandmarkType.OPENVINO]

        if LandmarkType.DLIB in all_models:
            self._load_dlib()
        if LandmarkType.MTCNN in all_models:
            self._load_MTCNN()
        # if LandmarkType.OPENVINO in all_models:
        #     self._load_OPENVINO()

    def extract(self,bgr_image,rgb_image,region, type,border=None):
        if type == LandmarkType.DLIB:
            return region, self._dlib_landmark(rgb_image,region)
        elif type == LandmarkType.MTCNN:
              # return self._mtcnn_face_landmark.detect_single_face(bgr_image)
              return self._mtcnn_face_landmark2.detect_face_single_auto(bgr_image)
        # elif type == LandmarkType.OPENVINO:
        #     r = FaceUtility.cvrect_to_openvino(region)
        #     return region, self._openvino_face_landmark.infer((rgb_image, [r]))[0]



    def _load_dlib(self):
        landmark_filename = r'E:\library\DLib\dlib-19.22\examples\build_cpu\Release\shape_predictor_5_face_landmarks.dat'
        self._dlib_landmark = dlib.shape_predictor(landmark_filename)

    def _load_MTCNN(self):
        self._mtcnn_face_landmark = MTCNNFace_LandmarkDetection()
        self._mtcnn_face_landmark2 = MTCNN()

    def get_config(self, device):
        config = {
            "PERF_COUNT": "YES" if self.perf_count else "NO",
        }
        if device == 'GPU' and self.gpu_ext:
            config['CONFIG_FILE'] = self.gpu_ext
        return config

    def _load_OPENVINO(self):
        model_filename = r'F:\dataset\openvino\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml'
        ie = IECore()

        QUEUE_SIZE = 16

        self._openvino_face_landmark = LandmarksDetector(ie,model_filename)
        self.perf_count = False
        self._openvino_face_landmark.deploy('CPU', self.get_config('CPU'), QUEUE_SIZE)
        
    @staticmethod
    def is_empty(landmark):
        result = False
        if isinstance(landmark,list):
            result = len(landmark) > 0
        elif isinstance(landmark, np.ndarray):
            result = landmark.size != 0
        else:
            result = landmark != None
        return result




