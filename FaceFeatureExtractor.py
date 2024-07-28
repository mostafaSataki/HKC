
from enum import Enum
import dlib
from .FaceUtility import *
from .util.extract_feature_v2 import *
from .util import Facenet
from .util import Facenet512
from .open_vino.feature_extractor_ov import *


class FaceFeatureExtrator:
    def __init__(self,all_models = None):
        if all_models == None:
            all_models = [FaceFeatureType.DLIB,FaceFeatureType.IR50,FaceFeatureType.FACENET_128,
                          FaceFeatureType.FACENET_512,FaceFeatureType.SF]
            all_models = [FaceFeatureType.IR50]
            # FaceFeatureType.OPENVINO,

        if FaceFeatureType.DLIB in all_models:
            self._load_dlib()
        if FaceFeatureType.IR50 in all_models:
            self._load_ir50()
        if FaceFeatureType.FACENET_128 in all_models:
            self._load_facenet128()
        if FaceFeatureType.FACENET_512 in all_models:
            self._load_facenet512()

        # if FaceFeatureType.OPENVINO in all_models:
        #     self._load_openvino()
        if FaceFeatureType.SF in all_models:
            self._load_SF()


    def feature_extract(self,bgr_image,rgb_image,type):
        if type == FaceFeatureType.DLIB:
            return self._dlib_fe.compute_face_descriptor(rgb_image)
        elif type == FaceFeatureType.IR50:
            return self._ir50_fe.extract(bgr_image)
        elif type == FaceFeatureType.FACENET_128:
            return  self._facenet128_fe.predict(rgb_image)[0,:]
        elif type == FaceFeatureType.FACENET_512:
            return  self._facenet512_fe(rgb_image)
        elif type == FaceFeatureType.OPENVINO:
            return  self._openvino_fe.extarct(rgb_image)
        elif type == FaceFeatureType.SF:
            return  self._sf_fe.feature(rgb_image)


    def _load_dlib(self):
        model_filename = r'E:\library\DLib\dlib-19.22\examples\build_cpu\Release\dlib_face_recognition_resnet_model_v1.dat'
        self._dlib_fe = dlib.face_recognition_model_v1(model_filename)

    def _load_ir50(self):
        self._ir50_fe = FaceFeatureExtractIR(FaceFeatureType.IR50,
                                    r'd:\Database\data_deep\face_models\face_evoLVe\backbone_ir50_ms1m_epoch120.pth')

    def _load_facenet128(self):
        self._facenet128_fe = Facenet.loadModel()

    def _load_facenet512(self):
        self._facenet512_fe = Facenet512.loadModel()

    def _get_config(self, device):
        config = {
            "PERF_COUNT": "YES" if self.perf_count else "NO",
        }
        if device == 'GPU' and self.gpu_ext:
            config['CONFIG_FILE'] = self.gpu_ext
        return config

    def _load_openvino(self):
        
        self._openvino_fe = FaceExtractorOV(FaceFeatureType.OPENVINO)

        self.perf_count = False
        self._openvino_fe.deploy('CPU', self._get_config('CPU'))

    def _load_SF(self):
        self._sf_fe = cv2.FaceRecognizerSF_create( r"E:\Models\face_recognizer_fast.onnx", "")
