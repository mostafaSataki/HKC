import dlib
from enum import  Enum
import cv2
from  .FaceUtility import  *
from  .RectUtility import  *
from .open_vino.face_detector import *
from openvino.inference_engine import IECore


class FaceDetection:

    def __init__(self,all_models = None):
        if all_models == None:
            all_models =[DetectionType.DLIB,DetectionType.CV_CASCADE,DetectionType.CV_DNN,DetectionType.MTCNN,
                         DetectionType.OPENVINO_RETAIL, DetectionType.OPENVINO_ADAS]


        if DetectionType.DLIB in all_models:
            self.load_dlib()
        if DetectionType.CV_CASCADE in all_models:
            self.load_cv_cascade()
        if DetectionType.CV_DNN in all_models:
            self.load_cv_dnn()
        if DetectionType.MTCNN in all_models:
            pass
        if DetectionType.OPENVINO_RETAIL in all_models:
            self.load_openvino_retail()
        if DetectionType.OPENVINO_ADAS in all_models:
            self.load_openvino_adas()
        if DetectionType.CV_YUNET:
            self.load_yunet()


    def load_yunet(self):
        score_thresh = 0.9
        nms_thresh = 0.3

        self._yunet_detector = cv2.FaceDetectorYN_create(r"D:\Models\yunet.onnx", "", (150, 150), score_thresh, nms_thresh)

    def load_dlib(self):
        self._dlib_detector = dlib.get_frontal_face_detector()


    def load_cv_cascade(self):
        cascade_face_filename = r'D:\library\opencv4\opencv4.10\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml'
        self._cascade_detector = cv2.CascadeClassifier(cascade_face_filename)

    def load_cv_dnn(self):
        # model_filename = r'C:\Source\Repo\Detection\x64\deploy.prototxt'
        # config_filename = r'C:\Source\Repo\Detection\x64\res10_300x300_ssd_iter_140000_fp16.caffemodel'
        models_path = r'C:\Source\Repo\FaceLiveness\x64\assets\model'
        model_filename = os.path.join(models_path, r'opencv_face_detector.pbtxt')
        config_filename = os.path.join(models_path, r'opencv_face_detector_uint8.pb')

        self._cv_dnn = cv2.dnn.readNet(model_filename,config_filename)

    def get_config(self, device):
        config = {
            "PERF_COUNT": "YES" if self.perf_count else "NO",
        }
        if device == 'GPU' and self.gpu_ext:
            config['CONFIG_FILE'] = self.gpu_ext
        return config

    def load_openvino_retail(self):
        # model_filename = r'E:\Database\openvino\intel\face-detection-retail-0004\FP32\face-detection-retail-0004.xml'
        model_filename = r'F:\dataset\openvino\intel\face-detection-retail-0004\FP32\face-detection-retail-0004.xml'
        ie = IECore()
        # ie.add_extension('', 'CPU')
        self._openvino_detector_retail = FaceDetector(ie, model_filename,
                                                      (0,0),
                                                      confidence_threshold=0.6,
                                                      roi_scale_factor=1.15)
        self.perf_count = False
        self._openvino_detector_retail.deploy('CPU', self.get_config('CPU'))

    def load_openvino_adas(self):
        model_filename = r'F:\dataset\openvino\intel\face-detection-adas-0001\FP32\face-detection-adas-0001.xml'
        # model_filename = r'E:\Database\openvino\intel\face-detection-adas-0001\FP16-INT8\face-detection-adas-0001.xml'
        ie = IECore()
        # ie.add_extension('', 'CPU')
        self._openvino_detector_adas = FaceDetector(ie, model_filename,
                                                      (0,0),
                                                      confidence_threshold=0.6,
                                                      roi_scale_factor=1.15)
        self.perf_count = False
        self._openvino_detector_adas.deploy('CPU', self.get_config('CPU'))

    def detect_regions(self,bgr_image,rgb_image, type):
        if type == DetectionType.MTCNN:
            regions =  self._detect_mtcnn()

        elif type == DetectionType.CV_DNN:
            regions = self._detect_cv_dnn(bgr_image)

        elif type == DetectionType.CV_CASCADE:
            regions = self._detect_cascade(bgr_image)

        elif type == DetectionType.DLIB:
            regions = self._dlib_detector(bgr_image)

        elif type == DetectionType.OPENVINO_RETAIL:
            regions = self._detect_openvino_retail(bgr_image)

        elif type == DetectionType.OPENVINO_ADAS:
            regions = self._detect_openvino_adas(bgr_image)

        elif type == DetectionType.CV_YUNET:
            regions = self._detect_yunet(bgr_image)

        return regions

    def crop_regions(self,bgr_image,rgb_image, type,expand_size =(0,0)):
        regions = self.detect_regions(bgr_image,rgb_image,type)
        regions = CvUtility.expandRects(regions, expand_size)
        back_rect = CvUtility.getImageRect(bgr_image)
        regions = RectUtility.cropRects(back_rect,regions)
        return CvUtility.crop_rois(bgr_image,regions)
        


    def detect(self, bgr_image,rgb_image, master_type, slave_type):
        regions = self.detect_regions(bgr_image,rgb_image,master_type)
        if master_type == DetectionType.CV_YUNET:
            return regions
        if regions != None:
            if len(regions):
              result,_ = FaceUtility.get_max_rect(regions)
            else :
                if master_type != DetectionType.MTCNN:
                    regions = self._detect_regions(bgr_image,rgb_image,slave_type)
                    if len(regions):
                      result,_ = FaceUtility.get_max_rect(regions)
                      result = FaceUtility.convert_rect(result, master_type, slave_type)



        else :
            result = None

        return result

    def _detect_mtcnn(self):
        return None

    def _detect_cascade(self,image):

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self._cascade_detector.detectMultiScale(gray_image, 1.1, 5)

    def _detect_cv_dnn(self,image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self._cv_dnn.setInput(blob)
        result = self._cv_dnn.forward()
        regions = []
        max_confidence = 0
        for i in range(0, result.shape[2]):
            confidence = result[0, 0, i, 2]
            if confidence > 0.6 and confidence > max_confidence:
                box = result[0, 0, i, 3:7] * np.array([w, h, w, h])
                regions.append([int(box[0]),int(box[1]),int(box[2]),int(box[3])])
                max_confidence  = confidence
        return regions

    def _detect_openvino_retail(self,image):
        rois = self._openvino_detector_retail.infer((image,))
        return FaceUtility.openvino_rects_to_cv(rois)

    def _detect_openvino_adas(self,image):
        rois = self._openvino_detector_adas.infer((image,))
        return FaceUtility.openvino_rects_to_cv(rois)



    def _detect_dlib(self,image):
        dlib_image = FaceDetection
        regions = self._dlib_detector(image, 1)
        if len(regions) == 0:
            if slave_type == DetectionType.CV_CASCADE:
                gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                regions = self._cascade_detector.detectMultiScale(gray_image, 1.1, 5, minSize=(30, 30),
                                                                  flags=cv2.CASCADE_SCALE_IMAGE)
                regions = FaceUtility.cv_rects_to_dlib(regions)
        return regions

    def _detect_yunet(self,image):
        self._yunet_detector.setInputSize((image.shape[1], image.shape[0]))
        result, faces = self._yunet_detector.detect(image)
        return faces










