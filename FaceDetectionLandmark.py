
from  .FaceDetection import  *
from  .FaceLandmark import  *
import cv2

class FaceDetectionLandmark:
    def __init__(self,detections_models=None,landmark_models=None):
        self._detector = FaceDetection(detections_models)
        self._landmark = FaceLandmark(landmark_models)

    def extract(self,bgr_image,rgb_image,master_detection_type,slave_detection_type,master_landmark_type,slave_landmark_type):
        face_region = self._detector.detect(bgr_image,rgb_image,master_detection_type,slave_detection_type)

        if (master_detection_type == DetectionType.CV_YUNET):
           return face_region,face_region
        else:
            if (master_detection_type == DetectionType.CV_DNN or master_detection_type == DetectionType.CV_CASCADE) and\
                master_landmark_type == LandmarkType.DLIB:
                face_region = FaceUtility.convert_rect(face_region, master_detection_type, DetectionType.DLIB)


            face_region, face_landmark = self._landmark.extract(bgr_image, rgb_image,face_region, master_landmark_type)


            if face_region and FaceLandmark.is_empty(face_landmark):
               return face_region,face_landmark
            else:
                if master_detection_type == DetectionType.MTCNN:
                    face_region = self._detector.detect(bgr_image, rgb_image, slave_detection_type, slave_detection_type)
                    if ( slave_detection_type == DetectionType.CV_DNN or slave_detection_type == DetectionType.CV_CASCADE) and \
                            slave_landmark_type == LandmarkType.DLIB:
                        face_region = FaceUtility.convert_rect(face_region, slave_detection_type, DetectionType.DLIB)


                    face_region, face_landmark = self._landmark.extract(bgr_image, rgb_image, face_region, slave_landmark_type)
                    if master_landmark_type == LandmarkType.MTCNN and slave_landmark_type == LandmarkType.DLIB:
                        face_landmark = FaceUtility.convert_landmark_dlib_to_mtcnn(face_landmark)
                return face_region,face_landmark
