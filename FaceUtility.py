
import cv2
import dlib
from enum import Enum
from PIL import Image
from HKC.CvUtility import *
from HKC.open_vino.face_detector import FaceDetector
class RectType(Enum):
    DLIB = 1,
    CV = 2

class DetectionType(Enum):
    DLIB = 1
    CV_CASCADE = 2
    CV_DNN = 3,
    MTCNN = 4,
    OPENVINO_RETAIL = 5,
    OPENVINO_ADAS = 6,
    CV_YUNET = 7



class LandmarkType(Enum):
    DLIB = 1,
    MTCNN = 2,
    OPENVINO = 3
    CV_YUNET = 4

class MatchType(Enum):
    EUCLIDEAN = 1,
    COSINE = 2


class FaceFeatureType(Enum):
    DLIB = 1,
    IR50 = 2,
    FACENET_128 = 3,
    FACENET_512 = 4,
    OPENVINO = 5
    SF = 6

class AlignmentType(Enum):
    DLIB = 1,
    MTCNN = 2,
    OPENVINO = 3,
    SF = 4


class RectConvertType(Enum):
    DLIB_CV = 1,
    CV_DLIB = 2

class FaceUtility:
    @staticmethod
    def cv_rect_to_dlib(rect):
        if rect :
            x, y, w, h = rect
            return dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        else : return None

    @staticmethod
    def cv_rects_to_dlib(rects):
        result = []
        for rect in rects:
            result.append(FaceUtility.cv_rect_to_dlib(rect))
        return  result

    @staticmethod
    def dlib_rect_to_cv(rect):
        if rect :
            r = dlib.rectangle()
            x1= r.left()
            y1 = r.top()
            x2 = r.right()
            y2 = r.bottom()

            return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        else :return None

    @staticmethod
    def dlib_rects_to_cv(rects):
        result = []
        for rect in rects:
            result.append(FaceUtility.dlib_rect_to_cv(rect))
        return  result


    @staticmethod
    def convert_rect(rect,src_type,dst_type):
        if (src_type == DetectionType.CV_CASCADE or src_type == DetectionType.CV_DNN) and dst_type == DetectionType.DLIB:
            return FaceUtility.cv_rect_to_dlib(rect)
        elif src_type == DetectionType.DLIB or (dst_type == DetectionType.CV_CASCADE or dst_type == DetectionType.CV_DNN) :
            return FaceUtility.dlib_rect_to_cv(rect)

    @staticmethod
    def convert_rects(rects,src_type,dst_type):
        if ( src_type == DetectionType.CV_CASCADE or src_type == DetectionType.CV_DNN) and dst_type == DetectionType.DLIB:
            return FaceUtility.cv_rects_to_dlib(rects)
        elif src_type == DetectionType.DLIB and (dst_type == DetectionType.CV_CASCADE or dst_type == DetectionType.CV_DNN):
            return FaceUtility.dlib_rects_to_cv(rects)

    @staticmethod
    def dlib_rect_area( rect):
        if rect :
          return (rect.right() - rect.left()) * (rect.bottom() - rect.top())
        else: return None

    @staticmethod
    def cv_rect_area(rect):
        return (rect[2] * rect[3])

    @staticmethod
    def get_area(rect):
        if isinstance(rect,dlib.rectangle):
            return FaceUtility.dlib_rect_area(rect)
        elif isinstance(rect , list):
            return FaceUtility.cv_rect_area(rect)

    @staticmethod
    def get_max_rect(rects):
        result = None
        if len(rects)== 0:
            return result

        result = rects[0]
        max_area = 0
        max_index = 0
        for i, rect in enumerate(rects):
            cur_area = FaceUtility.get_area(rect)
            if cur_area > max_area:
                result = rect
                max_area = cur_area
                max_index = i

        return result,max_index

    @staticmethod
    def get_dlib_image(image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    @staticmethod
    def get_pil_image(image):
        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(result)
        return result

    @staticmethod
    def get_cv_from_pil_image(image):
        return np.asarray(image)

    @staticmethod
    def draw_regions(image,regions,color=(0,255,0),thickness = 2):
        result = image.copy()
        for i, region in enumerate(regions):
            x1, y1, x2, y2, _ = region
            result = cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
        return result

    @staticmethod
    def mtcnn_to_cv_rect(rect):
        x1,y1,x2,y2,_ = rect
        return [int(x1),int(y1),int(x2-x1),int(y2-y1)]

    @staticmethod
    def mtcnn_to_cv_rects(rects):
        result = []
        for rect in rects:
            result.append(FaceUtility.mtcnn_to_cv_rect(rect))
        return  result

    @staticmethod
    def get_rect_2points(rect):
        if isinstance(rect,dlib.rectangle):
            return ( int(rect.left()),int(rect.top())),(int(rect.right()),int(rect.bottom()))
        else :return (int(rect[0]),int(rect[1])),(int(rect[0]+rect[2]),int(rect[1]+rect[3]))

    @staticmethod
    def draw_region_landmark(image, region, landmark, region_color=(0, 255, 0), landmark_color=(0, 0, 255),
                               landmark_rad=4, region_thicness=2):
        if region:
            p1, p2 = FaceUtility.get_rect_2points(region)
            image = cv2.rectangle(image, p1, p2, region_color, region_thicness)

        # if not (landmark == None):
        if isinstance(landmark,dlib.full_object_detection):
            points = FaceUtility.get_dlib_landmark_points(landmark)
            for pnt in points :
                image = cv2.circle(image, (pnt[0],pnt[1]), landmark_rad, landmark_color, -1)
        else :
            if isinstance(region,dlib.rectangle):
                for i, pnt in enumerate(landmark):
                    image = cv2.circle(image, (pnt[0], pnt[1]), landmark_rad, landmark_color, -1)
            else :
                x,y,w,h = region[0],region[1],region[2],region[3]
                for pnt in landmark:
                   # image = cv2.circle(image, (int(pnt[0]*w)+x,int(pnt[1]*h)+y), landmark_rad, landmark_color, -1)
                    image = cv2.circle(image, (int(pnt[0]),int(pnt[1])), landmark_rad, landmark_color, -1)
        return image


    @staticmethod
    def draw_regions_landmarks(image,regions,landmarks,region_color=(0,255,0),landmark_color=(0,0,255),landmark_rad = 4,region_thicness = 2):
        result = image.copy()

        for i, region in enumerate(regions):
            landmark = landmarks[i]
            result  = FaceUtility.draw_region_landmark(result,region,landmark,region_color,landmark_color,landmark_rad,region_thicness)

        return result


    @staticmethod
    def get_dlib_landmark_points(landmark):
        result = []
        if isinstance(landmark, dlib.full_object_detection):
            for pnt in landmark.parts():
                result.append([int(pnt.x), int(pnt.y)])
        return result

    @staticmethod
    def convert_landmark_dlib_to_mtcnn(dlib_points):
        points = FaceUtility.get_dlib_landmark_points(dlib_points)
        result = []
        p1 = CvUtility.mean_point(points[0],points[1])
        p2 = CvUtility.mean_point(points[2], points[3])
        result.append(p1)
        result.append(p2)
        result.append(points[4])
        eye_dist = CvUtility.points_distance_y( points[4] , points[0])
        mouth_dist = eye_dist // 2
        result.append([p1[0],points[4][1]+mouth_dist])
        result.append([p2[0], points[4][1] + mouth_dist])
        return result

    @staticmethod
    def remove_rect_border(rect,border=None):
        if rect and border:
            if isinstance(rect,dlib.rectangle):
                l = rect.left()
                t = rect.top()
                r = rect.right()
                b = rect.bottom()

                t -= border[0]
                b -= border[0]
                l -= border[2]
                r -= border[2]

                return dlib.rectangle(l,t,r,b)

            else :
                x,y,w,h = rect
                x -= border[0]
                y -= border[2]
                return [x,y,w,h]



    @staticmethod
    def remove_landmark_border(landmark,border=None):
        result = landmark
        if len(landmark) and border:
            result = []
            l = border[2]
            t = border[0]
            if border:
                for pnt in landmark:
                    result.append(CvUtility.offset_point(pnt,[-l,-t]))

        return result

    @staticmethod
    def cvrect_to_openvino(cv_rect):
        result = FaceDetector.Result([0,0,0,cv_rect[0],cv_rect[1],cv_rect[2],cv_rect[3]])
        # result.position = (cv_rect[0],cv_rect[1])
        # result.size = (cv_rect[2],cv_rect[3])
        return result

    @staticmethod
    def cvrects_to_openvino(cv_rects):
        result = []
        for cv_rect in cv_rects:
            result.append(FaceUtility.cvrect_to_openvino(cv_rect))
        return result


    @staticmethod
    def openvino_rect_to_cv(ov_rect):
        return [int(ov_rect.position[0]), int(ov_rect.position[1]), int(ov_rect.size[0]), int(ov_rect.size[1])]

    @staticmethod
    def openvino_rects_to_cv(ov_rects):
        result = []
        for ov_rect in ov_rects:
            result.append(FaceUtility.openvino_rect_to_cv(ov_rect))
        return result





