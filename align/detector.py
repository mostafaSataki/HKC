import numpy as np
import torch
from torch.autograd import Variable
from .get_nets import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from .first_stage import run_first_stage
from ..FaceUtility import *
import cv2

class MTCNNFace_LandmarkDetection:
    def __init__(self):
        self._load_models()

    def _load_models(self):
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.onet.eval()


    def draw(self,image,regions,landmarks,region_color =(0,255,0),landmark_color = (0,0,255)):
        result = image.copy()
        for i,region in enumerate(regions):
            landmark = landmarks[i]
            x1,y1,x2,y2,_ = region
            result = cv2.rectangle(result,(int(x1),int(y1)),(int(x2),int(y2)),region_color,4)
            for l in range(len(landmark)//2):
                result = cv2.circle(image,(int(landmark[l]),int(landmark[5+l])),4,landmark_color,-1)
        return result


    def _get_landmark_points(self,landmark):
        result = []
        for j in range(5):
            result.append([landmark[j],landmark[j+5]])

        return result

    def _get_landmarks_points(self,landmarks):
        result = []
        for landmark in landmarks:
          result.append(self._get_landmark_points(landmark))
        return result

    def detect_faces(self,cv_image, min_face_size = 20.0,
                     thresholds=[0.6, 0.7, 0.77],
                     nms_thresholds=[0.7, 0.7, 0.7]):
        """
          Arguments:
              image: an instance of PIL.Image.
              min_face_size: a float number.
              thresholds: a list of length 3.
              nms_thresholds: a list of length 3.

          Returns:
              two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
              bounding boxes and facial landmarks.
          """

        image = FaceUtility.get_pil_image(cv_image)
        # LOAD MODELS



        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        output = self.rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        output = self.onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return FaceUtility.mtcnn_to_cv_rects( bounding_boxes),self._get_landmarks_points(landmarks)


    def detect_face(self,image, min_face_size = 20.0,
                     thresholds=[0.6, 0.7, 0.8],
                     nms_thresholds=[0.7, 0.7, 0.7]):
        regions,landmarks = self.detect_faces(image,min_face_size,thresholds)
        if len(regions) :
            region,index = FaceUtility.get_max_rect(regions)
            return region,landmarks[index]
        else: return (None,None)

    def correct(self,face_region, face_landmark, border):
        for i in range(4):
            face_region[i] = face_region[i] - border
        for i in range(10):
            face_landmark[i] = face_landmark[i] - border
        return face_region, face_landmark
    def correct2(self,face_region, face_landmark, border):
        for i in range(2):
            face_region[i] = face_region[i] - border
        for i in range(5):
            face_landmark[i][0] = face_landmark[i][0] - border
            face_landmark[i][1] = face_landmark[i][1] - border
        return face_region, face_landmark
    def applayCLAHE(self,img):
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        lab_planes = cv2.split(lab)

        lab_planes = list(lab_planes)
        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def detect_single_face(self,img, borders=[10, 20, 40, 60, 80]):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        black_color = [0, 0, 0]
        white_color = [255, 255, 255]
        border_color = white_color

        for i, border in enumerate(borders):
            if i == 2:
                img_rgb = self.applayCLAHE(img_rgb)
                border_color = black_color

            img_br = cv2.copyMakeBorder(img_rgb, border, border, border, border, cv2.BORDER_CONSTANT,
                                        value=border_color)
            # image = Image.fromarray(img_br)
            try:
                bounding_boxes, landmarks = self.detect_faces(img_br)
            except:
                return None, None
            if len(bounding_boxes) == 0:
                continue
            bounding_box, landmark = self.correct2(bounding_boxes[0], landmarks[0], border)
            return bounding_box, landmark
        return None, None

    def detect_face_border(self,image, min_face_size = 20.0,
                     thresholds=[0.6, 0.7, 0.8],
                     nms_thresholds=[0.7, 0.7, 0.7],
                           border=None):
        image_b = CvUtility.add_border(image,border)
        region,landmark = self.detect_face(image,min_face_size,thresholds,nms_thresholds)


def detect_faces( image, min_face_size=20.0,
                    thresholds=[0.6, 0.7, 0.8],
                    nms_thresholds=[0.7, 0.7, 0.7]):
    """
    Arguments:
        image: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """

    # LOAD MODELS
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

    # BUILD AN IMAGE PYRAMID
    width, height = image.size
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size/min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    for s in scales:
        boxes = run_first_stage(image, pnet, scale = s, threshold = thresholds[0])
        bounding_boxes.append(boxes)

    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]

    # use offsets predicted by pnet to transform bounding boxes
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    # shape [n_boxes, 5]

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, image, size = 24)
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile = True)
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    probs = output[1].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, image, size = 48)
    if len(img_boxes) == 0: 
        return [], []
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile = True)
    output = onet(img_boxes)
    landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    probs = output[2].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode = 'min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes, landmarks
