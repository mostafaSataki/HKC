import math
import os.path

import numpy as np
from PIL import Image
import torch
from .model import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square, _preprocess,\
    nms2,calibrate_box2,get_image_boxes2,convert_to_square2
import cv2
class MTCNN:
    def __init__(self):
        self.trace_models()
        if os.path.exists(self._pnet_filename):
            self._pnet = torch.jit.load(self._pnet_filename)
        else :self._pnet = PNet()
        if os.path.exists(self._rnet_filename):
            self._rnet = torch.jit.load(self._rnet_filename)
        else :self._rnet = RNet()
        if os.path.exists(self._onet_filename):
            self._onet = torch.jit.load(self._onet_filename)
        else:self._onet = ONet()
            
        
        
        self._onet.eval()
        self._rnet.eval()
        self._pnet.eval()
 
 
    def trace_models(self):
        models_path = r'C:\Source\Repo\PayeshChehre\test2'
        self._onet_filename = os.path.join(models_path, 'ONet.pt')
        self._rnet_filename = os.path.join(models_path, 'RNet.pt')
        self._pnet_filename = os.path.join(models_path, 'PNet.pt')

    def trace(self,model_index,example):
        if model_index == 0:
            if os.path.exists(self._pnet_filename):
                return
            traced_script_module = torch.jit.trace(self._pnet, example)
            traced_script_module.save(self._pnet_filename)

        elif model_index == 1:
            if os.path.exists(self._rnet_filename):
                return
            traced_script_module = torch.jit.trace(self._rnet,example)
            traced_script_module.save(self._rnet_filename)

        elif model_index == 2:
            if os.path.exists(self._onet_filename):
                return
            traced_script_module = torch.jit.trace(self._onet,example)
            traced_script_module.save(self._onet_filename)



    def _applayCLAHE(self,img):
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        lab_planes = cv2.split(lab)

        lab_planes = list(lab_planes)
        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _correct(self,face_region, face_landmark, border):
        for i in range(4):
            face_region[i] = face_region[i] - border
        for i in range(10):
            face_landmark[i] = face_landmark[i] - border
        return face_region, face_landmark

    def _correct2(self,face_region, face_landmark, border):
        f_region = []

        f_region.append( int(face_region[0] - border))
        f_region.append(int(face_region[1] - border))
        f_region.append(int((face_region[2] -face_region[0]+1) ))
        f_region.append(int((face_region[3] - face_region[1]+1) ))

        f_landmark = []
        for i in range(5):
            f_landmark.append( [int(face_landmark[i] - border) ,int(face_landmark[i+5] - border)])
            # face_landmark[i][1] = face_landmark[i][1] - border
        return f_region, f_landmark

    def getLarge(self,face_regions,landmarks):
        max_area = 0
        max_index = 0
        for i in range(len(face_regions)):
            if face_regions[i][2] * face_regions[i][3] > max_area:
                max_area = (face_regions[i][2] -face_regions[i][0]+1) * (face_regions[i][3] - face_regions[i][1]+1)
                max_index = i
        return face_regions[max_index], landmarks[max_index]

    def detect_face_single_auto(self,img, borders=[0, 20, 40, 60, 80]):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        black_color = [0, 0, 0]
        white_color = [255, 255, 255]
        border_color = white_color

        for i, border in enumerate(borders):
            if i == 2:
                img_rgb = self._applayCLAHE(img_rgb)
                border_color = black_color

            img_br = cv2.copyMakeBorder(img_rgb, border, border, border, border, cv2.BORDER_CONSTANT,
                                        value=border_color)
            image = Image.fromarray(img_br)
            # try:
            bounding_boxes, landmarks = self.detect(image)
            # except:
            #     pass
            #     # return None, None
            # cv2.imwrite(r'd:\input_p.bmp',img_br)
            if len(bounding_boxes) == 0:
                continue
            bounding_box, landmark = self.getLarge(bounding_boxes, landmarks)
            bounding_box, landmark = self._correct2(bounding_box, landmark, border)
            return bounding_box, landmark
        return None, None

    def draw_face_landmark(self,image, face_region, face_landmark):

        for i in range(5):
            cv2.circle(image, (int(face_landmark[i ][0]), int(face_landmark[i ][1])), 3, (0, 0, 255), -1)
        cv2.rectangle(image, (int(face_region[0]), int(face_region[1])),
                      (int( face_region[0]+face_region[2]), int(face_region[1] + face_region[3])), (0, 255, 0), 2)
        return image

    def _run_first_stage(self,image, net, scale, threshold):
        """
            Run P-Net, generate bounding boxes, and do NMS.
        """
        width, height = image.size
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img = np.asarray(img, 'float32')
        img = torch.FloatTensor(_preprocess(img))

        output = net(img)
        self.trace(0,img)
        probs = output[1].data.numpy()[0, 1, :, :]
        offsets = output[0].data.numpy()


        boxes = self._generate_bboxes(probs, offsets, scale, threshold)

        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)

        return boxes[keep]

    def _run_first_stage2(self, image, net, scale, threshold):
        """
            Run P-Net, generate bounding boxes, and do NMS.
        """
        width, height = image.size
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img.save(r'd:\sample_python.bmp')
        img = np.asarray(img, 'float32')
        img = torch.FloatTensor(_preprocess(img))

        output = net(img)


        probs = output[1][0, 1, :, :]
        offsets = output[0]

        boxes = self._generate_bboxes2(probs, offsets, scale, threshold)
        if boxes.size(dim =0) == 0:
            return torch.Tensor([])

        keep = nms2(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]
    def _generate_bboxes(self,probs, offsets, scale, threshold):
        """
           Generate bounding boxes at places where there is probably a face.
        """
        stride = 2
        cell_size = 12

        inds = np.where(probs > threshold)

        if inds[0].size == 0:
            return np.array([])

        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]

        offsets = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        # P-Net is applied to scaled images, so we need to rescale bounding boxes back
        bounding_boxes = np.vstack([
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score, offsets
        ])

        return bounding_boxes.T

    def _generate_bboxes2(self,probs, offsets, scale, threshold):
        """
           Generate bounding boxes at places where there is probably a face.
        """
        stride = 2
        cell_size = 12

        inds = torch.where(probs > threshold)

        if inds[0].size(dim=0) == 0:
            return torch.Tensor([])

        a= inds[0]
        b = inds[1]
        c =offsets[0, 0, inds[0], inds[1]]
        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]



        offsets = torch.vstack([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        # P-Net is applied to scaled images, so we need to rescale bounding boxes back
        bounding_boxes = torch.vstack([
            torch.round((stride * inds[1] + 1.0) / scale),
            torch.round((stride * inds[0] + 1.0) / scale),
            torch.round((stride * inds[1] + 1.0 + cell_size) / scale),
            torch.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score, offsets
        ])

        return bounding_boxes.t()

    def detect(self,image, min_face_size=20.0,
                 # thresholds=[0.6, 0.7, 0.8],
                 thresholds=[0.6, 0.7, 0.76],
                  nms_thresholds=[0.7, 0.7, 0.7]):
        return self.detect2(image, min_face_size, thresholds, nms_thresholds)
        width, height = image.size
        min_length = min(height, width)
        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        scales = []
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1
        bounding_boxes = []
        for s in scales:  # run P-Net on different scales
            boxes = self._run_first_stage(image, self._pnet, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2
        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = torch.FloatTensor(img_boxes)
        output = self._rnet(img_boxes)
        self.trace(1, img_boxes)
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
        img_boxes = torch.FloatTensor(img_boxes)
        output = self._onet(img_boxes)
        self.trace(2, img_boxes)
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

        return bounding_boxes, landmarks

    def detect2(self,image, min_face_size=20.0,
                 # thresholds=[0.6, 0.7, 0.8],
                 thresholds=[0.6, 0.7, 0.76],
                  nms_thresholds=[0.7, 0.7, 0.7]):

        width, height = image.size
        min_length = min(height, width)
        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        scales = []
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1
        bounding_boxes = []
        for s in scales:  # run P-Net on different scales
            boxes = self._run_first_stage2(image, self._pnet, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)


        bounding_boxes = [i for i in bounding_boxes if i.size(0)]
        bounding_boxes = torch.vstack(bounding_boxes)

        keep = nms2(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box2(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        bounding_boxes = convert_to_square2(bounding_boxes)
        bounding_boxes[:, 0:4] = torch.round(bounding_boxes[:, 0:4])

        # STAGE 2
        img_boxes = get_image_boxes2(bounding_boxes, image, size=24)
        img_boxes = torch.FloatTensor(img_boxes)
        output = self._rnet(img_boxes)

        offsets = output[0]  # shape [n_boxes, 4]
        probs = output[1]  # shape [n_boxes, 2]

        keep = torch.where(probs[:, 1] > thresholds[1])[0]

        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms2(bounding_boxes, nms_thresholds[1])
        if len(keep) == 0:
            return [], []
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box2(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square2(bounding_boxes)
        bounding_boxes[:, 0:4] = torch.round(bounding_boxes[:, 0:4])

        # STAGE 3
        img_boxes = get_image_boxes2(bounding_boxes, image, size=48)
        if len( img_boxes) == 0:
            return [], []
        img_boxes = torch.FloatTensor(img_boxes)
        output = self._onet(img_boxes)

        landmarks = output[0]  # shape [n_boxes, 10]
        offsets = output[1]  # shape [n_boxes, 4]
        probs = output[2]  # shape [n_boxes, 2]

        keep = torch.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        # bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        bounding_boxes[:, 4] = torch.reshape( probs[keep, 1],(-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = xmin.unsqueeze( 1) + width.unsqueeze( 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = ymin.unsqueeze( 1) + height.unsqueeze( 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box2(bounding_boxes, offsets)
        keep = nms2(bounding_boxes, nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes.detach().numpy(), landmarks.detach().numpy()
