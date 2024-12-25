import os
import numpy as np
from HKC.FileUtility import *
from typing import List, Tuple
from HKC.LabelmeJson import *
from HKC.CvUtility import *

class YoloDetectionUtility:
    def __init__(self,class_names, min_area_cofi = None):
        self.class_names = class_names
        self.min_area_cofi = min_area_cofi

    

    # def _find_valid_contour(self, mask: np.ndarray, min_contour_area: float) -> List[np.ndarray]:
    #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    #         if len(valid_contours) > 0 :
    #            return valid_contours[0]
    #         else: return None




    def get_detection_data(self, yolo_results, image):
        original_height, original_width = image.shape[:2]

        if self.min_area_cofi is not None:
            min_contour_area = original_width * original_height * self.min_area_cofi

        result = []
        for yolo_result in yolo_results:
            boxes = yolo_result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes.data):
                # Convert to CPU and to float list
                box_cpu = box.cpu().float()

                # Explicitly unpack coordinates and other info
                x1, y1, x2, y2, conf, cls = box_cpu.tolist()

                # Get class name
                class_name = self.class_names[int(cls)]

                result.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_name': class_name
                })

        return result

                    
