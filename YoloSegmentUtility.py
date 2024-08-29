import os
import numpy as np
from HKC.FileUtility import *
from typing import List, Tuple
from HKC.LabelmeJson import *
from HKC.CvUtility import *

class YoloSegmentUtility:
    def __init__(self,class_names,is_rect_contour = False, min_area_cofi = None):
        self.class_names = class_names
        self.is_rect_contour = is_rect_contour
        self.min_area_cofi = min_area_cofi

    
    def _preprocess_mask(self, mask: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
        mask = mask.cpu().numpy()
        mask = cv2.resize(mask, (original_width, original_height))
        mask = (mask > 0.5).astype(np.uint8) * 255
        return mask

    def _find_valid_contour(self, mask: np.ndarray, min_contour_area: float) -> List[np.ndarray]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
            if len(valid_contours) > 0 :
               return valid_contours[0]
            else: return None


    def get_segment_data(self, yolo_results, image):
        original_height, original_width = image.shape[:2]

        if self.min_area_cofi is not None:
            min_contour_area = original_width * original_height *  self.min_area_cofi

        result = []
        for yolo_result in yolo_results:
            masks = yolo_result.masks
            if masks is None:
                continue

            for i, mask in enumerate(masks.data):
                class_name = self.class_names[int(yolo_result.boxes.cls[i].item())]


                mask = self._preprocess_mask(mask,original_width,original_height)
                contour = self._find_valid_contour(mask,min_contour_area)
                if self.is_rect_contour and contour is not None:
                    approx_contour = CvUtility.approximate_rect_contour(contour)
                    if approx_contour is not None:
                        contour_dim = CvUtility.get_rect_contour_dimension(approx_contour)
                        reg_image = CvUtility.rectify_rect_image(image, approx_contour, contour_dim)
                        result.append((reg_image, class_name,approx_contour))
                        
        return result

                    
