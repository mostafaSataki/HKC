import os

import cv2
import numpy as np
from HKC.FileUtility import *
from typing import List, Tuple
from HKC.LabelmeJson import *
from HKC.CvUtility import *
import matplotlib.pyplot as plt

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

    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    def detect_quadrilateral_corners2(self,contour, angle_tolerance=15, debug_plot=False):
        """
        Detect 4 corners of a quadrilateral from a given contour

        Parameters:
        - contour: Input contour points (numpy array)
        - angle_tolerance: Maximum allowed deviation from expected orthogonal angles (degrees)
        - debug_plot: Whether to generate a debug visualization

        Returns:
        - 4 corners of the quadrilateral
        """
        # Ensure contour is numpy array of integers
        contour = np.array(contour, dtype=np.int32)

        # Find bounding rectangle to create appropriate image size
        x, y, w, h = cv2.boundingRect(contour)
        blank_image = np.zeros((h + 20, w + 20), dtype=np.uint8)

        # Shift contour to fit in image
        shifted_contour = contour - [x - 10, y - 10]

        # Draw contour on blank image
        cv2.drawContours(blank_image, [shifted_contour], -1, 255, 1)

        # Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            blank_image,
            rho=1,  # Distance resolution
            theta=np.pi / 180,  # Angle resolution
            threshold=50,  # Minimum number of intersections to detect a line
            minLineLength=min(w, h) * 0.1,  # Minimum line length
            maxLineGap=10  # Maximum allowed gap between line segments
        )

        if lines is None or len(lines) < 4:
            return None
            # raise ValueError("Unable to detect 4 lines in the contour")

        # Convert lines to standard format
        line_params = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            line_params.append({
                'start': np.array([x1 + x - 10, y1 + y - 10]),
                'end': np.array([x2 + x - 10, y2 + y - 10]),
                'length': length,
                'angle': angle
            })

        # Helper functions for line processing
        def is_similar_angle(angle1, angle2, tolerance):
            """Check if two angles are similar within tolerance"""
            diff = abs(angle1 - angle2)
            return min(diff, 360 - diff) <= tolerance

        def group_lines_by_angle(lines, angle_tolerance):
            """Group lines with similar angles"""
            grouped_lines = []
            used_indices = set()

            for i, line in enumerate(lines):
                if i in used_indices:
                    continue

                group = [line]
                used_indices.add(i)

                for j, other_line in enumerate(lines):
                    if j in used_indices:
                        continue

                    if is_similar_angle(line['angle'], other_line['angle'], angle_tolerance):
                        group.append(other_line)
                        used_indices.add(j)

                grouped_lines.append(group)

            return grouped_lines

        # Group similar lines
        line_groups = group_lines_by_angle(line_params, angle_tolerance)

        # Select top 4 line groups by length
        line_groups.sort(key=lambda x: sum(line['length'] for line in x), reverse=True)

        # Select representative line from each group
        selected_lines = []
        for group in line_groups[:4]:
            # Select longest line in the group
            selected_lines.append(max(group, key=lambda x: x['length']))

        # Compute line intersections
        def line_intersection(line1, line2):
            """Compute intersection of two lines"""
            x1, y1 = line1['start']
            x2, y2 = line1['end']
            x3, y3 = line2['start']
            x4, y4 = line2['end']

            # Compute denominator
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

            # Parallel lines
            if abs(denom) < 1e-8:
                return None

            # Intersection point
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

            return np.array([px, py])

        # Compute corners
        corners = []
        for i in range(4):
            corner = line_intersection(selected_lines[i], selected_lines[(i + 1) % 4])
            if corner is not None:
                corners.append(corner)

        # Debug visualization
        if debug_plot:
            plt.figure(figsize=(10, 10))
            plt.scatter(contour[:, 0], contour[:, 1], alpha=0.5, label='Contour')

            # Plot lines
            for line in selected_lines:
                plt.plot([line['start'][0], line['end'][0]],
                         [line['start'][1], line['end'][1]], 'r-')

            # Plot corners
            corners_array = np.array(corners)
            plt.scatter(corners_array[:, 0], corners_array[:, 1], color='green', s=100, label='Corners')
            plt.legend()
            plt.title('Quadrilateral Detection')
            plt.show()

        return np.array(corners)

    def get_convex_hull(self,contour):
        # Compute the convex hull
        hull = cv2.convexHull(contour)

        # Convert the hull points back to a contour format if necessary
        hull_contour = hull.squeeze() if len(hull.shape) > 2 else hull

        return hull_contour
    # Example usage
    # corners = detect_quadrilateral_corners(contour, debug_plot=True)
    def get_segment_data(self, yolo_results, image):
        original_height, original_width = image.shape[:2]

        if self.min_area_cofi is not None:
            min_contour_area = original_width * original_height *  self.min_area_cofi

        result = []
        for yolo_result in yolo_results:
            masks = yolo_result.masks
            if masks is None:
                continue

            # img = np.ones((original_width, original_height, 3), dtype=np.uint8) * 255
            img = image.copy()

            for i, mask in enumerate(masks.data):
                class_name = self.class_names[int(yolo_result.boxes.cls[i].item())]

                conf = yolo_result.boxes.conf[i].item()
                conf = round(conf, 2)

                mask = self._preprocess_mask(mask,original_width,original_height)
                contour = self._find_valid_contour(mask,min_contour_area)
                if self.is_rect_contour and contour is not None:
                    approx_contour = self.get_convex_hull(contour)
                    # approx_contour = CvUtility.approximate_rect_contour(contour)
                    # approx_contour = self.detect_quadrilateral_corners2(contour,debug_plot=True)

                    if approx_contour is None:
                        continue



                    # Draw original contour (blue)



                    # If approximation succeeded, draw approximated contour (green)
                    if approx_contour is not None:
                        approx_contour = np.array(approx_contour, dtype=np.int32)
                        cv2.drawContours(img, [approx_contour], -1, (0, 255, 0), 2)

                        for point in approx_contour:
                            cv2.circle(img, (point[0], point[1]),4,(0,0,255),-1)

                    cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)

                    if approx_contour is not None:
                        contour_dim = CvUtility.get_rect_contour_dimension(approx_contour)
                        reg_image = CvUtility.rectify_rect_image(image, approx_contour, contour_dim)
                        result.append((reg_image, class_name,approx_contour,contour,conf))

                cv2.imshow("view", img)
                cv2.waitKey(0)

        return result

                    
