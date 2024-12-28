import numpy as np
from HKC.FileUtility import *
import json

class LabelmeJson:
    def __init__(self):
        pass
    def save_segmentation(self,results,image_filename, json_filename,image):
        image_height, image_width = image.shape[:2]
        shapes = []
        for result in results:
            contour = result[2]
            label = result[1]

            contour = np.array(contour).squeeze().tolist()
            shapes.append({
                "label": label,
                "points": contour,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {}
            })

        data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": FileUtility.getFilename(image_filename),
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width
        }

        with open(json_filename, 'w') as json_file:
            json.dump(data, json_file, indent=2)
            
    def save_detection(self, results, image_filename, json_filename, image):
        image_height, image_width = image.shape[:2]
        shapes = []
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            conf = result['confidence']
            if conf < 0.65:
                continue
            class_name = result['class_name']



            # Convert bounding box to two points (top-left and bottom-right corners)
            bbox_points = [
                [x1, y1],  # Top-left corner
                [x2, y2]   # Bottom-right corner
            ]
            shapes.append({
                "label": class_name,
                "points": bbox_points,
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

        data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": FileUtility.getFilename(image_filename),
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width
        }


        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)
