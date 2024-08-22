import numpy as np
from HKC.FileUtility import *
import json

class LabelmeJson:
    def __init__(self):
        pass
    def save(self,results,image_filename, json_filename,image):
        image_height, image_width = image.shape[:2]
        shapes = []
        for result in results:
            contour = result[0]
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

  
