import os

import cv2
import numpy as np
from HKC.FileUtility import *
from typing import List, Tuple
from HKC.LabelmeJson import *
from HKC.YoloDetectionUtility import  *
from ultralytics import YOLO

class YoloDetectionInference:
    def __init__(self, model_filename: str, labels_file: str,draw = True,save_json = False,
                  min_area_cofi = None,crop_dir = None, crop_size = None,border_size = 0,border_color = (255,255,255)):


        self.model = YOLO(model_filename)
        self.labels, self.colors = self._load_labels_and_colors(labels_file)
        self.draw = draw
        self.save_json = save_json
        self.labelme_json  = LabelmeJson()
        self.crop_size = crop_size
        self.crop_dir = crop_dir
        self.detection_utilty =  YoloDetectionUtility(self.labels, min_area_cofi)
        self.crop_counter =0
        self.border_size = border_size
        self.border_color = border_color

        if crop_dir is not None and os.path.exists(crop_dir):
            FileUtility.createSubfolders(self.crop_dir,self.labels)


    def _load_labels_and_colors(self,labels_file: str) -> Tuple[List[str], List[Tuple[int, int, int]]]:
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in labels]
        return labels, colors

    def _get_label_index(self,label: str) -> int:
        if label in self.labels:
            return self.labels.index(label)
        return None

    def _get_label_color(self,label: str) -> Tuple[int, int, int]:
        index = self._get_label_index(label)
        if index is not None:
            return self.colors[index]
        else:return  None

    def _draw_contour_and_label(self, image,results,  alpha =0.5) -> np.ndarray:

        output = image.copy()

        for result in results:
            boxes = result.boxes
            masks = result.masks

            if masks is not None:
                for i, (mask, box) in enumerate(zip(masks.data, boxes.data)):
                    mask = mask.cpu().numpy()
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8) * 255

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        max_contour = max(contours, key=cv2.contourArea)
                        overlay = image.copy()
                        color = self.colors[int(box[5])]  # Use class index to get color
                        cv2.drawContours(overlay, [max_contour], 0, color, -1)

                        # Blend the original image and the overlay
                        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                        # Draw label
                        label = self.labels[int(box[5])]
                        cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return output


    
    def _draw_bbox_and_label2(self, image, results, alpha=0.5) -> np.ndarray:
        output = image.copy()

        for result in results:
            # Ensure coordinates are integers
            x1, y1, x2, y2 = [int(coord) for coord in result['bbox']]
            conf = result['confidence']
            class_name = result['class_name']

            # Get color for the class
            color = self._get_label_color(class_name)

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Create label text with class name and confidence
            label = f"{class_name} {conf:.2f}"

            # Draw label background for better readability
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(output, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)

            # Draw label text
            cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return output
    
    def get_crop_filename(self,filename, classname):
        self.crop_counter += 1
        tokens = FileUtility.getFileTokens(filename)
        class_dir =os.path.join(self.crop_dir,classname)
        return os.path.join(class_dir ,tokens[1]+"_"+str(self.crop_counter)+tokens[2])
    
    def save_crop_results(self,image,filename,data):

        for i, data_item in enumerate(data):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in data_item['bbox']]

            # Extract confidence and class name
            conf = data_item['confidence']
            class_name = data_item['class_name']

            # Crop the image
            cropped_image = image[y1:y2, x1:x2]

            # Generate crop filename
            crop_filename = self.get_crop_filename(filename, class_name)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(crop_filename), exist_ok=True)

            # Save the cropped image
            cv2.imwrite(crop_filename, cropped_image)
            

    def inference_detection(self, src_image_filename, dst_image_filename, dst_json_filename):
        src_image = cv2.imread(src_image_filename, 1)

        if self.border_size:
            src_image = cv2.copyMakeBorder(
                src_image,
                top=self.border_size,
                bottom=self.border_size,
                left=self.border_size,
                right=self.border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=self.border_color
            )

        detection_data = self.model(src_image, verbose=False)
    
        detection_data = self.detection_utilty.get_detection_data(detection_data, src_image)
        dst_image = src_image.copy()



    
        if self.draw:
            dst_image = self._draw_bbox_and_label2(src_image, detection_data)
    
        if self.save_json:
            self.labelme_json.save_detection(detection_data, dst_image_filename, dst_json_filename, src_image)
    
        if self.crop_dir is not None:
            self.save_crop_results(dst_image, dst_image_filename, detection_data)
    
        if self.draw or self.save_json:
            cv2.imwrite(dst_image_filename, dst_image)

    def draw_negative2(self, src_image_filename, dst_image_filename, dst_json_filename,negative_path):
        src_image = cv2.imread(src_image_filename, 1)
        detection_data = self.model(src_image, verbose=False)

        detection_data = self.detection_utilty.get_detection_data(detection_data, src_image)
        dst_image = src_image.copy()

        if self.draw:
            dst_image = self._draw_bbox_and_label2(src_image, detection_data)

        if self.save_json:
            self.labelme_json.save_detection(detection_data, dst_image_filename, dst_json_filename, src_image)

        if self.crop_dir is not None:
            self.save_crop_results(dst_image_filename, detection_data)

        if self.draw or self.save_json:
            cv2.imwrite(dst_image_filename, dst_image)

    def inference_detection_dir(self, src_path: str, dst_path: str):
        src_image_filenames = FileUtility.getFolderImageFiles(src_path)
        dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames,dst_path,src_path)
        dst_json_filenames = FileUtility.changeFilesExt(dst_image_filenames,'json')





        for src_image_filename,dst_image_filename,dst_json_filename in tqdm(zip( src_image_filenames,dst_image_filenames,dst_json_filenames)):
            self.inference_detection(src_image_filename, dst_image_filename,dst_json_filename)


