import os

import cv2
import numpy as np
from HKC.FileUtility import *
from typing import List, Tuple
from HKC.LabelmeJson import *
from HKC.YoloSegmentationUtility import *
from ultralytics import YOLO

class YoloSegmentationInference:
    def __init__(self, model_filename: str, labels_file: str,draw = True,save_json = False,
                 is_rect_contour = False, min_area_cofi = None,crop_dir = None, crop_size = None):


        self.model = YOLO(model_filename)
        self.labels, self.colors = self._load_labels_and_colors(labels_file)
        self.draw = draw
        self.save_json = save_json
        self.labelme_json  = LabelmeJson()
        self.crop_size = crop_size
        self.crop_dir = crop_dir
        self.segment_utilty =  YoloSegmentUtility(self.labels,is_rect_contour,min_area_cofi)
        self.crop_counter =0

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

    def _draw_contour_and_label2_segmentation(self, image, results, alpha=0.5) -> np.ndarray:

        output = image.copy()

        for result in results:
            contour = result[3]
            label = result[1]
            conf = result[4]
            label_color = self._get_label_color(label)

            overlay = image.copy()
            # Create a new color with the specified green intensity
            # label_color2 = (b, 255, r)
            cv2.drawContours(overlay, [contour], 0, (0,255,0), -1)

            # Blend the original image and the overlay
            output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box
            # cv2.rectangle(output, (x, y), (x + w, y + h), label_color, 2)
            # Draw label
            cv2.putText(output,str(conf), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
            #
            # b, g, r = label_color





        return output
    
    def _draw_bbox_and_label2_detection(self, image, results, alpha=0.5) -> np.ndarray:

        output = image.copy()

        for result in results:
            contour = result[1]
            label = result[2]
            label_color = self._get_label_color(label)

            overlay = image.copy()
            cv2.drawContours(overlay, [contour], 0, label_color, -1)

            # Blend the original image and the overlay
            output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return output
    
    def get_crop_filename(self,filename, classname):
        self.crop_counter += 1
        tokens = FileUtility.getFileTokens(filename)
        class_dir =os.path.join(self.crop_dir,classname)
        return os.path.join(class_dir ,tokens[1]+"_"+str(self.crop_counter)+tokens[2])
    
    def save_crop_results(self,filename, segment_data):
        
        for i, (image, label, contour) in enumerate(segment_data):
            crop_filename = self.get_crop_filename(filename,label)
            cv2.imwrite(crop_filename,image)
            
            



    def inference_segmentation(self, src_image_filename, dst_image_filename,dst_json_filename):
        src_image = cv2.imread(src_image_filename, 1)
        segment_data = self.model(src_image, verbose=False)
        
        segment_data = self.segment_utilty.get_segmentation_data(segment_data, src_image)
        dst_image = src_image.copy()

        if self.draw:
            dst_image =  self._draw_contour_and_label2_segmentation(src_image, segment_data)

        if self.save_json:
            self.labelme_json.save_segmentation(segment_data, dst_image_filename, dst_json_filename, src_image)

        if self.crop_dir is not None:
            self.save_crop_results(dst_image_filename, segment_data)

        if self.draw or self.save_json:
            cv2.imwrite(dst_image_filename, dst_image)



    def inference_segmentation_dir(self, src_path: str, dst_path: str):
        src_image_filenames = FileUtility.getFolderImageFiles(src_path)
        dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames,dst_path,src_path)
        dst_json_filenames = FileUtility.changeFilesExt(dst_image_filenames,'json')


        for src_image_filename,dst_image_filename,dst_json_filename in tqdm(zip( src_image_filenames,dst_image_filenames,dst_json_filenames)):
            self.inference_segmentation(src_image_filename, dst_image_filename,dst_json_filename)


    def inference_detection(self, src_image_filename, dst_image_filename, dst_json_filename):
        src_image = cv2.imread(src_image_filename, 1)
        detection_data = self.model(src_image, verbose=False)
    
        detection_data = self.segment_utilty.get_detection_data(detection_data, src_image)
        dst_image = src_image.copy()
    
        if self.draw:
            dst_image = self._draw_bbox_and_label2_detection(src_image, detection_data)
    
        if self.save_json:
            self.labelme_json.save_detection(detection_data, dst_image_filename, dst_json_filename, src_image)
    
        if self.crop_dir is not None:
            self.save_crop_results(dst_image_filename, detection_data)
    
        if self.draw or self.save_json:
            cv2.imwrite(dst_image_filename, dst_image)
    
    
    def inference_detection_dir(self, src_path: str, dst_path: str):
            src_image_filenames = FileUtility.getFolderImageFiles(src_path)
            dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames,dst_path,src_path,True)
            dst_json_filenames = FileUtility.changeFilesExt(dst_image_filenames,'json')
    
    
            for src_image_filename,dst_image_filename,dst_json_filename in tqdm(zip( src_image_filenames,dst_image_filenames,dst_json_filenames)):
                self.inference_detection(src_image_filename, dst_image_filename,dst_json_filename)


