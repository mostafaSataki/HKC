import os
import numpy as np
from HKC.FileUtility import *
from typing import List, Tuple
from HKC.LabelmeJson import *
from HKC.YoloSegmentUtility import *
from ultralytics import YOLO

class YoloInference:
    def __init__(self, model_filename: str, labels_file: str,draw = True,save_json = False,
                 is_rect_contour = False, min_area_cofi = None,crop_size = None):
        self.model = YOLO(model_filename)
        self.labels, self.colors = self._load_labels_and_colors(labels_file)
        self.draw = draw
        self.save_json = save_json
        self.labelme_json  = LabelmeJson()
        self.crop_size = crop_size
        self.segment_utilty =  YoloSegmentUtility(self.labels,is_rect_contour,min_area_cofi)


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

    def _draw_contour_and_label2(self, image, results, alpha=0.5) -> np.ndarray:

        output = image.copy()

        for result in results:
            contour = result[0]
            label = result[1]
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

    def inference(self, src_image_filename, dst_image_filename,dst_json_filename):
        src_image = cv2.imread(src_image_filename, 1)
        results = self.model(src_image)
        
        results = self.segment_utilty.get_contours(results, src_image)
        dst_image = src_image.copy()
        if self.draw:
            dst_image =  self._draw_contour_and_label2(src_image, results)
        if self.save_json:
            self.labelme_json.save(results,dst_image_filename, dst_json_filename, src_image)

        cv2.imwrite(dst_image_filename, dst_image)

    def inference_dir(self, src_path: str, dst_path: str):
        src_image_filenames = FileUtility.getFolderImageFiles(src_path)
        dst_image_filenames = FileUtility.getDstFilenames2(src_image_filenames,dst_path,src_path)
        dst_json_filenames = FileUtility.changeFilesExt(dst_image_filenames,'json')
        
        for src_image_filename,dst_image_filename,dst_json_filename in tqdm(zip( src_image_filenames,dst_image_filenames,dst_json_filenames)):
            self.inference(src_image_filename, dst_image_filename,dst_json_filename)
