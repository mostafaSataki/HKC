import os
import numpy as np
from HKC.FileUtility import *
from typing import List, Tuple

class YoloInference:
    def __init__(self, model_filename: str, labels_file: str):
        self.model = YOLO(model_filename)
        self.labels, self.colors = self._load_labels_and_colors(labels_file)


    def _load_labels_and_colors(labels_file: str) -> Tuple[List[str], List[Tuple[int, int, int]]]:
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in labels]
        return labels, colors

    def _draw_contour_and_label(self, image,result,  alpha =0.5) -> np.ndarray:

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

    def inference(self, src_filename,dst_filename):
        image = cv2.imread(src_filename,1)
        results = self.model(image)
        dst_image =  self._draw_contour_and_label(image, results)
        cv2.imwrite(dst_filename, dst_image)

    def inference_dir(self, src_path: str, dst_path: str):
        src_image_filenames = FileUtility.getFolderImageFiles(src_path)
        dst_image_filenames = FileUtility.getFolderImageFiles(src_image_filenames,dst_path,src_path)
        
        for src_image_filename,dst_image_filename in tqdm(zip( src_image_filenames,dst_image_filenames)):
            self.inference(src_image_filename, dst_image_filename)
