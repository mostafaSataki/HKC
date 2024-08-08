# from ultralytics import YOLO
# import cv2
# import numpy as np
# from HKC.FileUtility import *
#
#
# class YoloInference:
#     def __init__(self, model_filename):
#         self.model = YOLO(model_filename)
#
#     def draw_contour(self, image, alpha=0.5):
#         results = self.model(image)
#         output = image.copy()
#         for result in results:
#             masks = result.masks
#             if masks is not None:
#                 mask = masks.data.cpu().numpy()[0]
#                 mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
#                 mask = (mask > 0.5).astype(np.uint8) * 255
#
#                 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#                 if contours:
#                     max_contour = max(contours, key=cv2.contourArea)
#                     overlay = image.copy()
#                     cv2.drawContours(overlay, [max_contour], 0, (0, 255, 0), -1)
#
#                     # Blend the original image and the overlay
#                     output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
#
#
#         return output
#
#
# def process_images(model_filename, src_path,dst_path):
#     src_filenames = FileUtility.getFolderImageFiles(src_path)
#     dst_filenames = FileUtility.getDstFilenames2(src_filenames,dst_path,src_path)
#     yolo_inference = YoloInference(model_filename)
#
#     for i in tqdm(range(len(src_filenames))):
#         src_filename = src_filenames[i]
#         dst_filename = dst_filenames[i]
#         src_image = cv2.imread(src_filename,1)
#         if src_image is None:
#             print(f"Could not read image: {src_filename}")
#             continue
#
#
#         dst_image = yolo_inference.draw_contour(src_image)
#         cv2.imwrite(dst_filename,dst_image)
#
#
#
#
# if __name__ == '__main__':
#     # model_filename =  r'D:\database\snapp\dl2_augment_yolo\runs\segment\train2\weights\best.pt'
#     # model_filename = r'D:\database\snapp\dl3_augment_yolo\runs\segment\train2\weights\best.pt'
#     model_filename = r'D:\database\snapp\main_augment_yolo\runs\segment\train\weights\best.pt'
#     src_path = r'D:\database\snapp\DL'
#     dst_path = r'D:\database\snapp\res'
#
#     process_images(model_filename, src_path,dst_path)



import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

class YoloInference:
    def __init__(self, model_filename: str, labels_file: str):
        self.model = YOLO(model_filename)
        self.labels, self.colors = self.load_labels_and_colors(labels_file)

    @staticmethod
    def load_labels_and_colors(labels_file: str) -> Tuple[List[str], List[Tuple[int, int, int]]]:
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in labels]
        return labels, colors

    def draw_contour_and_label(self, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        results = self.model(image)
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

def process_images(model_filename: str, labels_file: str, src_path: str, dst_path: str):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)

    yolo_inference = YoloInference(model_filename, labels_file)

    for src_file in tqdm(list(src_path.glob('*.jpg')) + list(src_path.glob('*.png'))):
        dst_file = dst_path / src_file.name

        src_image = cv2.imread(str(src_file))
        if src_image is None:
            print(f"Could not read image: {src_file}")
            continue

        dst_image = yolo_inference.draw_contour_and_label(src_image)
        cv2.imwrite(str(dst_file), dst_image)

if __name__ == '__main__':
    model_filename = r'D:\database\snapp\main_augment_yolo\runs\segment\train\weights\best.pt'
    labels_file = r'D:\database\snapp\main_augment_yolo\classes.txt'  # Add the path to your labels file
    src_path = r'D:\database\snapp\DL'
    dst_path = r'D:\database\snapp\res'

    process_images(model_filename, labels_file, src_path, dst_path)