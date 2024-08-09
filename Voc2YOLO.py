import os
import json
import cv2
from collections import OrderedDict
import math
from HKC import FileUtility
from tqdm import tqdm
from enum import Enum
from utility import ActionType
import xml.etree.ElementTree as ET
import os


class Voc2YOLO:
    def __init__(self,labels,action_type: ActionType):
        self.action_type = action_type
        self.labels = labels


    def convert_files(self, json_filenames, yolo_filenames):
        # self.label_id_map = self._get_label_id_map_filenames(json_filenames)
        for i in tqdm(range(len(json_filenames)), ncols=100):
            gt_filename = json_filenames[i]
            yolo_filename = yolo_filenames[i]
            if self.action_type == ActionType.detection:
                self._convert_file_detection(gt_filename, yolo_filename)
            # elif self.action_type == ActionType.segmentation:
            #     self._convert_file_segmentation(gt_filename, yolo_filename)
            # elif self.action_type == ActionType.pose_estimation:
            #     self._convert_file_pose_estimation(gt_filename, yolo_filename)
                
    def _convert_file_detection(self, xml_file, yolo_filename):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        yolo_lines = []
        for obj in root.iter('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = (int(bbox.find(x).text) for x in ['xmin', 'ymin', 'xmax', 'ymax'])
            x_center, y_center, w, h = (
                (xmax + xmin) / 2 / width,
                (ymax + ymin) / 2 / height,
                (xmax - xmin) / width,
                (ymax - ymin) / height,
            )
            yolo_lines.append(f"{name} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")


        with open(yolo_filename, 'w') as f:
            f.write('\n'.join(yolo_lines))



