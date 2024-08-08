import os
import json
import cv2
from collections import OrderedDict
import math
from HKC import FileUtility
from tqdm import tqdm
from enum import Enum
from utility import ActionType



class LableMeJsonYOLO:
    def __init__(self,labels,action_type: ActionType):
        self.action_type = action_type
        self._to_seg = True
        self.labels = labels

    def convert_files(self, json_filenames, yolo_filenames):
        # self.label_id_map = self._get_label_id_map_filenames(json_filenames)
        for i in tqdm(range(len(json_filenames)), ncols=100):
            json_filename = json_filenames[i]
            yolo_filename = yolo_filenames[i]
            if self.action_type == ActionType.segmentation:
                self._convert_file_segmentation(json_filename, yolo_filename)
            elif self.action_type == ActionType.detection:
                self._convert_file_detection(json_filename, yolo_filename)
            elif self.action_type == ActionType.pose_estimation:
                self._convert_file_pose_estimation(json_filename, yolo_filename)
    def _convert_file_segmentation(self, json_filename, yolo_filename):

        with open(json_filename, 'r') as file:
            data = json.load(file)

        # Extract image dimensions
        image_width = int(data['imageWidth'])
        image_height = int(data['imageHeight'])

        # Open a new text file to write the YOLO formatted data
        with open(yolo_filename, 'w') as yolo_file:
            # Iterate over each shape in the JSON data

            for shape in data['shapes']:
                lst = []
                label = shape['label']
                if label in self.labels:
                    label_index = self.labels[label]


                # Calculate the center x, y coordinates and width, height of the bounding box
                for point in shape['points']:
                    x_coords = float(point[0])
                    y_coords = float(point[1])
                    lst.append(float(x_coords) / image_width)
                    lst.append(float(y_coords) / image_height)
                line = ' '.join(str(x) for x in lst)
                yolo_file.write(f'{label_index} {line}\n')
        yolo_file.close()

    def _convert_file_detection(self, json_filename, yolo_filename):

        with open(json_filename, 'r') as file:
            data = json.load(file)

        # Extract image dimensions
        image_width = int(data['imageWidth'])
        image_height = int(data['imageHeight'])

        # Open a new text file to write the YOLO formatted data
        with open(yolo_filename, 'w') as yolo_file:
            # Iterate over each shape in the JSON data

            for shape in data['shapes']:
                lst = []
                label = shape['label']
                if label in self.labels:
                    label_index = self.labels[label]

                # Calculate the center x, y coordinates and width, height of the bounding box
                points = shape['points']
                bbox = self.points_to_bbox(points)
                yolo_rect = self.bbox_to_yolo(bbox, image_width, image_height)
                line = "{} {} {} {}".format(yolo_rect[0], yolo_rect[1], yolo_rect[2], yolo_rect[3])
                yolo_file.write(f'{label_index} {line}\n')
        yolo_file.close()

    def _convert_file_pose_estimation(self,json_filename, text_filename):
        # Load your JSON data
        with open(json_filename, 'r') as file:
            data = json.load(file)

        # Extract image dimensions
        image_width = int(data['imageWidth'])
        image_height =int(data['imageHeight'])

        # Open a new text file to write the YOLO formatted data
        with open(text_filename, 'w') as yolo_file:
            # Iterate over each shape in the JSON data

            for shape in data['shapes']:
                lst = []
                label = shape['label']
                if label in self.labels:
                    label_index = self.labels[label]

                # Calculate the center x, y coordinates and width, height of the bounding box
                for point in shape['points']:
                    x_coords = float(point[0])
                    y_coords = float(point[1])
                    lst.append( float(x_coords) / image_width)
                    lst.append(float(y_coords) / image_height)
                line = ' '.join(str(x) for x in lst)
                yolo_file.write(f'{label_index} {line}\n')
        yolo_file.close()

    def _get_label_id_map_dir(self, json_dir):
        json_filenames = FileUtility.getFolderFiles(json_dir,['json'])
        return self._get_label_id_map_filenames(json_filenames)


    def _save_yolo_file(self, yolo_filename,  yolo_obj_list):

        with open(yolo_filename, 'w+') as f:
            for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                yolo_obj_line = ""
                for i in yolo_obj:
                    yolo_obj_line += f'{i} '
                yolo_obj_line = yolo_obj_line[:-1]
                if yolo_obj_idx != len(yolo_obj_list) - 1:
                    yolo_obj_line += '\n'
                f.write(yolo_obj_line)

    def points_to_bbox(self,points):
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]

        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)

        bbox = [(min_x, min_y), (max_x, max_y)]
        return bbox
    def bbox_to_yolo(self,bbox, image_width, image_height):


        (x1, y1), (x2, y2) = bbox

        # Calculate center coordinates
        X = (x1 + x2) / 2
        Y = (y1 + y2) / 2

        # Calculate width and height
        W = x2 - x1
        H = y2 - y1

        # Normalize values
        X_normalized = X / image_width
        Y_normalized = Y / image_height
        W_normalized = W / image_width
        H_normalized = H / image_height

        return (X_normalized, Y_normalized, W_normalized, H_normalized)

    @staticmethod
    def clear_image_data_from_jsonfile(self,json_file):
            with open(json_file, 'r') as file:
                data = json.load(file)

            # Check if imageData exists and replace it with null
            if 'imageData' in data:
                data['imageData'] = None
            else:
                # Add imageData with null if it doesn't exist
                data['imageData'] = None

            # Write the updated JSON back to the file
            with open(json_file, 'w') as file:
                json.dump(data, file, indent=4)

    @staticmethod
    def clear_image_data_from_dir(self, json_dir):
        json_files = FileUtility.getFolderFiles(json_dir,['json'])
        for json_file in tqdm(json_files):
            self.clear_image_data_from_jsonfile(json_file)

    @staticmethod
    def replace_fname_with_imagepath(json_filename, image_filename):
        if not os.path.exists(json_filename):
            return

        with open(json_filename, 'r') as file:
            data = json.load(file)

        image_fname = FileUtility.getFilename(image_filename)
        data['imagePath'] = image_fname
        data['imageData'] = None

        with open(json_filename, 'w') as file:
            json.dump(data, file, indent=4)


    @staticmethod
    def replace_fname_with_imagepath_batch(src_dir):
        image_filenames = FileUtility.getFolderImageFiles(src_dir)
        json_filenames, image_filenames = FileUtility.changeFilesExt2(image_filenames, 'json')

        for i in tqdm(range(len(json_filenames)), ncols=100):
            json_filename = json_filenames[i]
            image_filename = image_filenames[i]
            LableMeJson2YOLO.editJsonFile(json_filename, image_filename)





