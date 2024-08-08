import os.path

from .FileUtility import *
import json
import cv2
import numpy as np
from .CvUtility import *

class YoloAugmentDB:
    def __init__(self):
        pass

    def augment_rotate(self,src_path:str,dst_path:str,generate_count:int,start_angle:float,end_angle:float):
        src_image_filenames = FileUtility.getFolderImageFiles(src_path)
        src_json_filenames = FileUtility.changeFilesExt(src_image_filenames,'json')

        FileUtility.copyFiles2DstPath(src_image_filenames,dst_path)
        FileUtility.copyFiles2DstPath(src_json_filenames, dst_path)

        random_numbers = self._generate_random_numbers(len(src_image_filenames), generate_count)
        random_angles = self._generate_random_ranges(start_angle,end_angle,generate_count)

        for i, index in tqdm(enumerate(random_numbers), total=len(random_numbers)):
            src_image_filename = src_image_filenames[index]
            dst_image_filename = FileUtility.changeFilenamePostfix(src_image_filename,'_'+str(i))
            dst_image_filename = FileUtility.getDstFilename2(dst_image_filename,dst_path,src_path)

            src_json_filename = src_json_filenames[index]
            dst_json_filename = FileUtility.getDstFilename2(src_json_filename,dst_path,src_path)
            dst_json_filename = FileUtility.changeFilenamePostfix(dst_json_filename, '_' + str(i))

            filename, rect, (image_width, image_height) = self._read_json(src_json_filename)
            filename = FileUtility.getFilename(dst_image_filename)
            image = cv2.imread(src_image_filename)
            rotated_image, rotated_rect_bbox = self._rotate_image_and_rect(image,rect,random_angles[i])

            h,w,_ = rotated_image.shape
            cv2.imwrite(dst_image_filename,rotated_image)
            self._update_json(src_json_filename,dst_json_filename,rotated_rect_bbox,(w,h),filename)





    def _generate_random_numbers(self,file_count, generate_count):
        random_numbers = []

        # Generate random numbers
        for _ in range(generate_count):
            random_numbers.append(random.randint(0, file_count-1))

        return random_numbers

    def _generate_random_ranges(self,a:float,b:float, generate_count:int):
        random_numbers = []

        # Generate random numbers
        for _ in range(generate_count):
            random_numbers.append(random.uniform(a, b))

        return random_numbers



    def _rotate_image_and_rect(self, image, rect, angle):
        # Convert angle to radians
        angle_rad = angle * np.pi / 180

        # Get the dimensions of the image
        rows, cols = image.shape[:2]

        # Get the center of the image
        center = (cols / 2, rows / 2)

        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate the image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        # Get the points of the ROI
        x1, y1, x2, y2 = rect
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        # Rotate the ROI points
        rotated_pts = cv2.transform(pts.reshape((-1, 1, 2)), rotation_matrix).squeeze().reshape((-1, 2))

        # Get the bounding box of the rotated ROI
        rotated_rect_bbox = cv2.boundingRect(rotated_pts)


        return rotated_image,CvUtility.cvrect2Rect(rotated_rect_bbox)


    def _read_json(self,json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        filename = data['imagePath']
        image_height = data['imageHeight']
        image_width = data['imageWidth']

        if data['shapes']:
            shape = data['shapes'][0]
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            rect = (x1, y1, x2, y2)
        else:
            rect = None

        return filename, rect, (image_width, image_height)

    def _update_json(self, src_filename, dst_filename, rect, image_size, new_filename):
        with open(src_filename, 'r') as f:
            data = json.load(f)

        if rect:
            x1, y1, x2, y2 = rect
            data['shapes'][0]['points'] = [[x1, y1], [x2, y2]]

        data['imageWidth'], data['imageHeight'] = image_size
        data['imagePath'] = new_filename

        with open(dst_filename, 'w') as f:
            json.dump(data, f, indent=4)

    def rotate_point(self,point, angle, width, height):
        x, y = point
        if angle == 90:
            return [height - y, x]
        elif angle == 180:
            return [width - x, height - y]
        elif angle == 270:
            return [y, width - x]
        else:
            raise ValueError("Angle must be 90, 180, or 270 degrees")

    def rotate_coordinates(self,points, angle, width, height):
        return [self.rotate_point(point, angle, width, height) for point in points]

    def rotate_image_and_lableme_json(self, source_image_path, source_json_path, dst_folder, angle):
        # Load image
        image = cv2.imread(source_image_path)
        (h, w) = image.shape[:2]

        # Rotate image
        if angle == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError("Angle must be 90, 180, or 270 degrees")

        # Load JSON
        with open(source_json_path, 'r') as f:
            data = json.load(f)

        # Rotate points
        rotated_shapes = []
        for shape in data['shapes']:
            rotated_points = self.rotate_coordinates(shape['points'], angle, w, h)
            rotated_shapes.append({
                'label': shape['label'],
                'points': rotated_points,
                'group_id': shape.get('group_id'),
                'description': shape.get('description', ''),
                'shape_type': shape['shape_type'],
                'flags': shape.get('flags', {})
            })

        # Save rotated image
        image_filename = os.path.basename(source_image_path)
        rotated_image_path = os.path.join(dst_folder, image_filename.replace('.jpg', f'_{angle}.jpg'))
        cv2.imwrite(rotated_image_path, rotated_image)

        # Update JSON data
        new_data = {
            "version": data["version"],
            "flags": data["flags"],
            "shapes": rotated_shapes,
            "imagePath": os.path.basename(rotated_image_path),
            "imageData": data.get("imageData", ""),
            "imageHeight": rotated_image.shape[0],
            "imageWidth": rotated_image.shape[1],
            'imageData':None
        }

        # Save JSON file
        json_filename = os.path.basename(source_json_path)
        rotated_json_path = os.path.join(dst_folder, json_filename.replace('.json', f'_{angle}.json'))
        with open(rotated_json_path, 'w') as f:
            json.dump(new_data, f, indent=4)

    def rotate_segmentation(self,src_path,dst_path,angles):
        ext = FileUtility.compare_extension_counts(src_path,'json','xml')

        if ext != 'json':
            return


        src_image_filenames = FileUtility.getFolderImageFiles(src_path)
        src_json_filenames = FileUtility.changeFilesExt(src_image_filenames,ext)




        for i in tqdm(range(len(src_image_filenames))):
            FileUtility.copy2Path(src_image_filenames[i],dst_path)
            FileUtility.copy2Path(src_json_filenames[i],dst_path)

            for angle in angles:
                src_image_filename = src_image_filenames[i]
                src_json_filename = src_json_filenames[i]
                if not os.path.exists(src_json_filename):
                    continue
                self.rotate_image_and_lableme_json(src_image_filename,src_json_filename,dst_path,angle)


def augment_db_rotate_proc():
    src_path = r'D:\database\FridgeEye\json_db'
    dst_path = r'D:\database\FridgeEye\json_db2'
    augmntor = AugmentDB()
    augmntor.augment_rotate(src_path,dst_path,18*3,-10,10)


def test_rotate_segmentation():
    src_path = r'D:\database\snapp\train_detection\seri2\total'
    dst_path = r'D:\database\snapp\train_detection\seri2\augment'
    angles =[90,180,270]
    augmntor = AugmentDB()
    augmntor.rotate_segmentation(src_path,dst_path,angles)

if __name__ == '__main__':
    # augment_db_rotate_proc()
    test_rotate_segmentation()




