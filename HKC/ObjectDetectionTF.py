import os
from .GTUtilityDET import *

class ObjectDetectionTF:
    def __init__(self,models_path , obj_detection_path = None,workspace_path = None):
        self._models_path = models_path
        self._obj_detection_path = obj_detection_path
        self._workspace_path = workspace_path

    def install(self,obj_detection_path = None):
        if obj_detection_path :
          od_path = obj_detection_path
        elif self._obj_detection_path :
            od_path = self._obj_detection_path
        else :
            print('please set object detection path')
            return

        self._obj_detection_path = od_path


    def addProject(self,project_name,workspace_path = ''):
        # self.
        if not(os.path.join(workspace_path)) :


    def downloadModel(self,URL,model_name,model_ext = '.tar.gz' ,clear_file = False):
        pass

    def setConfig(self):
        self._cur_session = r''

    def train(self):
        pass