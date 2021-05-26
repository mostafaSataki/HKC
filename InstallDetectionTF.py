# from google.protobuf import text_format
# from object_detection.protos import pipeline_pb
import os
import subprocess
from .Utility import *

class InstallDetectionTF:
    @staticmethod
    def install(ob_install_path,protoc_filename = ""):
        models_path = os.path.join(ob_install_path, 'models')
        research_path = os.path.join(models_path, 'research')
        slim_path = os.path.join(research_path, 'slim')

        os.chdir(ob_install_path)
        subprocess.call(['git', 'clone', '--quiet', 'https://github.com/tensorflow/models.git'])
        os.chdir(models_path)
        if Utility.isLinux():
            subprocess.call(['apt-get', 'install', '-qq', 'protobuf-compiler', 'python-tk'])
        elif Utility.isWindows():
            print("please specify the file path.")
            return


        subprocess.call(['pip', 'install', '-q', 'Cython', 'contextlib2', 'pillow', 'lxml', 'matplotlib', 'PyDrive'])
        subprocess.call(['pip', 'install', '-q', 'pycocotools'])
        os.chdir(research_path)
        # subprocess.call(['cd', '~/models/research'])
        if Utility.isLinux():
           subprocess.call(['protoc', 'object_detection/protos/*.proto', '--python_out', '.'])
        elif Utility.isWindows():
            subprocess.call([protoc_filename,'object_detection/protos/*.proto', '--python_out', '.'])

        subprocess.call(['pip', 'install', 'pascal_voc_writer'])
        subprocess.call(['pip', 'install', 'imgaug'])
        subprocess.call(['pip', 'install', 'selenium'])
        subprocess.call(['pip', 'install', 'tf-models-official'])

        subprocess.call(['pip', 'install', 'tf_slim'])

        os.environ['PYTHONPATH'] += research_path
        os.environ['PYTHONPATH'] += slim_path


        subprocess.call(['python', os.path.join(research_path, 'object_detection/builders/model_builder_test.py')])

        os.chdir(research_path)
        # subprocess.call(['cd', '/root/models/research'])
        subprocess.call(['python', 'setup.py', 'build'])
        if Utility.isLinux():
           subprocess.call(['sudo', 'python', 'setup.py', 'install'])
        elif Utility.isWindows():
            subprocess.call(['runas', '/user:Administrator', 'python', 'setup.py', 'install'])

