from . import Facenet
from pathlib import Path
import os
import gdown

def loadModel(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5'):

    model = Facenet.InceptionResNetV2(dimension = 512)

    #-------------------------

    model_path = r'E:\Database\data_deep\face_models\facenet'
    model_filename = os.path.join(model_path,'facenet512_weights.h5')
    if os.path.isfile(os.path.join(model_path,'facenet512_weights.h5')) != True:
        print("facenet512_weights.h5 will be downloaded...")


        gdown.download(url, model_filename, quiet=False)

    #-------------------------

    model.load_weights(model_filename)

    #-------------------------

    return model
