# Helper function for extracting features from pre-trained models
import torch
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
from ..FaceUtility import *
from .model_irse import *

class FaceFeatureExtractIR:
    def __init__(self,backbone,model_filename,input_size =(112,112)):
        self._loaded = False
        if FaceFeatureType.IR50:
            self._model_root = model_filename
            # self._device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._device = "cpu"
            self._tta = False
            if not os.path.exists(model_filename):
                return

            if FileUtility.getFileExt(self._model_root) == "pth":
                self._backbone = IR_50(input_size)

                # load backbone from a checkpoint
                print("Loading Backbone Checkpoint '{}'".format(self._model_root))

                self._backbone.load_state_dict(torch.load(self._model_root))

            elif FileUtility.getFileExt(self._model_root) == "pt":
                self._backbone = torch.jit.load(self._model_root)

        self._backbone.to(self._device)

        # extract features
        self._backbone.eval()  # set to evaluation mode

        self._loaded = True
    def trace(self,filename):
        example = torch.rand(1, 3, 112, 112)
        traced_script_module = torch.jit.trace(self._backbone, example)
        traced_script_module.save(filename)
        print("Traced model saved to '{}'".format(filename))

    def extract(self,img):
        # if img == None:
        #     return None
        if not self._loaded :
            return None

        # resize image to [128, 128]
        resized = cv2.resize(img, (128, 128))

        # center crop image
        a = int((128 - 112) / 2)  # x start
        b = int((128 - 112) / 2 + 112)  # x end
        c = int((128 - 112) / 2)  # y start
        d = int((128 - 112) / 2 + 112)  # y end
        chip_img = resized[a:b, c:d]  # center crop the image

        ccropped = chip_img[..., ::-1]  # BGR to RGB

        # flip image horizontally
        flipped = cv2.flip(ccropped, 1)

        # load numpy to tensor
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype=np.float32)

        ccropped = (ccropped - 127.5) / 128.0
        ccropped = torch.from_numpy(ccropped)
        if self._tta:
            flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
            flipped = np.reshape(flipped, [1, 3, 112, 112])
            flipped = np.array(flipped, dtype=np.float32)
            flipped = (flipped - 127.5) / 128.0
            flipped = torch.from_numpy(flipped)

        with torch.no_grad():
            if self._tta:
                emb_batch = self._backbone(ccropped.to(self._device)).cpu() + self._backbone(flipped.to(self._device)).cpu()
                features = l2_norm(emb_batch)
            else:
                features = l2_norm(self._backbone(ccropped.to(self._device)).cpu())

        #     np.save("features.npy", features)
        #     features = np.load("features.npy")

        return features,chip_img




def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(img_root, backbone, model_root, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True):
    # pre-requisites
    assert(os.path.exists(img_root))
    print('Testing Data Root:', img_root)
    assert (os.path.exists(model_root))
    print('Backbone Model Root:', model_root)

    # load image
    img = cv2.imread(img_root)

    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))

    # center crop image
    a=int((128-112)/2) # x start
    b=int((128-112)/2+112) # x end
    c=int((128-112)/2) # y start
    d=int((128-112)/2+112) # y end
    ccropped = resized[a:b, c:d] # center crop the image
    ccropped = ccropped[...,::-1] # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype = np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)


    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval() # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())
            
#     np.save("features.npy", features) 
#     features = np.load("features.npy")

    return features
