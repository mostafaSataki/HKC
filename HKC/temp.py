
import argparse
import cv2 as cv
import glob
import numpy as np
import os
import time

ALGORITHMS_TO_EVALUATE = [
    (cv.createb .createBackgroundSubtractorMOG, 'MOG', {}),
    (cv.createBackgroundSubtractorGMG, 'GMG', {}),
    (cv.createBackgroundSubtractorCNT, 'CNT', {}),
    (cv.createBackgroundSubtractorLSBP, 'LSBP-vanilla', {'nSamples': 20, 'LSBPRadius': 4, 'Tlower': 2.0, 'Tupper': 200.0, 'Tinc': 1.0, 'Tdec': 0.05, 'Rscale': 5.0, 'Rincdec': 0.05, 'LSBPthreshold': 8}),
    (cv.createBackgroundSubtractorLSBP, 'LSBP-speed', {'nSamples': 10, 'LSBPRadius': 16, 'Tlower': 2.0, 'Tupper': 32.0, 'Tinc': 1.0, 'Tdec': 0.05, 'Rscale': 10.0, 'Rincdec': 0.005, 'LSBPthreshold': 8}),
    (cv.createBackgroundSubtractorLSBP, 'LSBP-quality', {'nSamples': 20, 'LSBPRadius': 16, 'Tlower': 2.0, 'Tupper': 32.0, 'Tinc': 1.0, 'Tdec': 0.05, 'Rscale': 10.0, 'Rincdec': 0.005, 'LSBPthreshold': 8}),
    (cv.createBackgroundSubtractorLSBP, 'LSBP-camera-motion-compensation', {'mc': 1}),
    (cv.createBackgroundSubtractorGSOC, 'GSOC', {}),
    (cv.createBackgroundSubtractorGSOC, 'GSOC-camera-motion-compensation', {'mc': 1})
]

def main():
  filename = r'E:\Database\lpr\Video\Video\12-15-27.mp4'
  cap = cv.VideoCapture()
  cap.open(filename)
  while cap.isOpened():
    ret,frame = cap.read()
    cv.imshow("view",frame)

    for algo, algo_name, algo_arguments in ALGORITHMS_TO_EVALUATE:

       bgs = algo(**algo_arguments)
       mask = bgs.apply(frame)
       cv.imshow(algo_name,mask)


    ch = cv.waitKey(1000//24)
    if ch == 27:
      break


if __name__ == '__main__':
  main()
