# coding=utf-8
from time import time
from features import loadFeatures, calAndSaveFeatures
from setting import TEST
from model import calAndSaveModel, loadModel, getModel
from numpy import random
from sklearn.metrics import accuracy_score
from adaboost import Adaboost
from detector import Detector
from matplotlib import image
from PIL import Image
import os
import sys

def main():
    starttime = time()

    if len(sys.argv) <= 1:
        print("Missing arguments")
    else:
        fileName = sys.argv[1]
        arguments = {"show" : True, "save" : False, "saveInfo" : False}
        for i in range(2, len(sys.argv)):
            if (sys.argv[i].split("=")[0])[2:] in arguments.keys():
                arguments[(sys.argv[i].split("=")[0])[2:]] = \
                    (sys.argv[i].split("=")[1] == str(True))

        print("loading model...")
        clf = loadModel()
        detector = Detector(clf)

        print("detecting...")

        detector.detectFace(fileName, _show=arguments['show'], _save=arguments['save'],
                            _saveInfo=arguments['saveInfo'])

        endtime = time()
        print("cost: " + str(endtime - starttime))


if __name__ == "__main__":
    main()