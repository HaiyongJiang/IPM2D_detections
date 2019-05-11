#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : main.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 09.05.2019
# Last Modified Date: 11.05.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import os
import numpy as np
import shlex
from subprocess import Popen, PIPE
import cv2

def ghough_transform(fpath_main, fpath_template, th):
    """
    params:
        @fpath_main: file path to the main image
        @fpath_template: file path to the template image
        @th: threshold to filter examples
    returns:
        @rects: [B, 4, 2], a list of bounding box corners
    """
    print("Processing " + fpath_main)
    img = cv2.imread(fpath_template)
    area = np.sum(img > 128)
    print("img.size: ", img.shape, ", area: ", area)
    cmd = "./example_cpp_generalized_hough -i=" + fpath_main + " -t="\
        + fpath_template + " --votesThreshold=%d"%(area//5) +\
        " --minScale=0.2 --maxScale=2.0 --scaleStep=0.1" +\
        " --minAngle=0.0 --maxAngle=180 --angleStep=3"
    #  os.system(cmd)
    process = Popen(shlex.split(cmd), stdout=PIPE)
    process.communicate()
    exit_code = process.wait()

    with open(fpath_main + ".txt", "r") as f:
        lines = f.readlines()
        rects = [[float(v) for v in l.strip().split(" ")] for l in lines]
    rects = np.array(rects, dtype=np.int32).reshape(-1, 4, 2)
    print("#detections: ", rects.shape[0])
    return rects

def test_ghough():
    ghough_transform("images/Input1.png", "images/Input1Ref.png", th=200)

if __name__ == "__main__":
    test_ghough()

