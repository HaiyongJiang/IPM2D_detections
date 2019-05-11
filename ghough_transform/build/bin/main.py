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

def ghough_transform(fpath_main, fpath_template):
    """
    params:
        @fpath_main: file path to the main image
        @fpath_template: file path to the template image
        @th: threshold to filter examples
    returns:
        @rects: [B, 4, 2], a list of bounding box corners
    """
    print("*****************************************************")
    print("Processing ", fpath_main + ", " + fpath_template)
    img_ref = cv2.imread(fpath_main)
    img_template = cv2.imread(fpath_template)
    area = np.sum(img_template > 200)
    print("img.size: ", img_template.shape, ", area: ", area)
    cmd = "./example_cpp_generalized_hough -i=" + fpath_main + " -t="\
        + fpath_template + " --votesThreshold=%d"%(int(area*0.8)) +\
        " --scaleThresh=%d --angleThresh=%d"%(area*20, area*100) +\
        " --posThresh=%d"%(int(area*0.8)) +\
        " --minScale=0.2 --maxScale=2.0 --scaleStep=0.1" +\
        " --minAngle=0 --maxAngle=360 --angleStep=3" +\
        " --full --maxBufSize=500000"
    #  os.system(cmd)
    process = Popen(shlex.split(cmd), stdout=PIPE)
    (output, err) = process.communicate()
    print(output, err)
    process.wait()

    with open(fpath_main + ".txt", "r") as f:
        lines = f.readlines()
        rects = [[float(v) for v in l.strip().split(" ")] for l in lines]
    rects = np.array(rects, dtype=np.int32).reshape(-1, 4, 2)

    #  for ii in range(rects.shape[0]):
    #      pts = rects[ii].reshape(-1, 2)
    #      cv2.polylines(img_ref, [pts], True, (0, 255, 255))
    #  cv2.imshow("Image", img_ref)
    #  cv2.waitKey(0)

    print("#detections: ", rects.shape[0])
    fname = os.path.basename(fpath_template)
    print("Results are saved to " + fpath_main + "_" + fname + ".txt")
    save_corners(fpath_main + ".txt", rects)
    return rects

def save_corners(fpath, corners):
    with open(fpath, "w") as f:
        for bb in corners:
            bb = np.array(bb).reshape(-1)
            f.write(" ".join([str(v) for v in bb]) + "\n")

def test_ghough():
    ghough_transform("images/Input1.png", "images/Input3Ref.png")

if __name__ == "__main__":
    test_ghough()

