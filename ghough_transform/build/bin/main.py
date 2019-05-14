#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : main.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 09.05.2019
# Last Modified Date: 14.05.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import os
import numpy as np
import shlex
from subprocess import Popen, PIPE
import cv2

def ghough_transform11(fpath_main, fpath_template):
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

    edges = cv2.Canny(img_template, 150, 200)
    area = np.sum(edges > 100)
    th = int(area*0.8*1)
    print("img.size: ", img_template.shape, ", area: ", area)
#      cmd = "./example_cpp_generalized_hough -i=" + fpath_main + " -t="\
        #  + fpath_template + " --votesThreshold=%d"%(th) +\
        #  " --posThresh=%d"%(th) + " --minDist=30" +\
        #  " --minScale=0.5 --maxScale=2.5 --scaleStep=0.5" +\
        #  " --minAngle=0 --maxAngle=360 --angleStep=3" +\
        #  " --scaleThresh=%f --angleThresh=%f"%(th_scale, th_angle) +\
        #  " --full --maxBufSize=20000"
    cmd = "./example_cpp_generalized_hough -i=" + fpath_main + " -t="\
        + fpath_template + " --votesThreshold=%d"%(th) +\
        " --minDist=30" + " --maxBufSize=20000"

    #  os.system(cmd)
    process = Popen(shlex.split(cmd), stdout=PIPE)
    (output, err) = process.communicate()
    print(output, err)
    process.wait()

    with open(fpath_main + ".txt", "r") as f:
        lines = f.readlines()
        rects = [[float(v) for v in l.strip().split(" ")] for l in lines]
    rects = np.array(rects, dtype=np.int32).reshape(-1, 4, 2)

    for ii in range(rects.shape[0]):
        pts = rects[ii].reshape(-1, 2)
        cv2.polylines(img_ref, [pts], True, (0, 0, 255))
    cv2.imshow("Image", img_ref)
    cv2.waitKey(0)

    print("#detections: ", rects.shape[0])
    fname = os.path.basename(fpath_template)
    print("Results are saved to " + fpath_main + "_" + fname + ".txt")
    save_corners(fpath_main + "_" + fname + ".txt", rects)
    return rects

def nms(locs, scores, th):
    """
    Filter locations within th.
    params:
        @locs: [2,N]
        @scores: [N]
        @th: threshold
    returns:
        @locs: [2, N]
        @scores: [N]
    """
    if len(scores) == 0:
        return [], [], []

    ## remove redudency by hashing
    print("detected: ", len(scores))
    #  hash_val = (locs[0]%128)*128+ (locs[1]%128)
    #  _, idxs_uni = np.unique(hash_val, return_index=True)
    #  locs = locs[:, idxs_uni]
    #  scores = scores[idxs_uni]
    #  print("remove redundency: ", len(scores))

    idxs = np.argsort(scores)[::-1]
    idxs_select = []
    #  print("inputs: ", locs)
    while len(idxs) > 0:
        idx_cur = idxs[-1]
        idxs_select.append(idx_cur)
        if len(idxs_select) > 1000:
            break

        #  print(idx_cur, locs)
        loc_cur = locs[:, idx_cur]
        ratios = np.linalg.norm(locs - loc_cur.reshape(2, 1), axis=0)
        idxs_rm = np.where(ratios<=th)
        idxs_n = []
        for idx in idxs:
            if not np.any(idxs_rm == idx):
                idxs_n.append(idx)
        idxs = np.array(idxs_n)

    print("after nms: ", len(idxs_select))
    return locs[:,idxs_select], scores[idxs_select], idxs_select


def ghough_transform(fpath_main, fpath_template, ratio=1.0, border=255):
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
    img_template = cv2.imread(fpath_template, 0)

    w, h = img_template.shape[::-1] ## order: (h,w)
    #  print("img.size: ", img_template.shape, ", area: ", area)

    rect_list = []
    for rot in np.arange(0.0, 360, 3.0):
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, rot, 1)
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).transpose(1, 0) #[3, 4]
        corners_rot = np.matmul(M, corners).transpose(1, 0) #[4,2]
        #  corners_rot = np.maximum(0.0, corners_rot)
        w_r, h_r = np.max(corners_rot, axis=0).astype(np.int32)
        corners_rot = corners_rot.reshape(-1)
        templ_rot = cv2.warpAffine(img_template, M, (w_r,h_r), borderValue=border)
        #  cv2.imshow("rot", templ_rot)
        #  cv2.waitKey(0)

        #  for scale in np.arange(0.5, 1.0, 0.1):
        for scale in np.arange(0.5, 3.0, 0.1):
            templ_rot_scale = cv2.resize(
                templ_rot,
                dsize = (int(templ_rot.shape[1] * scale), int(templ_rot.shape[0] * scale)),
                interpolation=cv2.INTER_LANCZOS4
            )
            fpath_templ_sr = fpath_template.replace(".png", "_sr.png").replace(".jpg", "_sr.png")

            ## set threshold
            edges = cv2.Canny(templ_rot_scale, 120, 200, 3)
            area = np.sum(edges > 0)
            th = int(area*ratio)

            cv2.imwrite(fpath_templ_sr, templ_rot_scale)
            cmd = "./example_cpp_generalized_hough -i=" + fpath_main + " -t="\
                + fpath_templ_sr + " --votesThreshold=%d"%(th) +\
                " --minDist=30" + " --maxBufSize=20000"

            process = Popen(shlex.split(cmd), stdout=PIPE)
            (output, err) = process.communicate()
            #  print(output, err)
            process.wait()

            with open(fpath_main + ".txt", "r") as f:
                lines = f.readlines()
                rects = [[float(v) for v in l.strip().split(" ")] for l in lines]
            if len(rects) > 0:
                rects = np.array(rects, dtype=np.int32).reshape(-1, 4, 2)
                #  rects_sr = rects
                rects_sr = np.min(rects, axis=1, keepdims=True) + (corners_rot*scale).reshape(1, 4, 2)
                rect_list.append(rects_sr)

    if len(rect_list) > 0:
        rects = np.concatenate(rect_list, axis=0)
        _, _, idxs = nms(np.mean(rects, axis=1).transpose(1,0), np.random.rand(rects.shape[0]), th=20)
        print("#idxs: ", idxs)
        rects = np.array([rects[ii] for ii in idxs])
        print("#detections: ", rects.shape[0])

        for ii in range(rects.shape[0]):
            pts = np.int32(rects[ii].reshape(-1, 2))
            cv2.polylines(img_ref, [pts], True, (0, 0, 255))
        fname = os.path.basename(fpath_template)
        cv2.imwrite(fpath_main + "_" + fname + "_vis.png", img_ref)
        #  cv2.imshow("Image", img_ref)
        #  cv2.waitKey(0)

        print("Results are saved to " + fpath_main + "_" + fname + ".txt")
        save_corners(fpath_main + "_" + fname + ".txt", rects)
        return rects


def save_corners(fpath, corners):
    with open(fpath, "w") as f:
        for bb in corners:
            bb = np.array(bb).reshape(-1)
            f.write(" ".join([str(v) for v in bb]) + "\n")

def test_ghough():
    #  #  ghough_transform("images/Input2.png", "images/Input1Ref.png")


    ghough_transform("../../../data/artist/tree-0-1.png",
                    "../../../data/artist/0-branch-part-0.png", 0.9, 0.0)
    for ii in range(1, 10):
        ghough_transform("../../../data/artist/tree-0-1.png",
                        "../../../data/artist/0-branch-part-%d.png"%ii, 0.5, 0.0)

    #  ghough_transform("../../../data/tree/r3-000449.jpg",
    #                  "../../../data/tree/r3-000449-t1.jpg", 2.1, 255)
    #  ghough_transform("../../../data/tree/r3-000449.jpg",
    #                  "../../../data/tree/r3-000449-t2.jpg", 1.65, 255)
    #  ghough_transform("../../../data/tree/r3-000449.jpg",
    #                  "../../../data/tree/r3-000449-t3.jpg", 2.13, 255)

    #  for ii in range(2,4):
    #      ghough_transform("../../../data/tree/r3-000449.jpg",
    #                      "../../../data/tree/r3-000449-t%d.jpg"%ii)
    #      break

if __name__ == "__main__":
    test_ghough()

