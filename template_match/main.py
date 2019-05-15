#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : main.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 08.05.2019
# Last Modified Date: 15.05.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>

# Python program to illustrate
# multiscaling in template matching
import cv2
import os
import numpy as np

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

    idxs = np.argsort(scores)[::-1]
    idxs_select = []
    #  print("inputs: ", locs)
    while len(idxs) > 0:
        idx_cur = idxs[-1]
        idxs_select.append(idx_cur)

        #  print(idx_cur, locs)
        loc_cur = locs[:, idx_cur]
        ratios = np.linalg.norm(locs - loc_cur.reshape(2, 1), axis=0)
        idxs_rm = np.where(ratios<=th)
        idxs_n = []
        for idx in idxs:
            if not np.any(idxs_rm == idx):
                idxs_n.append(idx)
        idxs = np.array(idxs_n)

    #  print("after nms: ", len(idxs_select))
    return locs[:,idxs_select], scores[idxs_select], idxs_select


def match_template(fpath_mainimage, fpath_template, th=0.35,
                   minscale=0.4, maxscale=2.5, scalestep=0.1, border=0):
    """
    params:
        @fpath_mainimage: file path of image to be matched
        @fpath_template: file path of template image
        @th: threshold to filter matching results.
    returns:
        @corners: [N, 4, 2], oriented bb for matched instances
    """
    print("*****************************************************")
    print("Processing ", fpath_mainimage + ", " + fpath_template)
	# load data
    img_rgb = cv2.imread(fpath_mainimage)
    if len(img_rgb.shape) == 3:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_rgb
    template = cv2.imread(fpath_template,0)

    ## work on edges, rather than raw images
    #  img_gray = cv2.Canny(img_gray, 120, 200)
    #  template = cv2.Canny(template, 120, 200)

    #  cv2.imshow("template", template)
    #  cv2.imshow("Rotated", img_gray)
    #  cv2.waitKey(0)

    w, h = template.shape[::-1] ## order: (h,w)
    print("shape vs template: ", img_rgb.shape, (h,w))

    ## find objects in different rotations and scales
    th_nms = 40
    cands = []
    for rot in np.linspace(0, 360, 120):
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, rot, 1)
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).transpose(1, 0) #[3, 4]
        corners_rot = np.matmul(M, corners).transpose(1, 0)
        w_r, h_r = np.max(corners_rot, axis=0).astype(np.int32)

        #  template_rot = cv2.warpAffine(template, M, (w,h), borderValue=(255, 255, 255))
        template_rot = cv2.warpAffine(template, M, (w_r,h_r), borderValue=border)
        for scale in np.linspace(minscale, maxscale, int((maxscale-minscale)/scalestep)):
            templ_rs = cv2.resize(
                template_rot,
                dsize = (int(template.shape[1] * scale), int(template.shape[0] * scale)),
                #  interpolation=cv2.INTER_LANCZOS4
            )
            res = cv2.matchTemplate(img_gray, templ_rs, cv2.TM_CCOEFF_NORMED)
            #  res = cv2.matchTemplate(resized, template_rot, cv2.TM_CCORR_NORMED)
            #  cv2.imshow("rs", templ_rs)
            #  cv2.imshow("src", template)
            #  cv2.imshow("res", res)
            #  cv2.waitKey(0)

            max_loc = np.array(np.where(res>=th))
            max_val = np.array(res[res>=th])
            if len(max_val)==0 :
                continue

            max_loc = np.array(max_loc)[::-1] ## [2, N], (x,y)
            corners_rs = corners_rot.reshape(1, 4, 2).repeat(max_val.shape[0], axis=0)*scale
            if max_val.shape[0] > 0:
                #  print("detected", max_loc, max_val)
                vals = [max_val, max_loc, corners_rs]
                #  print(max_val.shape, max_loc.shape, corners_rs.shape)
                assert(vals[0].shape[0] == vals[2].shape[0])
                if len(cands) == 0:
                    cands = vals
                else:
                    for ii in range(3):
                        if ii == 1:
                            cands[ii] = np.concatenate([cands[ii], vals[ii]], axis=1)
                        else:
                            cands[ii] = np.concatenate([cands[ii], vals[ii]], axis=0)

    if len(cands) == 0:
        print("No shape is detected.")
        return
    else:
        print("Detected: ", np.array(cands[2]).shape[0])
    ## filter multiple detections on the same location
    max_val, max_loc, corners = cands
    idxs = nms(max_loc, max_val, th_nms)[-1]
    print("Filtered results: ", len(idxs), np.max(max_val))
    max_loc, corners = [max_loc[:, idxs], corners[idxs]]

    corners = corners + max_loc.transpose(1,0).reshape(-1, 1, 2)
    corners = corners.astype(np.int32)
    for ii in range(len(corners)):
        pts = corners[ii].reshape(-1, 1, 2)
        cv2.polylines(img_rgb, [pts], True, (0, 255, 0))
    cv2.imshow("Image", img_rgb)
    cv2.waitKey(0)

    fname = os.path.basename(fpath_template)
    cv2.imwrite(fpath_mainimage + "_" + fname + ".png", img_rgb)
    print("Results are saved to " + fpath_mainimage + "_" + fname + ".txt")
    save_corners(fpath_mainimage + "_" + fname + ".txt", corners)
    return corners

def save_corners(fpath, corners):
    with open(fpath, "w") as f:
        for bb in corners:
            bb = np.array(bb).reshape(-1)
            f.write(" ".join([str(v) for v in bb]) + "\n")

if __name__ == "__main__":
    #  match_template("images/Input2.png", "images/Input2Ref.png", th=0.4)
#      match_template("../data/artist/tree-0-1.png",
    #                  "../data/artist/0-branch-part-0.png", th=0.94, border=255)
    #  for ii in range(1, 10):
    #      match_template("../data/artist/tree-0-1.png",
    #                      "../data/artist/0-branch-part-%d.png"%ii,
    #                     th=0.7, border=255)
#

    for ii in range(1,4):
        match_template("../data/tree/r3-000449.jpg",
                        "../data/tree/r3-000449-t%d.jpg"%ii, th=0.7,
                       minscale=0.9, maxscale=1.1, scalestep=0.1, border=255)




