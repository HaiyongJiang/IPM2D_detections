#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : main.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 08.05.2019
# Last Modified Date: 11.05.2019
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

    idxs = np.argsort(scores)
    idxs_select = []
    #  print("inputs: ", locs)
    while len(idxs) > 0:
        idx_cur = idxs[-1]
        idxs_select.append(idx_cur)

        loc_cur = locs[:, idx_cur]
        ratios = np.linalg.norm(locs - loc_cur.reshape(2, 1), axis=0)

        idxs = np.delete(idxs, np.concatenate(([idx_cur], np.where(ratios<=th)[0])))

    #  print("selected: ", locs[:, idxs_select])
    return locs[:,idxs_select], scores[idxs_select], idxs_select


def match_template(fpath_mainimage, fpath_template, th=0.35, minscale=0.5, maxscale=2.0, scalestep=0.1):
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
    img_gray = cv2.Canny(img_gray, 150, 200)
    template = cv2.Canny(template, 150, 200)

    #  cv2.imshow("template", template)
    #  cv2.imshow("Rotated", img_gray)
    #  cv2.waitKey(0)

    w, h = template.shape[::-1] ## order: (h,w)
    print("shape vs template: ", img_rgb.shape, (h,w))

    ## find objects in different rotations and scales
    th_nms = np.minimum(w,h)*0.7
    cands = []
    for rot in np.linspace(0, 360, 120):
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, rot, 1)
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).transpose(1, 0) #[3, 4]
        corners_rot = np.matmul(M, corners).transpose(1, 0)
        #  corners_rot = np.maximum(0.0, corners_rot)
        w_r, h_r = np.max(corners_rot, axis=0).astype(np.int32)

        #  template_rot = cv2.warpAffine(template, M, (w,h), borderValue=(255, 255, 255))
        template_rot = cv2.warpAffine(template, M, (w_r,h_r), borderValue=(0, 0, 0))
        for scale in np.linspace(minscale, maxscale, int((maxscale-minscale)/scalestep))[::-1]:
            resized = cv2.resize(
                img_gray,
                dsize = (int(img_gray.shape[1] * scale), int(img_gray.shape[0] * scale)),
                interpolation=cv2.INTER_LANCZOS4
            )

            #  cv2.imshow("template", template_rot)
            #  cv2.imshow("Rotated", resized)
            #  cv2.waitKey(0)

            if resized.shape[0] < h_r or resized.shape[1] < w_r:
                break
            s_r = img_gray.shape[0] / resized.shape[0] #[h, w]
            res = cv2.matchTemplate(resized, template_rot, cv2.TM_CCOEFF_NORMED)
            #  res = cv2.matchTemplate(resized, template_rot, cv2.TM_CCORR_NORMED)

            max_loc = np.array(np.where(res>=th)[::-1])
            max_val = np.array(res[res>=th])
            max_loc, max_val, _ = nms(max_loc, max_val, th_nms)
            if len(max_val)==0 :
                continue

            hash_val = (max_loc[0]*s_r).astype(np.int32)*1024 + (max_loc[1]*s_r).astype(np.int32)
            max_loc = (np.array(max_loc) * s_r).transpose(1,0).astype(np.int32)
            #  idxs = nms(max_loc.transpose(1,0), max_val, th_nms)[-1]
            idxs = np.argsort(max_val)[::-1][:50]
            if max_val.shape[0] > 0:
                #  print("detected", max_loc, max_val)
                vals = [hash_val, max_val, max_loc, np.ones(hash_val.shape)*s_r, np.ones(hash_val.shape)*rot,
                        corners_rot.reshape(1, 4, 2)]
                for ii in range(5):
                    vals[ii] = vals[ii][idxs]

                if len(cands) == 0:
                    cands = vals
                else:
                    for ii in range(6):
                        #  print(ii, cands[ii].shape, vals[ii].shape)
                        cands[ii] = np.concatenate([cands[ii], vals[ii]], axis=0)

    if len(cands) == 0:
        print("No shape is detected.")
        return
    else:
        print("Detected: ", np.array(cands[2]).shape[0])
    ## filter multiple detections on the same location
    hash_val, max_val, max_loc, s_r, rot, corners = cands
    max_loc = max_loc.transpose(1,0)
    idxs = nms(max_loc, max_val, th_nms)[-1]
    cands_filter = [max_loc[:, idxs], s_r[idxs], rot[idxs], corners[idxs]]

    ## object patches (I did not add rotations)
    max_loc, s_r, rot, corners = cands_filter
    max_loc = np.array(max_loc).astype(np.int32)
    print("Filtered results: ", max_loc.shape[1], np.max(max_val))

    corners = corners + max_loc.transpose(1,0).reshape(-1, 1, 2)
    corners = corners.astype(np.int32)
#      for ii in range(len(corners)):
    #      pts = corners[ii].reshape(-1, 1, 2)
    #      cv2.polylines(img_rgb, [pts], True, (0, 255, 255))
    #  cv2.imshow("Image", img_rgb)
    #  cv2.waitKey(0)

    fname = os.path.basename(fpath_template)
    print("Results are saved to " + fpath_mainimage + "_" + fname + ".txt")
    save_corners(fpath_mainimage + ".txt", corners)
    return corners

def save_corners(fpath, corners):
    with open(fpath, "w") as f:
        for bb in corners:
            bb = np.array(bb).reshape(-1)
            f.write(" ".join([str(v) for v in bb]) + "\n")

def test_tm():
    match_template("images/Input1.png", "images/Input2Ref.png")

if __name__ == "__main__":
    match_template("images/Input2.png", "images/Input2Ref.png")

