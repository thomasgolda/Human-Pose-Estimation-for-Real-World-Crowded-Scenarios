import numpy as np
from config import cfg

import cv2



def get_keypoints_in_bb(box, keypoints):
    interference_kpts_list = []
    for keypoint_set in keypoints:
        interference_kpts = np.zeros(keypoints[0].shape)
        interference = False
        for i, kpt in enumerate(keypoint_set):
            if (kpt[0]>=box[0] and
                kpt[0]<=box[2] and
                kpt[1]>=box[1] and
                kpt[1]<=box[3] and
                kpt[2] != 0):
                interference_kpts[i,] = kpt
                interference = True
        if(interference):
            interference_kpts_list.append(interference_kpts)
            interference = False
    return interference_kpts_list



colors = [[[255, 0, 0],[0,0,255]],[[128, 0, 0],[0,0,128]]]
def draw_occ_kpts(img, kpts, v_flags):
    for i, kpt in enumerate(kpts):
        p1 = kpt[0].astype(np.int32), kpt[1].astype(np.int32)
        if(kpt[2] > 0.7):

            img = cv2.circle(img, p1,2, colors[0][v_flags[i]], thickness=2)
        else:
            img = cv2.circle(img, p1, 2, colors[1][v_flags[i]], thickness=2)

    # cv2.imshow("fig", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

def bbs_to_n_xyxy(bb_list):
    return np.array([bb_list[0], bb_list[1], bb_list[0] + bb_list[2], bb_list[1] + bb_list[3]])

def draw_box(img, box):
    # box = box.astype(np.int32)
    box = bbs_to_n_xyxy(box)
    p1 = box[0].astype(np.int32), box[1].astype(np.int32)
    p2 = box[2].astype(np.int32), box[3].astype(np.int32)
    img = cv2.rectangle(img, p1, p2, (0, 255, 0))
    return img

def get_keypoints_in_bb(box, keypoints):
    interference_kpts_list = []
    for keypoint_set in keypoints:
        interference_kpts = np.zeros(keypoints[0].shape)
        interference = False
        for i, kpt in enumerate(keypoint_set):
            if (kpt[0] >= box[0] and
                    kpt[0] <= box[2] and
                    kpt[1] >= box[1] and
                    kpt[1] <= box[3] and
                    kpt[2] != 0):
                interference_kpts[i,] = kpt
                interference = True
        if (interference):
            interference_kpts_list.append(interference_kpts)
            interference = False
    return interference_kpts_list