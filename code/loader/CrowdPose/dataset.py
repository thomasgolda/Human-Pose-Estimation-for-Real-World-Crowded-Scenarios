#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import glob
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval
import sys
import pickle

from utils.kpt_tools import get_keypoints_in_bb, bbs_to_n_xyxy

cur_dir = os.path.dirname(__file__)
sys.path.insert(0, osp.join(cur_dir, 'PythonAPI'))
SCALE = (0.35 + 1) * 1.25
class Dataset(object):
    
    dataset_name = 'CrowdPose'
    num_kps = 14
    kps_names =['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck']
    kps_symmetry = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    kps_lines = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (12, 13)]
    data_root = "../../data"
    #sigma values are copied from crowdpose eval tool: they copied from coco tool but do no information where
    #the sigmas for head_top and head_bottom are
    sigmas = np.array([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79]) / 10.0
    train_annot_path = "ss"


    def load_train_data(self, score=False, generate_interference_joints = False, reset_v_flags=False):
        print(self.train_annot_path)
        coco = COCO(self.train_annot_path)
        train_data = []
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            imgname = coco.imgs[ann['image_id']]['file_name']

            if(reset_v_flags):#setting all the nonsensical v-flags to visible
                joints = []
                for i, x in enumerate(ann['keypoints']):
                    if (i % 3) == 2 and x == 1:
                        joints.append(2)
                    else:
                        joints.append(x)
            else:
                joints = ann['keypoints']

            #todo: cannot check for num_keypoints == 0 because CrowdPose num_keypoints does not account for the head
            if (ann['image_id'] not in coco.imgs) or ann['iscrowd']:
                continue
            
            # sanitize bboxes
            x, y, w, h = ann['bbox']
            img = coco.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if x2 >= x1 and y2 >= y1: #ann['area'] > 0 #tobi: CrowdPose does not provide area
                bbox = [x1, y1, x2-x1, y2-y1]
            else:
                continue
            inteference_kpts = []
            if(generate_interference_joints):
                bbox_xy = bbs_to_n_xyxy([bbox[0], bbox[1], bbox[2] * SCALE, bbox[3] * SCALE])
                anns_ids = coco.getAnnIds(imgIds=ann['image_id'])
                for i, aid in enumerate(anns_ids):
                    ann_inter = coco.anns[aid]
                    if ann["id"] == aid:#cannot check for num_keypoints == 0 because CrowdPose num_keypoints does not account for the head
                        continue
                    inteference_kpts.append(np.array(ann_inter["keypoints"]).reshape((-1, 3)))
                inteference_kpts = get_keypoints_in_bb(bbox_xy, inteference_kpts)

            if score:
                data = dict(image_id = ann['image_id'], imgpath = imgname, bbox=bbox, joints=joints, interference_kpts=inteference_kpts, num_keypoints = ann['num_keypoints'], score=1)
            else:
                data = dict(image_id = ann['image_id'], imgpath = imgname, bbox=bbox, joints=joints, interference_kpts=inteference_kpts, num_keypoints = ann['num_keypoints'])
            train_data.append(data)

        return train_data
    
    def load_val_data_with_annot(self, test_set):
        coco =  self.load_annot(test_set)
        val_data = []
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            if ann['image_id'] not in coco.imgs:
                continue
            imgname = coco.imgs[ann['image_id']]['file_name']
            bbox = ann['bbox']
            joints = ann['keypoints']
            data = dict(image_id=ann['image_id'], imgpath=imgname, bbox=bbox, joints=joints, score=1)
            val_data.append(data)

        return val_data


    
    def load_annot(self, db_set):

        if db_set == 'train':
            coco = COCO(self.train_annot_path)
        elif db_set == 'val':
            coco = COCO(self.val_annot_path)
        elif db_set == 'test':
            coco = COCO(self.test_annot_path)
        else:
            print('Unknown db_set')
            assert 0
        return coco

    def load_imgid(self, annot):
        return annot.imgs

    def imgid_to_imgname(self, annot, imgid, db_set):
        imgs = annot.loadImgs(imgid)
        imgname = [i['file_name'] for i in imgs]
        return imgname


    def evaluation(self, result, gt, result_dir, db_set, experiment_name, test_epoch):

        result_path = osp.join(result_dir, 'result_{}_{}.json'.format(experiment_name, test_epoch))
        result_file_name = "result_{}_{}.txt".format(experiment_name, test_epoch)

        with open(result_path, 'w') as f:
            json.dump(result, f)

        result = gt.loadRes(result_path)
        cocoEval = COCOeval(gt, result, iouType='keypoints')

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', '__', '__', 'AR', 'AR .5', 'AR .75', 'AR (E)', 'AR (M)', 'AR (H)']
        result_path = osp.join(result_dir, result_file_name)
        with open(result_path, "a") as fp:
            for i, name in enumerate(stats_names):
                fp.write("{}: {}\n".format(name, cocoEval.stats[i]))
        result_path = osp.join(result_dir, 'result.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(cocoEval, f, 2)
            print("Saved result file to " + result_path)

    
    def vis_keypoints(self, img, kps, kp_thresh=0.4, alpha=1):
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(self.kps_lines) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # Perform the drawing on a copy of the image, to allow for blending.
        kp_mask = np.copy(img)

        # Draw mid shoulder / mid hip first for better visualization.
        mid_shoulder = (kps[:2, 0] + kps[:2, 1]) / 2.0
        sc_mid_shoulder = np.minimum(kps[2, 0], kps[2, 1])
        mid_hip = (kps[:2, 6] + kps[:2, 7]) / 2.0
        sc_mid_hip = np.minimum(kps[2, 6], kps[2, 7])

        head_bot_idx = 13
        if sc_mid_shoulder > kp_thresh and kps[2, head_bot_idx] > kp_thresh:
            cv2.line(
                kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(kps[:2, head_bot_idx].astype(np.int32)),
                color=colors[len(self.kps_lines)], thickness=2, lineType=cv2.LINE_AA)
        if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
            cv2.line(
                kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(mid_hip.astype(np.int32)),
                color=colors[len(self.kps_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

        for l in range(len(self.kps_lines)):
            i1 = self.kps_lines[l][0]
            i2 = self.kps_lines[l][1]
            p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
            p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
            if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                cv2.line(
                    kp_mask, p1, p2,
                    color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            if kps[2, i1] > kp_thresh:
                cv2.circle(
                    kp_mask, p1,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if kps[2, i2] > kp_thresh:
                cv2.circle(
                    kp_mask, p2,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

        # Blend the keypoints.
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

    def setup_paths(self, cfg):
        self.train_annot_path = os.path.join(cfg.DATASET.root, cfg.DATASET.train_set)
        self.test_annot_path = os.path.join(cfg.DATASET.root, cfg.DATASET.test_set)
        self.val_annot_path = os.path.join(cfg.DATASET.root, cfg.DATASET.val_set)
        self.human_det_path = os.path.join(cfg.DATASET.root, cfg.DATASET.human_dets)
        self.train_images = osp.join(cfg.DATASET.root, cfg.DATASET.train_images)
        self.val_images = osp.join(cfg.DATASET.root, cfg.DATASET.val_images)
        self.test_images = osp.join(cfg.DATASET.root, cfg.DATASET.test_images)


dbcfg = Dataset()
