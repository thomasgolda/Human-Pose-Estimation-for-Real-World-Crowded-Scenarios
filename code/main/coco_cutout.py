from __future__ import absolute_import

import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools.mask import toBbox, encode
import cv2
import random

class CocoGenerator():
    def __init__(self, dataset_root, anno_file, img_set):
        self.coco = COCO(anno_file)
        self.root = dataset_root
        self.img_set = img_set
        self.data_format = ".jpg"

    def image_path_from_index(self, index):
        file_name = '%012d.jpg' % index
        if '2014' in self.img_set:
            file_name = 'COCO_%s_' % self.img_set + file_name

        prefix = 'test2017' if 'test' in self.img_set else self.img_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.root, 'images', data_name, file_name)

        return image_path


class CutoutGenerator(CocoGenerator):
    def __init__(self, dataset_root, anno_file, img_set, inputShape, maxdoO = 0.7, cutout_prob=0.5, fullbodycut = False, avoid_center = False):
        super().__init__(dataset_root, anno_file, img_set)
        self.fullbodycut = fullbodycut
        self.avoid_center = avoid_center
        self.maxdoO = maxdoO
        self.cutout_prob =  1.0 - cutout_prob
        self.cats = [cat['name'] for cat in  self.coco.loadCats( self.coco.getCatIds())][1:-1]
        self.catIds = self.coco.getCatIds(catNms=self.cats)
        self.numCats = len(self.catIds)
        self.ImgIds = []
        self.image_set=img_set
        self.inputShape = [inputShape[1], inputShape[0]]
        self.area = inputShape[0] * inputShape[1]
        self.annIds = None
        if(avoid_center):
            print("avoiding center in cutouts")
        for catId in self.catIds:
            self.ImgIds.append(self.coco.getImgIds(catIds=catId))
        if fullbodycut and len(self.catIds) == 1:
            self.annIds = [x for x in self.coco.anns if self.coco.anns[x]["num_keypoints"] > 3]
        else:
            pass
            #das sollte eigentlich auch noch angepasst werden könnnte allerdings das verhalten des aktuellen Ansaztes
            #ändern deshalb folgt das später
            #einfach alle annotation ids hier reinloaden dann brauch später nicht mehr nach Kategorie gesampled werden
    def get_bounds_from_kpt(self, kpt, kernal_size_h = 100):
        offset_bot = kernal_size_h
        offset_top = kernal_size_h + 1
        return [int(kpt[1]) - offset_bot, int(kpt[1]) + offset_top, int(kpt[0]) - offset_bot, int(kpt[0]) + offset_top]

    def get_rnd_cutout(self):
        if(self.annIds is None):
            rnd = int(random.random() * self.numCats)
            catId =  self.catIds[rnd]
            imgIds = self.ImgIds[rnd]
            imgId = imgIds[np.random.randint(0, len(imgIds))]

            annIds = self.coco.getAnnIds(imgIds=imgId, catIds=catId)
            anns = self.coco.loadAnns(annIds)
            ann = anns[0]
        else:
            rnd_ann_id = self.annIds[int(random.random() * len(self.annIds))]
            ann = self.coco.anns[rnd_ann_id]
            imgId = ann["image_id"]
        filename = self.image_path_from_index(index=imgId)
        mask = self.coco.annToMask(ann)
        bbox = toBbox(encode(mask))
        maskn = np.zeros(mask.shape, dtype=np.uint8)

        if (ann["category_id"] == 1):
            if(self.fullbodycut):
                pass
            else:
                kpts = np.reshape(ann["keypoints"], (-1, 3))
                kpts = [x for x in kpts if x[2] == 2]
                if (len(kpts) > 0):
                    bound = self.get_bounds_from_kpt(kpts[random.randint(0, len(kpts) - 1)])
                    maskn[bound[0]: bound[1], bound[2] : bound[3]] = mask[bound[0]: bound[1], bound[2] : bound[3]]
                    mask = maskn
        image_data = cv2.imread(filename)
        res = cv2.bitwise_and(image_data, image_data, mask=mask)
        bbox = bbox.astype(int)
        return res[bbox[1]:bbox[3] + bbox[1], bbox[0]:bbox[2] + bbox[0]]

    def add_random_obj_toImg(self, img):
        # sample degree of Occlusion of the bb
        occ_mask = np.zeros(img.shape[0:2], dtype=np.bool)
        if (random.random() > self.cutout_prob):
            doO = random.random() * self.maxdoO
            if (doO < 0.08):
                return occ_mask
            area = self.area * doO

            #resize to fit doO
            cutout = self.get_rnd_cutout()
            new_height = int(np.sqrt(area / (cutout.shape[0] / cutout.shape[1])))
            new_width = int(area / new_height)
            cutout = cv2.resize(cutout, (new_height, new_width))

            #sample random Pos and
            x_min, x_max, y_min, y_max = self.get_coords_of_mask_within_bounds(new_height, new_width)

            cutout = cutout[0:x_max - x_min, 0:y_max - y_min]

            cutout_mask_s = np.ma.masked_greater(cutout, 0)

            if(len(cutout_mask_s.mask.shape) == 3):
                np.copyto(img[x_min:x_max, y_min:y_max, :], cutout, where=(cutout_mask_s.mask))
                occ_mask[x_min:x_max, y_min:y_max] = np.logical_or(occ_mask[x_min:x_max, y_min:y_max], cutout_mask_s.mask[:, :, 0])
        return occ_mask

    def get_coords_of_mask_within_bounds(self, h, w):
        if (self.avoid_center):
            if (random.random() > 0.5):
                x_min, y_min = np.random.randint(0, self.inputShape[1]), np.random.randint(self.inputShape[0] // 3 * 2, self.inputShape[0])
                y_max = min(self.inputShape[0], y_min + h)

            else:
                x_min, y_max = np.random.randint(0, self.inputShape[1]), np.random.randint(0, self.inputShape[0]  // 3)
                y_min = max(0, y_max - h)
        else:
            x_min, y_min = self.sample_Random_Pos()
            y_max = min(self.inputShape[0], y_min + h)

        x_max = min(self.inputShape[1], x_min + w)

        return x_min, x_max, y_min, y_max

    def sample_Random_Pos(self):

        offset_x = 40
        offset_y = 30
        return np.random.randint(0 + offset_x, self.inputShape[1] - offset_x), np.random.randint(0 +offset_y, self.inputShape[0] - offset_y)

class KeypointGenerator(CocoGenerator):
    def __init__(self, dataset_root, anno_file, img_set):
        super().__init__(dataset_root, anno_file, img_set)
        self.cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.imgIds = self.coco.getImgIds(catIds=1)

    def __getitem__(self, index):
        imgId = self.imgIds[index]
        im_ann = self.coco.loadImgs(imgId)[0]

        annIds = self.coco.getAnnIds(imgIds=imgId, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        return objs, self.image_path_from_index(imgId)

    def __len__(self):
        return len(self.imgIds)