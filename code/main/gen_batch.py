import os
import numpy as np
import cv2
import random
from config import cfg

from coco_cutout import CutoutGenerator

if cfg.TRAIN.coco_cutout:
    object_cutout_gen = CutoutGenerator(cfg.TRAIN.coco_root,
                                              os.path.join(cfg.TRAIN.coco_root,
                                                           "annotations",
                                                           "instances_train2017.json"),
                                              "train2017",
                                              cfg.MODEL.input_shape)

if cfg.TRAIN.coco_person_cutout:
    if cfg.TRAIN.fullbodycut:
        body_cutout_gen = CutoutGenerator(cfg.TRAIN.coco_root,
                                                  os.path.join(cfg.TRAIN.coco_root,
                                                               "annotations",
                                                               "person_keypoints_train2017.json"),
                                                  "train2017",
                                          cfg.MODEL.input_shape, maxdoO = 0.45, cutout_prob = 0.8, fullbodycut=True, avoid_center=True)

    else:

        body_cutout_gen = CutoutGenerator(cfg.TRAIN.coco_root,
                                                  os.path.join(cfg.TRAIN.coco_root,
                                                               "annotations",
                                                               "person_keypoints_train2017.json"),
                                                  "train2017",
                                          cfg.MODEL.input_shape, maxdoO = 0.7, cutout_prob = 0.8)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def generate_heatmap(joints,  num_joints = 14, output_shape=(64, 48), sigma = 2, input_shape=(256, 192)):

    joints = np.array(joints).reshape(num_joints, 3)
    valid_mask = joints[:, 2]

    target = np.zeros((output_shape[0],
                       output_shape[1],
                       num_joints),
                      dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = [input_shape[1] / output_shape[1], input_shape[0] / output_shape[0]]#input_shape / output_shape
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= output_shape[1] or ul[1] >= output_shape[0] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            valid_mask[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], output_shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], output_shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], output_shape[1])
        img_y = max(0, ul[1]), min(br[1], output_shape[0])

        v = valid_mask[joint_id]
        if v > 0.5:
            target[img_y[0]:img_y[1], img_x[0]:img_x[1], joint_id] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, valid_mask



def generate_interference_maps(interference_joints,  num_joints = 14, output_shape=(64, 48), sigma = 2, input_shape=(256, 192)):
    hm_tot = np.zeros((output_shape[0],
                       output_shape[1],
                       num_joints),
                      dtype=np.float32)
    valid_tot = np.zeros(num_joints)
    for joints in interference_joints:
        hm, valid = generate_heatmap(joints,  num_joints, output_shape, sigma, input_shape)
        valid_tot += valid
        hm_tot += hm
    return hm_tot  * 255 * 0.5, valid_tot

#generate_batch generates returns only one instance NOT THE ENTIRE BATCH
def generate_batch(d, stage='train'):

    img = cv2.imread(os.path.join(cfg.img_path, d['imgpath']), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        print('cannot read ' + os.path.join(cfg.img_path, d['imgpath']))
        assert 0

    bbox = np.array(d['bbox']).astype(np.float32)
    
    x, y, w, h = bbox
    aspect_ratio = cfg.MODEL.input_shape[1]/cfg.MODEL.input_shape[0]
    center = np.array([x + w * 0.5, y + h * 0.5])
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w,h]) * 1.25
    rotation = 0

    if stage == 'train':

        joints = np.array(d['joints']).reshape(cfg.num_kps, 3).astype(np.float32)
        
        # data augmentation
        scale = scale * np.clip(np.random.randn()*cfg.TRAIN.scale_factor + 1, 1-cfg.TRAIN.scale_factor, 1+cfg.TRAIN.scale_factor)
        rotation = np.clip(np.random.randn()*cfg.TRAIN.rotation_factor, -cfg.TRAIN.rotation_factor*2, cfg.TRAIN.rotation_factor*2)\
                if random.random() <= 0.6 else 0
        # img = draw_box(img, [bbox[0], bbox[1], bbox[2] * 1.5, bbox[3] * 1.5])
        if random.random() <= 0.5:
            flipped = True
            img = img[:, ::-1, :]
            center[0] = img.shape[1] - 1 - center[0]
            joints[:,0] = img.shape[1] - 1 - joints[:,0]
            for (q, w) in cfg.kps_symmetry:
                joints_q, joints_w = joints[q,:].copy(), joints[w,:].copy()
                joints[w,:], joints[q,:] = joints_q, joints_w
        else:
            flipped = False
        trans = get_affine_transform(center, scale, rotation, (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]))

        cropped_img = cv2.warpAffine(img, trans, (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), flags=cv2.INTER_LINEAR)

        if(cfg.TRAIN.random_cutout):
            if (random.random() <= 0.5 and cfg.TRAIN.coco_cutout):
                occ_mask = object_cutout_gen.add_random_obj_toImg(cropped_img)
            elif (cfg.TRAIN.coco_person_cutout):
                if (not cfg.TRAIN.fullbodycut or d['num_keypoints'] > 6):
                    occ_mask = body_cutout_gen.add_random_obj_toImg(cropped_img)
                else:
                    occ_mask = np.zeros(cropped_img.shape[0:2], dtype=np.bool)
        else:

            if(cfg.TRAIN.coco_cutout):
                occ_mask = object_cutout_gen.add_random_obj_toImg(cropped_img)

            if(cfg.TRAIN.coco_person_cutout):
                if(not cfg.TRAIN.fullbodycut or d['num_keypoints'] > 6):
                    occ_mask = body_cutout_gen.add_random_obj_toImg(cropped_img)


        cropped_img = cfg.normalize_input(cropped_img)


        for i in range(cfg.num_kps):
            if joints[i,2] > 0:
                joints[i,:2] = affine_transform(joints[i,:2], trans)
                #set flag to 0 if kpt is out of bounds after transform
                joints[i,2] *= ((joints[i,0] >= 0) & (joints[i,0] < cfg.MODEL.input_shape[1]) & (joints[i,1] >= 0) & (joints[i,1] < cfg.MODEL.input_shape[0]))
                #set occ flag if cutout occluds the target joints[i, :2].astype(np.int)
                if(cfg.TRAIN.coco_cutout and joints[i,2] > 0 and occ_mask[int(joints[i,1]), int(joints[i,0])]):
                    #todo: would be better to check the local region of the keypoint
                    #e.g. np.all(occ_mask[joint[0]-1: joint[0]+ 2, joint[1]-1: joint[1]+ 2])
                    joints[i, 2] = 1

        target_coord = joints[:,:2]
        target_valid = joints[:,2]

        #generate heatmaps for interference joints
        if(cfg.MODEL.interference_joints):
            inf_set = []
            for inf_joints in d['interference_kpts']:
                inf_j = np.array(inf_joints).astype(np.float32)
                for i in range(cfg.num_kps):
                    if inf_j[i, 2] > 0:
                        inf_j[i,:2] = affine_transform(inf_j[i, :2], trans)
                inf_set.append(inf_j)
            inf_hms, inf_valid = generate_interference_maps(inf_set, cfg.num_kps,cfg.MODEL.output_shape,cfg.sigma,cfg.MODEL.input_shape)
            target_valid += inf_valid
            if (flipped):
                inf_hms = inf_hms[:,::-1,:].copy()
            return [cropped_img,
                    target_coord,
                    (target_valid > 0),
                    inf_hms]

        if (not cfg.MODEL.occluded_detection):
            return [cropped_img,
                    target_coord,
                    (target_valid > 0)]
        else:
            return [cropped_img,
                    target_coord,
                    (target_valid >= 2),#vis_valid_mask
                    (target_valid == 1),#occluded_valid_mask
                    (target_valid >= 1)]#valid_points


    else:
        trans = get_affine_transform(center, scale, rotation, (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]))
        cropped_img = cv2.warpAffine(img, trans, (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), flags=cv2.INTER_LINEAR)
        cropped_img = cfg.normalize_input(cropped_img)

        crop_info = np.asarray([center[0]-scale[0]*0.5, center[1]-scale[1]*0.5, center[0]+scale[0]*0.5, center[1]+scale[1]*0.5])

        return [cropped_img, crop_info]


