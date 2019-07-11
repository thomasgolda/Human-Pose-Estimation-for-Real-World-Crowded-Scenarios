import yaml
import os.path as osp
import os
import sys
from easydict import EasyDict
import numpy as np
cfg = None
root_dir = ".."
sys.path.insert(0, osp.join(root_dir, 'lib'))

def get_lr(epoch):
    for e in cfg.TRAIN.lr_dec_epoch:
        if epoch < e:
            break
    if epoch < cfg.TRAIN.lr_dec_epoch[-1]:
        i = cfg.TRAIN.lr_dec_epoch.index(e)
        return cfg.TRAIN.lr / (cfg.TRAIN.lr_dec_factor ** i)
    else:
        return cfg.TRAIN.lr / (cfg.TRAIN.lr_dec_factor ** len(cfg.TRAIN.lr_dec_epoch))


def normalize_input(img):
    return img - cfg.pixel_means
def denormalize_input(img):
    return img + cfg.pixel_means

def set_args(gpu_ids, continue_train=False):
    global cfg
    cfg.gpu_ids = gpu_ids
    cfg.num_gpus = len(cfg.gpu_ids.split(','))
    cfg.continue_train = continue_train
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    print('>>> Using /gpu:{}'.format(cfg.gpu_ids))


def set_config(cfg_path, train=True, valid_dataset=""):
    global cfg
    with open(cfg_path) as f:
        cfg = EasyDict(yaml.load(f))


    cfg.root_dir = root_dir
    cfg.output_dir = osp.join(cfg.root_dir, 'output', cfg.EXPERIMENT_NAME)
    with open(cfg.DATASET) as f:
        cfg.DATASET = EasyDict(yaml.load(f))

    cfg.loader_dir = osp.join(cfg.root_dir, 'loader')
    cfg.init_model = osp.join(cfg.root_dir, 'imagenet_weights', 'resnet_v1_' + cfg.MODEL.backbone[6:] + '.ckpt')
    cfg.tensorboard_dir = osp.join(cfg.output_dir, 't_log')
    cfg.model_dump_dir = osp.join(cfg.output_dir, 'model_dump', cfg.DATASET.name)
    cfg.vis_dir = osp.join(cfg.output_dir, 'vis', cfg.DATASET.name)
    cfg.log_dir = osp.join(cfg.output_dir, 'log', cfg.DATASET.name)
    cfg.result_dir = osp.join(cfg.output_dir, 'result', cfg.DATASET.name)


    if(valid_dataset != ""):
        with open(valid_dataset) as f:
            cfg.DATASET = EasyDict(yaml.load(f))
        cfg.vis_dir = osp.join(cfg.vis_dir, cfg.DATASET.name)
        cfg.result_dir = osp.join(cfg.result_dir, cfg.DATASET.name)

    cfg.DATASET.root = os.path.abspath(osp.join(os.getcwd(), cfg.DATASET.root))



    if cfg.TRAIN.structure_aware_loss:
        if (cfg.DATASET.loader == "CrowdPose"):
            cfg.joint_graph = np.array([[16, 14, 12], [15, 13, 11], [10, 8, 6], [9, 7, 5], [0, 6, 5]])  # todo put in
        elif (cfg.DATASET.loader == "COCO"):
            cfg.joint_graph = np.array([[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 10], [0, 1, 13]])
        else:
            print("Joint Graph is undefined for {} loader".format(cfg.DATASET.loader))
        cfg.additional_outputs = len(cfg.joint_graph)
    else:
        cfg.additional_outputs = 0

    cfg.MODEL.output_shape = (cfg.MODEL.input_shape[0]//4, cfg.MODEL.input_shape[1]//4)
    if cfg.MODEL.output_shape[0] == 64:
        cfg.sigma = 2
    elif cfg.MODEL.output_shape[0] == 96:
        cfg.sigma = 3
    cfg.pixel_means = np.array([[[123.68, 116.78, 103.94]]])
    cfg.TRAIN.coco_person_cutout = cfg.TRAIN.coco_person_cutout or cfg.TRAIN.fullbodycut
    cfg.MODEL.occluded_detection = cfg.MODEL.occluded_detection or cfg.MODEL.occluded_cross_branch
    cfg.MODEL.occluded_hard_loss = cfg.MODEL.occluded_detection and cfg.MODEL.occluded_hard_loss

    cfg.multi_thread_enable = True
    cfg.num_thread = 12
    cfg.gpu_ids = '0'
    cfg.num_gpus = 1
    cfg.continue_train = False
    cfg.display = 1
    cfg.tensorboard_update = 100


    from tfflat.utils import add_pypath, make_dir
    add_pypath(osp.join(cfg.loader_dir))
    add_pypath(osp.join(cfg.loader_dir, cfg.DATASET.kpt_format))
    make_dir(cfg.model_dump_dir)
    make_dir(cfg.vis_dir)
    make_dir(cfg.log_dir)
    make_dir(cfg.result_dir)

    from dataset import dbcfg

    dbcfg.setup_paths(cfg)
    cfg.num_kps = dbcfg.num_kps
    cfg.kps_names = dbcfg.kps_names
    cfg.kps_lines = dbcfg.kps_lines
    cfg.kps_symmetry = dbcfg.kps_symmetry

    cfg.human_det_path = dbcfg.human_det_path
    cfg.vis_keypoints = dbcfg.vis_keypoints
    cfg.get_lr = get_lr
    cfg.normalize_input = normalize_input
    cfg.denormalize_input = denormalize_input

    if(train):
        cfg.img_path = dbcfg.train_images

    elif(cfg.TEST.testset == "test"):
        cfg.img_path = dbcfg.test_images

    elif (cfg.TEST.testset == "valid"):
        cfg.img_path = dbcfg.val_images
    else:
        print("{} is not a defined test set.".format(cfg.TEST.testset))