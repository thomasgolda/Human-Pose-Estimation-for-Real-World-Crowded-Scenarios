import yaml
import argparse
import glob
from easydict import EasyDict
from os.path import join

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--old_cfg', type=str, dest="old_cfg", default = "")
    parser.add_argument('--outdir', type=str, dest="outdir", default = "")
    args = parser.parse_args()


    return args

def create_base_cfg():
    cfg = EasyDict()

    cfg.EXPERIMENT_NAME= "crowdpose_baseline"           #Name of the experiment. The output will be placed in a directory with the same name
    cfg.DATASET = "../datasets/CrowdPose.yaml"          #Path to the dataset config the experiment should be run on

    cfg.MODEL = EasyDict()
    cfg.MODEL.input_shape =  [256,192]          # [256,192], [384,288]
    cfg.MODEL.occluded_detection = False        # enables occlussion detection with OccNet
    cfg.MODEL.occluded_cross_branch= False      # enables occlussion detection with OccNetCB
    cfg.MODEL.occluded_hard_loss= False         # enables loss that
    cfg.MODEL.occluded_loss_weight= 1.5         # Weight factor to increase the importance of the occluded loss
    cfg.MODEL.stop_crossbranch_grad= False      # Disable gradient calculation in OccNetCB between the occluded and visible branch
    cfg.MODEL.interference_joints= False        # Train network to detect every joint within the target persons bounding box. Similar to AlphaPose
    cfg.MODEL.backbone= "resnet50"              # 'resnet50', 'resnet101', 'resnet152'

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.batch_size= 64
    cfg.TRAIN.optimizer= "adam"
    cfg.TRAIN.weight_decay= 0.00001
    cfg.TRAIN.batch_norm= True
    cfg.TRAIN.scale_factor= 0.35
    cfg.TRAIN.rotation_factor= 40
    cfg.TRAIN.structure_aware_loss= False       # Enable structure aware loss
    cfg.TRAIN.structure_aware_loss_weight= 0.6  # Weighting favtor for the structure aware loss.
    cfg.TRAIN.lr= 0.001
    cfg.TRAIN.lr_dec_epoch= [90, 120]
    cfg.TRAIN.lr_dec_factor= 10
    cfg.TRAIN.end_epoch= 140
    cfg.TRAIN.coco_cutout= False                # Enable cutout augmentation with objects from COCO Instance Segmenation.
    cfg.TRAIN.coco_person_cutout= False         # Enable cutout augmenation with body parts from COCO Keypoint Dataset.
    cfg.TRAIN.coco_root= "../../../data/coco"   # PATH to COCO dataset root
    cfg.TRAIN.fullbodycut= False                # Enable cutout augmenation with full persons from COCO Keypoint Dataset. Note that person will no be placed at the center of the image to avoid complete overlap
    cfg.TRAIN.random_cutout= False              # Randomly switch between two enabled cutout methods.
    cfg.TRAIN.fixed_seed= True
    cfg.TRAIN.reset_v_flags= False

    cfg.TEST = EasyDict()
    cfg.TEST.flip_test= True
    cfg.TEST.oks_nms_thr= 0.9
    cfg.TEST.score_thr= 0.2
    cfg.TEST.test_batch_size= 4
    cfg.TEST.testset = "test"      # 'resnet50', 'resnet101', 'resnet152'
    cfg.TEST.useGTbbox= True       # use human detections or groundtruth bounding boxes provided with the annotations.

    return cfg

def save_cfg(cfg, outdir=""):
    cfg = dict(cfg)
    for k, v in cfg.items():
        if isinstance(v, EasyDict):
            cfg[k] = dict(v)
    outfile = join(outdir, "{}.yaml".format(cfg["EXPERIMENT_NAME"]))
    with open(outfile, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def get_path_to_dataset(dataset_name):
    for yaml_cfg in glob.glob("../datasets/*.yaml"):
        with open(yaml_cfg) as f:
            dataset = EasyDict(yaml.load(f))
        if dataset.name == dataset_name:
            return yaml_cfg
    return None

def load_old_cfg(cfg_path, cfg):
    with open(cfg_path) as f:
        cfg_old = EasyDict(yaml.load(f))

    cfg.EXPERIMENT_NAME = cfg_old.experiment_name
    cfg.DATASET = get_path_to_dataset(cfg_old.dataset)

    cfg.MODEL.occluded_detection = cfg_old.occluded_detection
    cfg.MODEL.occluded_cross_branch = cfg_old.occluded_cross_branch
    cfg.MODEL.occluded_hard_loss = cfg_old.occluded_hard_loss
    cfg.MODEL.occluded_loss_weight = cfg_old.occluded_loss_weight
    cfg.MODEL.interference_joints = cfg_old.interference_joints
    cfg.MODEL.backbone = cfg_old.backbone
    cfg.MODEL.input_shape = cfg_old.input_shape

    cfg.TRAIN.batch_size = cfg_old.batch_size
    cfg.TRAIN.coco_cutout = cfg_old.coco_cutout
    cfg.TRAIN.coco_person_cutout = cfg_old.coco_person_cutout
    cfg.TRAIN.fullbodycut = cfg_old.fullbodycut
    cfg.TRAIN.structure_aware_loss = cfg_old.structure_aware_loss
    cfg.TRAIN.structure_aware_loss_weight = cfg_old.structure_aware_loss_weight
    cfg.TRAIN.end_epoch = cfg_old.end_epoch
    cfg.TRAIN.lr = cfg_old.lr
    cfg.TRAIN.lr_dec_epoch = cfg_old.lr_dec_epoch
    cfg.TRAIN.random_cutout = cfg_old.random_cutout

    cfg.TEST.flip_test = cfg_old.flip_test
    cfg.TEST.oks_nms_thr = cfg_old.oks_nms_thr
    cfg.TEST.score_thr = cfg_old.score_thr

    cfg.TEST.test_batch_size = cfg_old.test_batch_size
    cfg.TEST.testset = cfg_old.testset
    cfg.TEST.useGTbbox = cfg_old.useGTbbox

    return cfg


if __name__ == '__main__':

    args = parse_args()
    cfg = create_base_cfg()
    if (args.old_cfg != ""):
        load_old_cfg(args.old_cfg, cfg)

    save_cfg(cfg, args.outdir)





