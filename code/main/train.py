import tensorflow as tf
import argparse
import numpy as np
import random

import config as cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--cfg', type=str, dest="cfg")
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args
args = parse_args()
cfg.set_config(args.cfg, train=True)
cfg.set_args(args.gpu_ids, args.continue_train)
random.seed(2233)

from model import Model
from tfflat.base import Trainer
from tfflat.utils import mem_info
trainer = Trainer(Model(), cfg.cfg)
trainer.train()



