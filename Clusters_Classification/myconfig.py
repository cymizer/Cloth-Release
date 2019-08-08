import os
import os.path as ops
import numpy as np
from easydict import EasyDict as edict
import torch
_C = edict()

cfg = _C

_C.GPU_ID = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_C.TRAIN_BATCH_SIZE = 32
_C.NUM_WORKERS = 6
_C.LR = 0.001
_C.MOMENTUM = 0.5
_C.MODEL_PATH = './Model'
_C.TEST_BATCH_COUNT = 30
_C.EPOCH = 10
_C.Result_save = './Result'

_C.LOG_INTERVAL = 5
_C.DUMP_INTERVAL = 500
_C.TEST_INTERVAL = 100

_C.IMG_SIZE = 224
_C.CROP_SIZE = 224
_C.INTER_DIM = 512

_C.N_CLUSTERS = 50
_C.FREEZE_PARAM = True
