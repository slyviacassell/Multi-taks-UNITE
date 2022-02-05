# coding=utf-8

'''
Created: 2021/3/1
@author: Slyviacassell@github.com
'''

from yacs.config import CfgNode

_C = CfgNode()

# dataset
_C.DATASET_NAME = 'nyud'
# _C.DATASET_DIR = '/workspace/projects/NDDR/datasets/nyud/'  # local test path
_C.DATASET_DIR = '/workspace/nyud/'

_C.TRAIN = CfgNode()
_C.TRAIN.OUTPUT_SIZE = (321, 321)
_C.TRAIN.RANDOM_SCALE = True
_C.TRAIN.RANDOM_MIRROR = True
_C.TRAIN.RANDOM_CROP = True
_C.TRAIN.RANDOM_JITTER = False

_C.TEST = CfgNode()
_C.TEST.OUTPUT_SIZE = (-1, -1)
_C.TEST.RANDOM_SCALE = False
_C.TEST.RANDOM_MIRROR = False
_C.TEST.RANDOM_CROP = False
_C.TEST.RANDOM_JITTER = False

_C.IGNORE_LABEL = 255

# mtl cfg
_C.MTL = CfgNode()
_C.MTL.TASKS = ['seg', 'normal']
_C.MTL.N_OUTPUT_PER_TASK = [40, 3]
_C.MTL.BACKBONE_TYPE = 'VGG16V1'
# _C.MTL.BACKBONE_TYPE = 'R50V1'
# _C.MTL.WEIGHT_DIR = '/workspace/projects/NDDR/weights/'  # local test path for all weights
_C.MTL.WEIGHT_DIR = '/workspace/weights/'
_C.MTL.WEIGHT_PER_TASK = ['Seg', 'Normal']
# _C.MTL.WEIGHT_PER_TASK = ['DeepLabV3', 'DeepLabV3']
_C.MTL.LOSS_FACTORS = [1., 20.]

# unite cfg
_C.MODEL = CfgNode()
_C.MODEL.N_PATCHES = [4, 4, 4, 4, 4]

_C.MODEL.SIMILARITY = 'additive'
_C.MODEL.RECONSTRUCT_SRC = 'unpool'
_C.MODEL.INIT_METHOD = 'constant'
_C.MODEL.INIT_WEIGHTS = [0.9, 0.1]

_C.MODEL.SHORTCUT = True
_C.MODEL.BN_BEFORE_ACTIVATION = False

# optimization cfg
_C.BASE_LR = 1e-3
_C.BASE_FACTOR = 1.
_C.UNITES_FACTOR = 100.
# VGG
_C.FC8_W_FACTOR = 10.
_C.FC8_B_FACTOR = 20.
# V1
_C.HEAD_FACTOR = 50.
# V3 V3P
_C.CLASSIFIER_FACTOR = 50.

_C.WEIGHT_DECAY = 2.5e-4
_C.MOMENTUM = 0.9

_C.N_WARMUP_UNITS = 0
_C.SCHEDULER = 'Poly'
_C.POLY_POWER = 0.9


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return _C.clone()
