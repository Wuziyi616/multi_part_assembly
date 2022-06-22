import os
from yacs.config import CfgNode as CN
from multi_part_assembly.config.utils import merge_cfg

_base_ = {
    'exp': '../_base_/default_exp.py',
    'data': '../_base_/datasets/breaking_bad/artifact.py',
    'optimizer': '../_base_/schedules/adam_cosine.py',
    'model': '../_base_/models/global.py',
    'loss': '../_base_/models/loss/geometric_loss.py',
}

# Miscellaneous configs
_C = CN()


def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    return cfg
