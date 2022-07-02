import os
from yacs.config import CfgNode as CN
from multi_part_assembly.utils import merge_cfg

_base_ = {
    'exp': '../_base_/default_exp.py',
    'data': '../_base_/datasets/partnet/partnet_chair.py',
    'optimizer': '../_base_/schedules/adam_cosine.py',
    'model': '../_base_/models/lstm.py',
    'loss': '../_base_/models/loss/semantic_loss.py',
}

# Miscellaneous configs
_C = CN()

_C.data = CN()
_C.data.shuffle_parts = True  # avoid part label leakage from orders


def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    return cfg
