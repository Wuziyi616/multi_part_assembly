import os
from yacs.config import CfgNode as CN
from multi_part_assembly.utils import merge_cfg

_base_ = {
    'exp': '../_base_/default_exp.py',
    'data': '../_base_/datasets/breaking_bad/artifact.py',
    'optimizer': '../_base_/schedules/adam_cosine.py',
    'model': '../_base_/models/dgl.py',
    'loss': '../_base_/models/loss/geometric_loss.py',
}

# Miscellaneous configs
_C = CN()

_C.model = CN()
_C.model.merge_node = False  # no geometrically equivalent parts

_C.exp = CN()
_C.exp.val_every = 5  # DGL training is very slow

_C.data = CN()
_C.data.data_keys = ('part_ids', 'valid_matrix')


def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    return cfg
