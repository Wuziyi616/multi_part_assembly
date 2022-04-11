import os
from yacs.config import CfgNode as CN
from multi_part_assembly.config.utils import merge_cfg

_base_ = {
    'exp': '../../_base_/default_exp.py',
    'data': '../../_base_/datasets/artifact.py',
    'optimizer': '../../_base_/schedules/adam_cosine.py',
    'model': '../../_base_/models/pn_transformer/pn_transformer_gan.py',
    'loss': '../../_base_/models/loss/geometric_loss.py',
}

# Miscellaneous configs
_C = CN()

_C.exp = CN()
_C.exp.num_epochs = 400

_C.optimizer = CN()
_C.optimizer.d_lr = 1e-3
_C.optimizer.warmup_ratio = 0.05

_C.loss = CN()
_C.loss.g_loss_w = 1.
_C.loss.d_loss_w = 1.


def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    return cfg