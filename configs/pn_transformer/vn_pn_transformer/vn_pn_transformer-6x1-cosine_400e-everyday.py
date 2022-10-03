import os
from yacs.config import CfgNode as CN
from multi_part_assembly.utils import merge_cfg

_base_ = {
    'exp': '../../_base_/default_exp.py',
    'data': '../../_base_/datasets/breaking_bad/everyday.py',
    'optimizer': '../../_base_/schedules/adam_cosine.py',
    'model': '../../_base_/models/pn_transformer/vn_pn_transformer.py',
    'loss': '../../_base_/models/loss/geometric_loss.py',
}

# Miscellaneous configs
_C = CN()

_C.exp = CN()
_C.exp.num_epochs = 400
_C.exp.batch_size = 6  # VN models are very memory-consuming
_C.exp.num_workers = 6

_C.optimizer = CN()
_C.optimizer.warmup_ratio = 0.05


def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    return cfg
