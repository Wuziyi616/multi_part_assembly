"""Identity model."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'identity'
_C.rot_type = 'quat'
_C.pc_feat_dim = 128


def get_cfg_defaults():
    return _C.clone()
