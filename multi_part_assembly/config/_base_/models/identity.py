"""Identity model."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'identity'
_C.rot_type = 'quat'


def get_cfg_defaults():
    return _C.clone()
