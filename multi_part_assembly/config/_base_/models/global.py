"""Global model."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'global'
_C.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'
_C.pc_feat_dim = 128


def get_cfg_defaults():
    return _C.clone()