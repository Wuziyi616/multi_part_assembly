"""RGL-Net."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'rgl_net'
_C.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'
_C.pc_feat_dim = 128
_C.gnn_iter = 3


def get_cfg_defaults():
    return _C.clone()
