"""RGL-Net."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'rgl_net'
_C.rot_type = 'quat'
_C.pc_feat_dim = 128

_C.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'

_C.gnn_iter = 3  # same as DGL, 3 is adopted in the paper
_C.merge_node = True  # same as DGL


def get_cfg_defaults():
    return _C.clone()
