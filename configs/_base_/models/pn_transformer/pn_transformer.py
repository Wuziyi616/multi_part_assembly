"""PointNet-Transformer model."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'pn_transformer'
_C.rot_type = 'quat'
_C.pc_feat_dim = 256

_C.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'

_C.transformer_feat_dim = 1024
_C.transformer_heads = 8
_C.transformer_layers = 4
_C.transformer_pre_ln = True


def get_cfg_defaults():
    return _C.clone()
