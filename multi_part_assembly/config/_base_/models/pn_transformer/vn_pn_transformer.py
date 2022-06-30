"""VN-PointNet-Transformer model."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'vn_pn_transformer'
_C.rot_type = 'rmat'
_C.pc_feat_dim = 64  # use a smaller one because VN feature is 3xC

_C.encoder = 'vn-pointnet'
_C.encoder_pool1 = 'mean'  # pooling for the input edge-conv
_C.encoder_pool2 = 'max'  # pooling for the global feature

_C.transformer_heads = 4
_C.transformer_layers = 4
_C.transformer_relu = True


def get_cfg_defaults():
    return _C.clone()
