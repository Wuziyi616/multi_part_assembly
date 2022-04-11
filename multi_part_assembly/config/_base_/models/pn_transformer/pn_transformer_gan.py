"""PointNet-Transformer model with adversarial loss."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'pn_transformer_gan'
_C.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'
_C.pc_feat_dim = 256
_C.transformer_feat_dim = 1024
_C.transformer_heads = 8
_C.transformer_layers = 4
_C.transformer_pre_ln = True
_C.discriminator = 'pointnet'  # encoder used in the shape discriminator
_C.discriminator_num_points = 1024
_C.discriminator_loss = 'mse'  # 'ce'


def get_cfg_defaults():
    return _C.clone()
