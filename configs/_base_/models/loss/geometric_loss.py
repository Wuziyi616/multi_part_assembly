"""Loss configuration in geometric assembly task."""

from yacs.config import CfgNode as CN

# the best loss setting on Geometry dataset after some ablation
#   - translation l2 with weight = 1
#   - rotation cosine with weight = 0.2
#   - rotated part points per-point l2 with weight = 1
#   - rotated part points chamfer with weight = 10
#   - transformed whole shape points chamfer with weight = 10
#   - it turns out that rotation is VERY hard to learn in this dataset
#       we can achieve best PA and SCD using the same setting as PartNet
#       however, this will result in very bad rotation angle error (~90 deg)
#       so we apply l2 loss on rotation directly
#       also note that there is almost no symmetry in this dataset
_C = CN()
_C.noise_dim = 0  # no stochastic

_C.trans_loss_w = 1.
_C.rot_pt_cd_loss_w = 10.
_C.transform_pt_cd_loss_w = 10.
# cosine regression loss on rotation
_C.use_rot_loss = True
_C.rot_loss_w = 0.2
# per-point l2 loss between rotated part point clouds
_C.use_rot_pt_l2_loss = True
_C.rot_pt_l2_loss_w = 1.


def get_cfg_defaults():
    return _C.clone()
