"""Loss configuration in semantic assembly task."""

from yacs.config import CfgNode as CN

# the best loss setting on PartNet dataset after some ablation
#   - translation l2 with weight = 1
#   - rotated part points chamfer with weight = 10
#   - transformed whole shape points chamfer with weight = 10
#   - not using direct loss on rotation angle, l2 loss on rotated points
#       because there is no clear point correspondence here given the symmetry
#       of parts, and many parts are extremely similar to each other
_C = CN()
_C.noise_dim = 32  # stochastic PoseRegressor
_C.sample_iter = 5  # MoN loss sampling

_C.trans_loss_w = 1.
_C.rot_pt_cd_loss_w = 10.
_C.transform_pt_cd_loss_w = 10.
# cosine regression loss on rotation
_C.use_rot_loss = False
# per-point l2 loss between rotated part point clouds
_C.use_rot_pt_l2_loss = False


def get_cfg_defaults():
    return _C.clone()
