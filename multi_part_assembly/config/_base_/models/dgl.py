"""DGL model."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'dgl'
_C.rot_type = 'quat'
_C.pc_feat_dim = 128

_C.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'

# GNN refinement iteration
# I find that 3 iter is faster and better than 5 iter
_C.gnn_iter = 3  # 5
# pool and unpool geometrically equivalent parts
# it's indeed helpful in semantic assembly according to my ablation study
_C.merge_node = True


def get_cfg_defaults():
    return _C.clone()
