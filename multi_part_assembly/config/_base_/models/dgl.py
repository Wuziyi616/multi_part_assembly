"""DGL model."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'dgl'
_C.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'
_C.pc_feat_dim = 128
# GNN refinement iteration
# I find that 3 iter is faster and better than 5 iter
_C.gnn_iter = 3  # 5
# pool and unpool geometrically equivalent parts
# I find that not doing node pooling/unpooling is even better
_C.merge_node = False  # True


def get_cfg_defaults():
    return _C.clone()
