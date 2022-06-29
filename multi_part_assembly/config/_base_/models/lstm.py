"""LSTM model."""

from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'lstm'
_C.rot_type = 'quat'
_C.pc_feat_dim = 128

_C.encoder = 'pointnet'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'

_C.lstm_hidden_size = 256


def get_cfg_defaults():
    return _C.clone()
