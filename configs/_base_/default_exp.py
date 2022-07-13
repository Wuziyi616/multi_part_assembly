"""Default experimental settings."""

from yacs.config import CfgNode as CN

# Experiment related
_C = CN()
_C.ckp_dir = 'checkpoint/'
_C.weight_file = ''
_C.gpus = [0]
_C.num_workers = 8
_C.batch_size = 32
_C.num_epochs = 200
_C.val_every = 10  # evaluate model every n training epochs
_C.val_sample_vis = 5  # sample visualizations


def get_cfg_defaults():
    return _C.clone()
