"""Adam optimizer with Cosine LR decay."""

from yacs.config import CfgNode as CN

_C = CN()
_C.lr = 1e-3
_C.weight_decay = 0.
_C.warmup_ratio = 0.
_C.clip_grad = None
_C.lr_scheduler = 'cosine'
_C.lr_decay_factor = 100.


def get_cfg_defaults():
    return _C.clone()
