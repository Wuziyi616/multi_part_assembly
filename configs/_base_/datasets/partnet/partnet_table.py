"""PartNet semantic assembly dataset."""

from yacs.config import CfgNode as CN

_C = CN()
_C.dataset = 'partnet'
_C.data_dir = './data/partnet'
_C.data_fn = 'Table.{}.npy'
_C.category = 'Table'  # actually useless
_C.data_keys = ('part_ids', 'match_ids', 'contact_points')
_C.num_pc_points = 1000  # points per part
_C.num_part_category = 82
_C.min_num_part = 2
_C.max_num_part = 20
_C.shuffle_parts = False
_C.overfit = -1
_C.colors = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 204, 0],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
]


def get_cfg_defaults():
    return _C.clone()
