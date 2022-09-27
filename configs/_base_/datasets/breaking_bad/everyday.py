"""Everyday subset from the Breaking Bad dataset."""

from yacs.config import CfgNode as CN

_C = CN()
_C.dataset = 'geometry'
_C.data_dir = './data/breaking_bad'
_C.data_fn = 'everyday.{}.txt'
_C.data_keys = ('part_ids', )
_C.category = ''  # empty means all categories
_C.rot_range = -1.  # rotation range for curriculum learning
_C.num_pc_points = 1000  # points per part
_C.min_num_part = 2
_C.max_num_part = 20
_C.shuffle_parts = False
_C.overfit = -1
_C.all_category = [
    'BeerBottle', 'Bowl', 'Cup', 'DrinkingUtensil', 'Mug', 'Plate', 'Spoon',
    'Teacup', 'ToyFigure', 'WineBottle', 'Bottle', 'Cookie', 'DrinkBottle',
    'Mirror', 'PillBottle', 'Ring', 'Statue', 'Teapot', 'Vase', 'WineGlass'
]
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
