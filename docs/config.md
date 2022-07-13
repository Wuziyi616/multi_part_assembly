# Config System

Our config system is built upon [yacs](https://github.com/rbgirshick/yacs).
Please first familiarize yourself with `yacs` [here](https://github.com/rbgirshick/yacs#usage).

The original `yacs` system puts everything in a `py` file and use additional `yaml` files to override it.
Inspired by the config system of [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html), we instead extend `yacs` to support inheritance and composition of multiple `py` files.
By doing so we can put different config components (dataset, model, etc.) into different files, and composite them to obtain new settings.
We believe this provides a more clear and structured config system in the codebase.

## Config Inheritance

We support building one child config file by inheriting multiple parent config files, and changing some of parents' values.

For example, if we have a config file `datasets/partnet.py` for the PartNet dataset as:

```
from yacs.config import CfgNode as CN

_C = CN()
_C.dataset = 'partnet'
_C.data_dir = './data/partnet'


def get_cfg_defaults():
    return _C.clone()

```

And another config file `model/global.py` for the Global model as:

```
from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'global'
_C.encoder = 'pointnet'


def get_cfg_defaults():
    return _C.clone()

```

Then, we want a config to train the Global model on the PartNet dataset, we can write a `global-partnet.py` file adopting the values from `partnet.py` and `global.py`:

```
import os
from yacs.config import CfgNode as CN
from multi_part_assembly.utils import merge_cfg

# inheriting configs by specifying `_base_`
# 'data' field will be from `partnet.py`
# 'model' field will be from `global.py`
_base_ = {
    'data': 'datasets/partnet.py',
    'model': 'model/global.py',
}

_C = CN()  # create self in yacs

# to override a field in a parent config
# you need to first create a `CN()` base for it
# then modify the value by setting new values
_C.data = CN()
_C.data.data_dir = './datasets/partnet'  # overriding `data_dir`


# merging code
def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    return cfg

```

Then, we can import this `py` file and call `get_cfg_defaults()` to construct the config.
In this case, it will contain two fields: `data` and `model`.
The `model` field will be the same as `global.py`, while the `data_dir` value in the `data` field will be changed to `./datasets/partnet`.

## Config Structure

In this codebase, each training config is composed of a **exp** (general settings, e.g. checkpoint, epochs), a **data** (dataset setting), a **optimizer** (learning rate and scheduler), a **model** (network architecture), and a **loss** config.
These basic configs are stored under `configs/_base_/`.

We follow the below style to name each training config:

```
{model}_{batch_per_gpu x gpu}_{schedule}_{dataset}.py
```

To inspect one specific config file, simply call our provided script:

```
python scripts/print_cfg.py --cfg_file $CFG
```
