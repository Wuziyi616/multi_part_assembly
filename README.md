# Multi Part Assembly

## Prerequisite

Install custom CUDA ops for Chamfer distance and PointNet modules

1. Go to `multi_part_assembly/utils/chamfer` and run `pip install -e .`
2. Go to `multi_part_assembly/models/modules/encoder/pointnet2/pointnet2_ops_lib` and run `pip install -e .`

If you meet any errors, make sure your nvcc version is the same as the CUDA version that PyTorch is compiled for. You can get your nvcc version by `nvcc --version`

Install this package: run `pip install -e .`

## Config System

Our config system is built upon [yacs](https://github.com/rbgirshick/yacs), which is extended to support inheritance and composition of multiple config files.

For example, if we have a `datasets/partnet.py` for the PartNet dataset as:

```
from yacs.config import CfgNode as CN

_C = CN()
_C.dataset = 'partnet'
_C.data_dir = './data/partnet'
_C.category = 'Chair'

...


def get_cfg_defaults():
    return _C.clone()

```

Then, we write another config `foo.py` adopting the values from `partnet.py`:

```
from yacs.config import CfgNode as CN

# 'data' field will be from `partnet.py`
_base_ = {'data': 'datasets/partnet.py'}

_C = CN()

_C.exp = CN()
_C.exp.num_epochs = 200

...


# merging code
def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    return cfg

```

Then, when calling `foo` it will have both `exp` field and `data` field. Note that the values set in the child config will overwrite the base one.

In general, each training config is composed of a **exp** (general settings, e.g. checkpoint, epochs), a **data** (dataset setting), a **optimizer** (learning rate and scheduler), a **model** (network architecture), and a **loss** config.

## Training

To train a model, simply run:

```
python script/train.py --cfg_file multi_part_assembly/config/global/global-32x1-cosine_200e-partnet_chair.py
```

Optional arguments:

-   `--category`: train the model only on a subset of data
-   `--gpus`: setting training GPUs
-   `--weight`: loading pre-trained weights
-   `--fp16`: FP16 mixed precision training
-   `--cudnn`: setting `cudnn.benchmark = True`

## Testing

Similar to training, to test a pre-trained weight, simply run:

```
python script/test.py --cfg_file multi_part_assembly/config/global/global-32x1-cosine_200e-partnet_chair.py --weight path/to/weight
```

Optional auguments:

-   `--category`: test the model only on a subset of data
-   `--min_num_part` & `--max_num_part`: control the number of pieces we test
-   `--gpus`: setting testing GPUs

If you want to get per-category result of this model, and report performance averaged over all the categories (used in the paper), run:

```
python script/test.py --cfg_file multi_part_assembly/config/global/global-32x1-cosine_200e-partnet_chair.py --weight path/to/weight --category all
```

We will print the metrics on each category and the averaged results.

We also provide script to test your per-category trained models (**currently only support everyday dataset**). Suppose you train the models by running `./scrips/train_everyday_categories.sh $COMMAND $CFG.py`. Then the model checkpoint will be saved in `checkpoint/$CFG-$CATEGORY-dup$X`. To collect the performance on each category, run:

```
python script/collect_test.py --cfg_file $CFG.py --num_dup $X --ckp_suffix checkpoint/$CFG-
```

You can again control the number of pieces and GPUs to use.

## Visualization

To visualize the results produced by trained model, simply run:

```
python scrips/vis.py --cfg_file $CFG --weight path/to/weight --category $CATEGORY --vis $NUM_TO_SAVE
```

It will save the original meshes, input meshes after random transformation and meshes transformed by model predictions, as well as point clouds sampled from them in `path/to/vis` folder (same as the pre-trained weight).
