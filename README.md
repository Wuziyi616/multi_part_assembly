# Multi Part Assembly

## Prerequisite

### Python Packages

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment setup.
Please install PyTorch manually.
Note that the current code is only tested under PyTorch 1.10, and PyTorch 1.11 will fail due to changes to header files.
Below is an example script installing PyTorch with CUDA 11.3 (please make sure the CUDA version matches your machine, as we will compile custom ops later):

```
conda create -n breaking-bad python=3.8
conda activate breaking-bad
conda install pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Other related packages can installed via:

```
pip install -e .
```

### Custom Ops

Install custom CUDA ops for Chamfer distance and PointNet modules:

1. Go to `multi_part_assembly/utils/chamfer` and run `pip install -e .`
2. Go to `multi_part_assembly/models/modules/encoder/pointnet2/pointnet2_ops_lib` and run `pip install -e .`

If you meet any errors, make sure your nvcc version is the same as the CUDA version that PyTorch is compiled for. You can get your nvcc version by `nvcc --version`

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

-   `--category`: train the model only on a subset of data, e.g. `Chair`, `Table`, `Lamp` on PartNet
-   `--gpus`: setting training GPUs, note that by default we are using DP training. Please modify `script/train.py` to enable DDP training
-   `--weight`: loading pre-trained weights
-   `--fp16`: FP16 mixed precision training
-   `--cudnn`: setting `cudnn.benchmark = True`
-   `--vis`: visualize assembly results to wandb during training, may take large disk space

### Helper Scripts

Script for configuring and submitting jobs to cluster SLURM system:

```
GPUS=1 CPUS_PER_TASK=8 MEM_PER_CPU=5 QOS=normal ./script/sbatch_run.sh $PARTITION $JOB_NAME ./script/train.py --cfg_file $CFG --other_args...
```

Script for running a job multiple times:

```
GPUS=1 CPUS_PER_TASK=8 MEM_PER_CPU=5 QOS=normal REPEAT=$NUM_REPEAT ./script/dup_run_sbatch.sh $PARTITION $JOB_NAME ./script/train.py $CFG --other_args...
```

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
