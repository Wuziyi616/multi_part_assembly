# Multi Part Assembly

## Prerequisite

Install custom CUDA ops for Chamfer distance and PointNet modules

1. Go to `multi_part_assembly/utils/chamfer` and run `pip install -e .`
2. Go to `multi_part_assembly/models/modules/encoder/pointnet2/pointnet2_ops_lib` and run `pip install -e .`

If you meet any errors, make sure your nvcc version is the same as the CUDA version that PyTorch is compiled for. You can get your nvcc version by `nvcc --version`

Install this package: run `pip install -e .`

## Training

Our config system is built upon [yacs](https://github.com/rbgirshick/yacs). In general, a `.py` file will serve as the base config of this model, which contains all the options you need for this model. Then, you will create several `.yml` files, each of which overwrites some of the configurations of the base file, where you can conduct different experiments based on it.

To train a model, simply run `python script/train.py --cfg_file multi_part_assembly/config/baseline/global.py --yml_file config/baseline/global.yml`. Optional arguments:

- `--gpus`: setting training GPUs
- `--weight`: loading pre-trained weights
- `--fp16`: FP16 mixed precision training
- `--cudnn`: setting `cudnn.benchmark = True`

## Testing

Similar to training, to test a pre-trained weight, simply run `python script/test.py --cfg_file multi_part_assembly/config/baseline/global.py --yml_file config/baseline/global.yml --weight path/to/weight`
