# Multi Part Assembly

## Prerequisite

Install custom CUDA ops for Chamfer distance and PointNet modules

1. Go to `multi_part_assembly/utils/chamfer` and run `pip install -e .`
2. Go to `multi_part_assembly/models/encoder/pointnet2/pointnet2_ops_lib` and run `pip install -e .`

Install this package: run `pip install -e .`

## Training

To train a model, simply run `python script/train.py --cfg_file config/baseline/default.yml`. Optional arguments:

- `--gpus`: setting training GPUs
- `--weight`: loading pre-trained weights
- `--fp16`: FP16 mixed precision training
- `--cudnn`: setting `cudnn.benchmark = True`

## Testing

To test a pre-trained weight, simply run `python script/train.py --cfg_file config/baseline/default.yml --test --weight path/to/weight`
