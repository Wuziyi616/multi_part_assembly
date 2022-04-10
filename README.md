# Multi Part Assembly

## Prerequisite

Install custom CUDA ops for Chamfer distance and PointNet modules

1. Go to `multi_part_assembly/utils/chamfer` and run `pip install -e .`
2. Go to `multi_part_assembly/models/modules/encoder/pointnet2/pointnet2_ops_lib` and run `pip install -e .`

If you meet any errors, make sure your nvcc version is the same as the CUDA version that PyTorch is compiled for. You can get your nvcc version by `nvcc --version`

Install this package: run `pip install -e .`

## Training

Our config system is built upon [yacs](https://github.com/rbgirshick/yacs). In general, a `.py` file will serve as the base config of this model, which contains all the options you need for this model. Then, you will create several `.yml` files, each of which overwrites some of the configurations of the base file, where you can conduct different experiments based on it.

To train a model, simply run:

```
python script/train.py --cfg_file multi_part_assembly/config/baseline/global.py --yml_file config/baseline/global.yml
```

Optional arguments:

- `--category`: train the model only on a subset of data
- `--gpus`: setting training GPUs
- `--weight`: loading pre-trained weights
- `--fp16`: FP16 mixed precision training
- `--cudnn`: setting `cudnn.benchmark = True`

## Testing

Similar to training, to test a pre-trained weight, simply run:

```
python script/test.py --cfg_file multi_part_assembly/config/baseline/global.py --yml_file config/baseline/global.yml --weight path/to/weight
```

Optional auguments:

- `--category`: test the model only on a subset of data
- `--min_num_part` & `--max_num_part`: control the number of pieces we test
- `--gpus`: setting testing GPUs

If you want to get per-category result of this model, and report performance averaged over all the categories (used in the paper), run:

```
python script/test.py --cfg_file multi_part_assembly/config/baseline/global.py --yml_file config/baseline/global.yml --weight path/to/weight --category all
```

We will print the metrics on each category and the averaged results.

We also provide script to test your per-category trained models. Suppose you train one model on each of the 20-category geometric assembly dataset, by running `./scrips/run_geometry_categories.sh $COMMAND config/baseline/global_geo.yml`. Then the model checkpoint will be saved in `checkpoint/global/global_geo-$CATEGORY-dup$X`. To collect the performance on each category, run:

```
python script/collect_test.py --cfg_file multi_part_assembly/config/baseline/global.py --yml_file config/baseline/global_geo.yml --num_dup $X --ckp_suffix checkpoint/global/global_geo-
```

You can again control the number of pieces and GPUs to use.

## Visualization

To visualize the results produced by trained model, simply run:

```
python scrips/vis.py --cfg_file multi_part_assembly/config/baseline/global.py --yml_file config/baseline/global.yml --weight path/to/weight --category $CATEGORY --vis $NUM_TO_SAVE
```

It will save the original meshes, input meshes after random transformation and meshes transformed by model predictions, as well as point clouds sampled from them in `path/to/vis` folder.
