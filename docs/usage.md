# Basic Usage

## Training

To train a model, simply run:

```
python script/train.py --cfg_file $CFG --other_args ...
```

For example, to train the Global baseline model on PartNet chair, replace `$CFG` with `configs/global/global-32x1-cosine_200e-partnet_chair.py`.
Other optional arguments include:

-   `--category`: train the model only on a subset of data, e.g. `Chair`, `Table`, `Lamp` on PartNet
-   `--gpus`: setting training GPUs, note that by default we are using DP training. Please modify `script/train.py` to enable DDP training
-   `--weight`: loading pre-trained weights
-   `--fp16`: FP16 mixed precision training
-   `--cudnn`: setting `cudnn.benchmark = True`
-   `--vis`: visualize assembly results to wandb during training, may take large disk space

### Logging

We use [wandb](https://wandb.ai/site) for logging.
Please set up your account on the machine before running training commands.

### Helper Scripts

Script for configuring and submitting jobs to cluster SLURM system:

```
GPUS=1 CPUS_PER_TASK=8 MEM_PER_CPU=5 QOS=normal ./script/sbatch_run.sh $PARTITION $JOB_NAME ./script/train.py --cfg_file $CFG --other_args...
```

Script for running a job multiple times over different random seeds:

```
GPUS=1 CPUS_PER_TASK=8 MEM_PER_CPU=5 QOS=normal REPEAT=$NUM_REPEAT ./script/dup_run_sbatch.sh $PARTITION $JOB_NAME ./script/train.py $CFG --other_args...
```

We also provide scripts for training on single/all categories of the Breaking-Bad dataset's `everyday` subset.
See [train_everyday_categories.sh](../script/train_everyday_categories.sh) and [train_one_category.sh](../script/train_one_category.sh).

## Testing

Similar to training, to test a pre-trained weight, simply run:

```
python script/test.py --cfg_file $CFG --weight path/to/weight
```

Optional auguments:

-   `--category`: test the model only on a subset of data
-   `--min_num_part` & `--max_num_part`: control the number of pieces we test
-   `--gpus`: setting testing GPUs

If you want to get per-category result of this model (currently only support `everyday` subset of the Breaking-Bad dataset), and report performance averaged over all the categories (used in the paper), run:

```
python script/test.py --cfg_file $CFG --weight path/to/weight --category all
```

We will print the metrics on each category and the averaged results.

We also provide script to test your per-category trained models (currently only support `everyday` subset of the Breaking-Bad dataset). Suppose you train the models by running `./scrips/train_everyday_categories.sh $COMMAND $CFG.py`. Then the model checkpoint will be saved in `checkpoint/$CFG-$CATEGORY-dup$X`. To collect the performance on each category, run:

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
