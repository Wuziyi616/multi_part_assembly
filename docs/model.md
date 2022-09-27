# Supported Models

All assembly models follow a similar pipeline:

1. A point cloud encoder extracts features from each input part.
   We support common encoders such as PointNet, PointNet++ and DGCNN
2. A correlation module performs relation reasoning between part features, which can be LSTM, GNN, and Transformer
3. A MLP-based PoseRegressor predicts rotation and translation for each part

## Model Details

Below we briefly describe methods we implement in this repo:

### Global

A naive baseline from [DGL](https://arxiv.org/pdf/2006.07793.pdf).
This model concatenates all part point clouds and extract a _global feature_.
Then, it concatenates the global feature with each part feature as the input to the pose regressor.

### LSTM

A naive baseline from [DGL](https://arxiv.org/pdf/2006.07793.pdf).
This model applies a Bidirectional-LSTM over part features for reasoning, and use the LSTM output for pose prediction.

Note that, in PartNet the order of parts in pre-processed data follow some patterns (e.g. from chair leg to seat to back), which causes information leak if using LSTM.
Therefore, we need to shuffle the order of parts in training data.

### DGL (NeurIPS'20)

Proposed in [DGL](https://arxiv.org/pdf/2006.07793.pdf).
This model leverages a GNN to perform message passing and interactions between parts.
Besides, it adopts an iterative refinement process.
The model first outputs a rough prediction given initial input parts.
Then, it applies the predicted transformation to each part, and runs the model on the transformed parts to predict a _residual_ transformation.
DGL repeats this process for several (3 by default) times, thus refining the prediction to get a good result.

### RGL-NET (WACV'22)

Proposed in [RGL-NET](https://arxiv.org/pdf/2107.12859.pdf).
Intuitively, RGL-NET is a combination of DGL and LSTM.
It applies both the GNN and Bidirectional-LSTM to reason part relations.
It assumes the input parts follow some orders (not necessarily need GT part labels, can also be e.g. part volumes, see Table 4 in their paper).

We do not implement the input sorting operation.
This is because on the one hand, the pre-processed PartNet data does follow some partterns, and we observe that RGL-NET can indeed leverage such partterns.
On the other hand, in geometric assembly, there are no semantically meaningful order of parts.
Indeed, in our Breaking Bad benchmark, RGL-NET performs similarly to DGL, so we do not include it in our paper.

### Transformer-based Methods (our designed)

This class of methods simply replace the GNN with a standard TransformerEncoder to learn part interactions.
We also provide a variant that adopts the iterative refinement process as in DGL.

**Experimental**:

-   We try to incorporate the shape adversarial loss proposed by [NSM](https://arxiv.org/pdf/2205.14886.pdf) to this Transformer-based method.
    However, we do not observe performance improvement.
    Nevetheless, we leave the implementation of this model named `pn_transformer_gan`.
-   We argue that the geometric assembly task requires specific properties of the model, i.e. the predicted translation should be _invariant_ to the input part's rotation, while the predicted rotation should be equivariant.
    Therefore, we build upon the SO(3)-equivariant (Vector Neurons) modules to design a `vn_pn_transformer` model.
    In the current implementation, we have verified that the invariance and equivariance indeed hold.
    However, we do not observe performance improvement.
    Besides, the VN modules are very memory-consuming, making training challenging.

## Benchmarks

### Semantic Assembly

-   Results on PartNet chair:

|                                                            Method                                                             | Shape Chamfer (SCD) ↓ | Part Accuracy (%) ↑ | Connectivity Accuracy (%) |
| :---------------------------------------------------------------------------------------------------------------------------: | :-------------------: | :-----------------: | :-----------------------: |
|                             [Global](../configs/global/global-32x1-cosine_200e-partnet_chair.py)                              |        0.0128         |        23.82        |           16.29           |
|                                [LSTM](../configs/lstm/lstm-32x1-cosine_200e-partnet_chair.py)                                 |        0.0114         |        22.03        |           14.88           |
|                                  [DGL](../configs/dgl/dgl-32x1-cosine_300e-partnet_chair.py)                                  |        0.0079         |        40.56        |           27.58           |
|                            [RGL-NET](../configs/rgl_net/rgl_net-32x1-cosine_300e-partnet_chair.py)                            |        0.0068         |        44.24        |           29.38           |
|           [Transformer](../configs/pn_transformer/pn_transformer/pn_transformer-32x1-cosine_400e-partnet_chair.py)            |        0.0089         |        41.90        |           29.11           |
| [Refine-Transformer](../configs/pn_transformer/pn_transformer_refine/pn_transformer_refine-32x1-cosine_400e-partnet_chair.py) |        0.0079         |        42.97        |           31.25           |

See [wandb report](https://wandb.ai/dazitu616/Multi-Part-Assembly/reports/Benchmark-on-PartNet-Chair-Assembly--VmlldzoyNzI0NTg5?accessToken=zhov8augcax9ud8rvwemv3k9n120i2hvnjiskms6o2nx1esd3xkz8o18l55ugxhv) for detailed training logs.

To reproduce the result, take DGL for example, simply run:

```
GPUS=1 CPUS_PER_TASK=8 MEM_PER_CPU=4 QOS=normal REPEAT=1 ./scripts/dup_run_sbatch_ddl.sh $PARTITION dgl-32x1-cosine_300e-partnet_chair scripts/train.py configs/dgl/dgl-32x1-cosine_300e-partnet_chair.py --fp16 --cudnn
```

Then, you can go to wandb to find the results.

### Geometric Assembly

-   Results on Breaking Bad Dataset: see our [paper](https://openreview.net/forum?id=mJWt6pOcHNy)

**To reproduce our main results on the everyday subset (paper Table 3)**, take DGL for example, please run:

```
./scripts/train_everyday_categories.sh "GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=4 QOS=normal REPEAT=3 ./scripts/dup_run_sbatch.sh $PARTITION dgl-32x1-cosine_200e-everyday-CATEGORY ./scripts/train.py configs/dgl/dgl-32x1-cosine_200e-everyday.py --fp16 --cudnn" configs/dgl/dgl-32x1-cosine_200e-everyday.py
```

-   This assumes you are working on a slurm-based computing cluster.
    If you work on servers then you will need to manually train the model on all categories.
-   In Table 4, we train one model per category, and report the numbers averaged over all categories.
-   Since some categories have only a few base shapes, the results may vary among different runs.
    Therefore, we run all the experiments 3 times and report the average results.
    You can modify the `REPEAT=3` flag above for your need.

After running the above script, the model weights will be saved in `checkpoint/dgl-32x1-cosine_200e-everyday-$CATEGORY-dup$X`, where `$CATEGORY` is the category (e.g. Bottle, Teapot), and `X` indexes different runs.
To collect the results, run (assuming you are in a GPU environment):

```
python scripts/collect_test.py --cfg_file configs/dgl/dgl-32x1-cosine_200e-everyday.py --num_dup 3 --ckp_suffix checkpoint/dgl-32x1-cosine_200e-everyday-
```

It will automatically test each model and collect its evaluation metrics, doing the calculation, and format them into **LaTeX** format, which you can directly copy paste to your table.

**To reproduce our ablation study results (paper Table 4)**, you need to create new config files for each model, and set the `_C.data.max_num_part` to the number you want to try.
Then, you can train the model in the same way as detailed above.

To collect the results, again you can use the `scripts/collect_test.py` script.
To control the number of pieces to test, you can set the `--min_num_part` and `--max_num_part` flags.

**To reproduce our results in the appendix (Table 11 bottom)**, i.e. train one model on all the categories, simply run:

```
GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=4 QOS=normal REPEAT=3 ./scripts/dup_run_sbatch.sh $PARTITION dgl-32x1-cosine_200e-everyday scripts/train.py configs/dgl/dgl-32x1-cosine_200e-everyday.py --fp16 --cudnn
```

Then, you can use the same script to collect the results as detailed above (add a `--train_all` flag because the model is trained on all categories jointly).
