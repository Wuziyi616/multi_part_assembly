# Tutorial

## Data

We wrap data in `dict` and pass it through the model.
Some key items include:

-   `part_pcs`: point clouds sampled from each object part, usually of shape `[batch_size, num_parts, num_points, 3]`.
    To enable batching, we pad all shape to a pre-defined number (usually 20) of parts with zeros.
-   `part_trans`: ground-truth translation of each part.
    Shape `[batch_size, num_parts, 3]` and padded with zeros.
-   `part_quat`: ground-truth rotation ([quaternion](https://en.wikipedia.org/wiki/Quaternion)) of each part.
    Shape `[batch_size, num_parts, 4]` and padded with zeros.
    Note that we load rotations using quaternion for ease of dataloading.
    See `Rotation Representation` section for more details.
-   `part_valids`: binary mask indicating padded parts.
    Shape `[batch_size, num_parts]`.
    1 means existed parts while 0 stands for padded parts.

For other data items, see comments in the [dataset files](../multi_part_assembly/datasets/) for more details.

## Model

Shape assembly models usually consist of a point cloud feature extractor (e.g. PointNet), a relationship reasoning module (e.g. GNNs), and a pose predictor (usually implemented as MLPs).
See [model](./model.md) for details about the baselines supported in this codebase.

### Base Model

We implement a `BaseModel` class as an instance of PyTorch-Lightning's `LightningModule`, which support general methods such as `training/validation/test_step/epoch_end()`.
It also implements general loss computation, metrics calculation, and visualization during training.
See [base_model.py](../multi_part_assembly/models/modules/base_model.py).
Below we detail some core methods we implement for all assembly models.

### Assembly Models

All the assembly models inherit from `BaseModel`.
In general, you only need to implement three methods of a new model:

-   `__init__()`: initialize all the model components such as feature extractors, pose predictors
-   `forward()`: the input to this function is the `data_dict` from the dataloader, which contains part point clouds and other items specified by you.
    The model needs to leverage these inputs to predict two items:

    -   `rot`: rotation of each parts.
        Shape `[batch_size, num_parts, 4]` if using quaternion or `[batch_size, num_parts, 3, 3]` if using rotation matrix
    -   `trans`: translation of each parts.
        Shape `[batch_size, num_parts, 3]`.

    Once the output dictionary contains these two values, the loss, metrics and visualization code in `BaseModel` can run smoothly

-   `_loss_function()`: this function applies some pre-/post-processing of the model input-output and loss computation.
    For example, you can specify the inputs to `model.forward()` by constructing `forward_dict` from `data_dict`.
    Or reuse some features calculated in previous samples

## Loss

Common loss terms include:

-   MSE between predicted and ground-truth translations
-   Cosine loss for rotation, i.e. `|<q1, q2> - 1|_2` for quaternion or `|R1^T @ R2 - I|_2` for rotation matrix
-   L2/Chamfer distance between point clouds transformed by predicted and ground-truth rotations and translations

### Semantic Assembly

Since there are multiple plausible assembly solutions for a set of parts, we adopt the MoN loss sampling mechanism from DGL.
See Section 3.4 of their [paper](https://arxiv.org/pdf/2006.07793.pdf) for more details.

Besides, since there are often geometrically equivalent parts in a shape (e.g. 4 legs of a chair), we perform a matching step to minimize the loss.
This is similar to the [Bipartite Matching](<https://en.wikipedia.org/wiki/Matching_(graph_theory)>) used in [DETR](https://arxiv.org/pdf/2005.12872.pdf).
See `_match_parts()` method of `BaseModel` class.

### Geometric Assembly

Usually, there is no geometrically equivalent parts in this setting, which means we do not need to match GT for loss computation.

**Experimental**:

However, sometimes it is hard to define a canonical pose for objects due to symmetry.
In semantic assembly, for example for a chair, it is easy to align the chair's back with the negative X-axis and define this as the canonical pose for all chairs.
However in geometric assembly, the canonical pose for a e.g. cylinder bottle/vase is ill-defined.

Therefore, we borrow idea from the part matching step in semantic assembly and develop a rotation matching step to minimize the loss.
We rotate the ground-truth object along Z-axis for different angles and select one as the new ground-truth, mitigating the effect of rotation symmetry ambiguity.
See `_match_rotation()` method of `BaseModel` class.

If you want to enable this matching step, change `_C.num_rot` under `loss` field to a positive int value larger than 1.
E.g. if set `_C.num_rot = 12`, then the ground-truth object will be rotated 12 times along Z-axis (30 degree each time), and compare with the predicted assembly.

## Metrics

For semantic assembly, we adopt Shape Chamfer Distance (SCD), Part Accuracy (PA) and Connectivity Accuracy (CA).
Please refer to Section 4.3 of the [paper](https://arxiv.org/pdf/2006.07793.pdf) for more details.

For geometric assembly, we adopt SCD and PA, as well as MSE/RMSE/MAE between translations and rotations.
Please refer to Section 6.1 of the [paper](https://arxiv.org/pdf/2205.14886.pdf) for more details.

**Experimental**:

-   As discussed above, these metrics are sometimes problematic due to the symmetry ambiguity.
    E.g., if we rotate a cylinder bottle by 5 degree along Z-axis, it is still perfectly assembled.
    However, the MSE of both rotation and translation will give a non-zero error (even if we do rotation matching, if 5 degree is smaller than the granularity we check, the errors will not be eliminated).
    Therefore, we propose to compute the relative pose errors.
    Suppose there are 5 parts in a shape.
    Every time we treat 1 part as canonical, and calculate the relative poses between the other 4 parts to it.
    We repeat this process for 5 times, and take the min error as the final result.
    This is because relative poses between parts will not be affected by global shape transformations.
    See `relative_pose_metrics()` function in [eval_utils.py](../multi_part_assembly/utils/eval_utils.py).
-   As pointed out by some papers (e.g. [this](https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf)), MSE between rotations is not a good metric.
    Therefore, we adopt the geodesic distance between two rotations as another metric.
    See `rot_geodesic_dist()` function in [eval_utils.py](../multi_part_assembly/utils/eval_utils.py).

## Rotation Representation

-   We use real part first (w, x, y, z) quaternion in this codebase following [PyTorch3D](https://pytorch3d.org/), while `scipy` use real part last format.
    Please be careful when using the code
-   For ease of data batching, we always represent rotations as quaternions from the dataloaders.
    However, to build a compatible interface for util functions, model input-output, we wrap the predicted rotations in a `Rotation3D` class, which supports common format conversion and tensor operations.
    See [rotation.py](../multi_part_assembly/utils/rotation.py) for detailed definitions
-   Rotation representations we support (change `_C.rot_type` under `model` field to use different rotation representations):
    -   Quaternion (`quat`), by default
    -   6D representation (rotation matrix, `rmat`): see CVPR'19 [paper](https://zhouyisjtu.github.io/project_rotation/rotation.html).
        The predicted `6`-len tensor will be reshaped to `(2, 3)`, and the third row is obtained via cross product.
        Then, the 3 vectors will be stacked along the `-2`-th dim.
        In a `Rotation3D` object, the 6D representation will be converted to a 3x3 rotation matrix
