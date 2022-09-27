# Prerequisites

## Python Packages

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment setup.
Please install [PyTorch](https://pytorch.org/) and [PyTorch3D](https://pytorch3d.org/) manually.
Below is an example script installing PyTorch with CUDA 11.3 (please make sure the CUDA version matches your machine, as we will compile custom ops later):

```
conda create -n assembly python=3.8
conda activate assembly
# pytorch
conda install pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch
# pytorch-lightning
conda install pytorch-lightning=1.6.2
# pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```

You can use `nvcc --version` to see the CUDA version of your machine.
**Note that the current code is only tested under PyTorch 1.10, and PyTorch 1.11 will fail due to changes to header files**.

Finally, install other related packages and this package via:

```
pip install -e .
```

## Custom Ops

Install custom CUDA ops for Chamfer distance and PointNet modules:

1. Go to `multi_part_assembly/utils/chamfer` and run `pip install -e .`
2. Go to `multi_part_assembly/models/modules/encoder/pointnet2/pointnet2_ops_lib` and run `pip install -e .`

If you meet any errors, make sure your PyTorch version <= 1.10.1 and your nvcc version is the same as the CUDA version that PyTorch is compiled for (`cudatoolkit` version from conda).

## Data Preparation

The codebase currently supports two assembly datasets:

-   PartNet is a semantic assembly dataset, where each shape (furniture) is decomposed to semantically meaningful parts (e.g. chair legs, backs and arms). We adopt the pre-processed data provided by [DGL](https://github.com/hyperplane-lab/Generative-3D-Part-Assembly). Please follow their [instructions](https://github.com/hyperplane-lab/Generative-3D-Part-Assembly#file-structure) to download the data in `.npy` format.
-   Breaking-Bad is a geometric assembly dataset, where each shape breaks down to several fractures without clear semantics. Please follow their [instructions](https://github.com/Breaking-Bad-Dataset/Breaking-Bad-Dataset.github.io/blob/main/README.md) to process the data. The main experiments are conducted on the `everyday` and `artifact` subsets. The `other` subset is very large (~900G) so you may exclude it.

After downloading and processing all the data, please modify the `_C.data_dir` key in the config files under `configs/_base_/datasets`.

## Troubleshooting

Please refer to [faq.md](faq.md) for a list of potential errors.
