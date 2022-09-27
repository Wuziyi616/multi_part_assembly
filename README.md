# Multi Part Assembly

## Introduction

This is an open-source 3D shape assembly codebase based on PyTorch + PyTorch-Lightning.

### Major Features

We mainly focus on vision based 3D shape assembly task, which takes in point clouds of multiple parts from an object, and predicts their rotations and translations of the correct assembly.

-   Support datasets:
    -   [PartNet](https://partnet.cs.stanford.edu/) (semantic assembly)
    -   [Breaking-Bad](https://breaking-bad-dataset.github.io/) (geometric assembly)
-   Support models:
    -   Global, LSTM, DGL ([NeurIPS 2020](https://arxiv.org/pdf/2006.07793.pdf))
    -   RGL-NET ([WACV 2022](https://arxiv.org/pdf/2107.12859.pdf))
    -   Transformer-based (designed by us)

We carefully benchmark the models to match their performance in their original papers, which you can easily extend to new datasets.
You can also leverage our codebase to develop your new shape assembly algorithms.

## Installation

Please refer to [install.md](docs/install.md) for step-by-step guidance on how to install the packages and prepare the data.

## Config System

To learn about the config system used in this codebase, please refer to the [config.md](docs/config.md).

## Quick Start

We provide detailed usage of the codebase in [usage.md](docs/usage.md).
You can train, test and visualize the results using our provided config files, or develop your new methods.

## Tutorial

We explain our code design in [tutorial.md](docs/tutorial.md).
Please read it before modifying the codebase or implementing your new algorithms.

## Benchmark

We benchmark the baselines and report them in [model.md](docs/model.md).
We also present detailed instructions on how to reproduce our results.

## License

This project is released under the [MIT license](LICENSE).

## Acknowledgement

We thank the authors of the following repos for open-sourcing their wonderful works:

-   [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): general project structure
-   [PyTorch3D](https://github.com/facebookresearch/pytorch3d): 3D transformation implementation
-   [DGL](https://github.com/hyperplane-lab/Generative-3D-Part-Assembly): shape assembly model code
-   [RGL-NET](https://github.com/absdnd/RGL_NET_Progressive_Part_Assembly): shape assembly model code

We also thank the authors of all the packages we use.
We appreciate all the contributors as well as users who give valuable feedbacks.
We wish this codebase could serve as a benchmark and a flexible toolkit for researchers to re-implement existing algorithms and develop their own new shape assembly methods.
