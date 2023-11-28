# Multi-Part Shape Assembly

## Introduction

This is an open-source 3D shape assembly codebase based on PyTorch + PyTorch-Lightning, released alongside the NeurIPS 2022 Dataset Paper: [Breaking Bad: A Dataset for Geometric Fracture and Reassembly](https://breaking-bad-dataset.github.io/).

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

## News & Updates

-   2023.2: **BC-breaking change:** we update our mesh data as discussed in [issue#6](https://github.com/Wuziyi616/multi_part_assembly/issues/6).
    This requires you to re-run [the decompression script](https://github.com/Breaking-Bad-Dataset/Breaking-Bad-Dataset.github.io/blob/main/decompress.py) to get a new version of data.
    We have re-run the benchmark results, which are released [here](https://github.com/Wuziyi616/multi_part_assembly/blob/master/docs/model.md#geometric-assembly-with-inner-face-removed-data).
    Overall, the differences are minor.
-   2022.10: Code release.
-   2022.9: The paper is accepted by NeurIPS 2022 Datasets and Benchmarks Track as a **Featured Paper Presentation**!

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
