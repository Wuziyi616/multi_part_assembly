# FAQ

We list some potential problems encountered by developers and users, along with their corresponding solutions.
Feel free to enrich this list by opening issues when you encounter new troubles.

1. Error when compiling/importing custom ops

Please make sure your CUDA version (nvcc) matches the PyTorch's CUDA version (cudatoolkit).
Also, the CUDA codes we adopt currently only support PyTorch < 1.11.
Please downgrade to PyTorch 1.10 if necessary.

2. `AttributeError: module 'distutils' has no attribute 'version'`

Try `pip install setuptools==59.5.0`.
See this [issue](https://github.com/pytorch/pytorch/issues/69894#issuecomment-1080635462).

