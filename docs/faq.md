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

3. `OSError: /lib/x86_64-linux-gnu/libm.so.6: version 'GLIBC_2.27' not found`

This is an error when importing `open3d`.
Please downgrade it to 0.9 by `pip install open3d==0.9`.
According to this [issue](https://github.com/isl-org/Open3D/issues/1307), this is because the latest `open3d` requires `GLIBC_2.27` file, which only exists in Ubuntu 18.04 (and later versions), not in Ubuntu 16.04.
