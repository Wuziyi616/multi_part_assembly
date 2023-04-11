import copy
import pickle
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

import torch.nn as nn
from torch.nn import LayerNorm, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


def pickle_load(file, **kwargs):
    if isinstance(file, str):
        with open(file, 'rb') as f:
            obj = pickle.load(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = pickle.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def pickle_dump(obj, file=None, **kwargs):
    kwargs.setdefault('protocol', 2)
    if file is None:
        return pickle.dumps(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'wb') as f:
            pickle.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        pickle.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def save_pc(pc, file):
    """Save point cloud to file.

    Args:
        pc (np.ndarray): [N, 3]
        file (str)
    """
    pcd_df = pd.DataFrame(data=pc, columns=['x', 'y', 'z'])
    pcd_cloud = PyntCloud(pcd_df)
    pcd_cloud.to_file(file)


def colorize_part_pc(part_pc, colors):
    """Colorize part point cloud.

    Args:
        part_pc (np.ndarray): [P, N, 3]
        colors (np.ndarray): [max_num_parts, 3 (RGB)]

    Returns:
        np.ndarray: [P, N, 6]
    """
    P, N, _ = part_pc.shape
    colored_pc = np.zeros((P, N, 6))
    colored_pc[:, :, :3] = part_pc
    for i in range(P):
        colored_pc[i, :, 3:] = colors[i]
    return colored_pc


def array_equal(a, b):
    """Compare if two arrays are the same.

    Args:
        a/b: can be np.ndarray or torch.Tensor.
    """
    if a.shape != b.shape:
        return False
    try:
        assert (a == b).all()
        return True
    except:
        return False


def array_in_list(array, lst):
    """Judge whether an array is in a list."""
    for v in lst:
        if array_equal(array, v):
            return True
    return False


def filter_wd_parameters(model, skip_list=()):
    """Create parameter groups for optimizer.

    We do two things:
        - filter out params that do not require grad
        - exclude bias and Norm layers
    """
    # we need to sort the names so that we can save/load ckps
    w_name, b_name, no_decay_name = [], [], []
    for name, m in model.named_modules():
        # exclude norm weight
        if isinstance(m, (LayerNorm, GroupNorm, _BatchNorm, _InstanceNorm)):
            w_name.append(name)
        # exclude bias
        if hasattr(m, 'bias') and m.bias is not None:
            b_name.append(name)
        if name in skip_list:
            no_decay_name.append(name)
    w_name.sort()
    b_name.sort()
    no_decay_name.sort()
    no_decay = [model.get_submodule(m).weight for m in w_name] + \
        [model.get_submodule(m).bias for m in b_name]
    for name in no_decay_name:
        no_decay += [
            p for p in model.get_submodule(m).parameters()
            if p.requires_grad and not array_in_list(p, no_decay)
        ]

    decay_name = []
    for name, param in model.named_parameters():
        if param.requires_grad and not array_in_list(param, no_decay):
            decay_name.append(name)
    decay_name.sort()
    decay = [model.get_parameter(name) for name in decay_name]
    return {'decay': list(decay), 'no_decay': list(no_decay)}


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
