"""
Transformation functions. Adopted from:
- https://github.com/hyperplane-lab/Generative-3D-Part-Assembly
- https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms
"""

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from scipy.spatial.transform import Rotation as R

import torch

from pytorch3d.transforms import quaternion_invert, quaternion_apply, \
    quaternion_raw_multiply
from pytorch3d.transforms import random_quaternions as _random_quaternions
from pytorch3d.transforms import matrix_to_quaternion, matrix_to_axis_angle, \
    quaternion_to_matrix, quaternion_to_axis_angle, \
    axis_angle_to_quaternion, axis_angle_to_matrix

from .rotation import Rotation3D

# PyTorch-backed implementations
# quaternion-based transformations


def random_quaternions(shape):
    """
    Generate random quaternions representing rotations,
        i.e. versors with nonnegative real part.

    This extends PyTorch3D's implementation with arbitrary shape.

    Args:
        shape: [N1, N2, ...]

    Returns:
        Quaternions as tensor of shape (N1, N2, ..., 4).
    """
    assert isinstance(shape, (int, list, tuple))
    if isinstance(shape, int):
        shape = [shape]
    else:
        shape = list(shape)
    num_quat = np.prod(shape)
    quat = _random_quaternions(num_quat)
    return quat.view(shape + [4])


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any
        number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    return quaternion_raw_multiply(q, r)


def qrmat(q):
    """
    Convert quaternion(s) q to rotation matrix(s).
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3, 3).
    """
    assert q.shape[-1] == 4
    return quaternion_to_matrix(q)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    # repeat to e.g. apply the same quat for all points in a point cloud
    # [4] --> [N, 4], [B, 4] --> [B, N, 4], [B, P, 4] --> [B, P, N, 4]
    if len(q.shape) == len(v.shape) - 1:
        q = q.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)
    assert q.shape[:-1] == v.shape[:-1]
    return quaternion_apply(q, v)


def qtransform(t, q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q,
        and then translate it by the translation described by t.
    Expects a tensor of shape (*, 3) for t, a tensor of shape (*, 4) for q and
        a tensor of shape (*, 3) for v, where * denotes any dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert t.shape[-1] == 3

    # repeat to e.g. apply the same trans for all points in a point cloud
    # [3] --> [N, 3], [B, 3] --> [B, N, 3], [B, P, 3] --> [B, P, N, 3]
    if len(t.shape) == len(v.shape) - 1:
        t = t.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)

    assert t.shape == v.shape

    qv = qrot(q, v)
    tqv = qv + t
    return tqv


def qtransform_invert(t, q, tqv):
    """Reverse transformation of (t, q)"""
    assert t.shape[-1] == 3
    if len(t.shape) == len(tqv.shape) - 1:
        t = t.unsqueeze(-2).repeat_interleave(tqv.shape[-2], dim=-2)

    assert t.shape == tqv.shape

    qv = tqv - t
    q_inv = quaternion_invert(q)
    v = qrot(q_inv, qv)
    return v


# rmat-based transformations


def random_rotation_matrixs(shape):
    """
    Generate random rotation matrixs representing rotations.

    We apply quat2rmat on random quaternions.

    Args:
        shape: [N1, N2, ...]

    Returns:
        Rotation matrixs as tensor of shape (N1, N2, ..., 3, 3).
    """
    quat = random_quaternions(shape)
    return quaternion_to_matrix(quat)


def rmatq(r):
    """
    Convert quaternion(s) q to rotation matrix(s).
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3, 3).
    """
    assert r.shape[-1] == r.shape[-2] == 3
    return matrix_to_quaternion(r)


def rmat_rot(r, v):
    """
    Rotate vector(s) v about the rotation described by rmat(s) r.
    Expects a tensor of shape (*, 3, 3) for r and a tensor of
        shape (*, 3) for v, where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert r.shape[-1] == r.shape[-2] == 3
    assert v.shape[-1] == 3

    # repeat to e.g. apply the same quat for all points in a point cloud
    if len(r.shape) == len(v.shape):
        r = r.unsqueeze(-3).repeat_interleave(v.shape[-2], dim=-3)

    assert r.shape[:-2] == v.shape[:-1]

    rv = (r @ v.unsqueeze(-1)).squeeze(-1)
    return rv


def rmat_transform(t, r, v):
    """
    Rotate vector(s) v about the rotation described by rmat(s) r,
        and then translate it by the translation described by t.
    Expects a tensor of shape (*, 3) for t, a tensor of shape (*, 3, 3) for q
        and a tensor of shape (*, 3) for v, where * denotes any dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert t.shape[-1] == 3

    # repeat to e.g. apply the same trans for all points in a point cloud
    if len(t.shape) == len(v.shape) - 1:
        t = t.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)

    assert t.shape == v.shape

    rv = rmat_rot(r, v)
    trv = rv + t
    return trv


# wrapper on arbitrary 3D rotation format


def rot_pc(rot, pc, rot_type=None):
    """Rotate the 3D point cloud.

    If `rot_type` is specified, `rot` is torch.Tensor. Otherwise, it is a
        Rotation object and the type will be inferred from it.

    Args:
        rot (Rotation3D or torch.Tensor): quat and rmat are supported now.
    """
    if rot_type is None:
        assert isinstance(rot, Rotation3D)
        r = rot.rot
        rot_type = rot.rot_type
    else:
        assert isinstance(rot, torch.Tensor)
        r = rot
    if rot_type == 'quat':
        return qrot(r, pc)
    elif rot_type == 'rmat':
        return rmat_rot(r, pc)
    else:
        raise NotImplementedError(f'{rot.rot_type} is not supported')


def transform_pc(trans, rot, pc, rot_type=None):
    """Rotate and translate the 3D point cloud.

    If `rot_type` is specified, `rot` is torch.Tensor. Otherwise, it is a
        Rotation object and the type will be inferred from it.

    Args:
        rot (Rotation3D or torch.Tensor): quat and rmat are supported now.
    """
    if rot_type is None:
        assert isinstance(rot, Rotation3D)
        r = rot.rot
        rot_type = rot.rot_type
    else:
        assert isinstance(rot, torch.Tensor)
        r = rot
    if rot_type == 'quat':
        return qtransform(trans, r, pc)
    elif rot_type == 'rmat':
        return rmat_transform(trans, r, pc)
    else:
        raise NotImplementedError(f'{rot_type} is not supported')


# Numpy-backed implementations


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()


def qtransform_np(t, q, v):
    t = torch.from_numpy(t).contiguous()
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qtransform(t, q, v).numpy()


def rmat_rot_np(r, v):
    r = torch.from_numpy(r).contiguous()
    v = torch.from_numpy(v).contiguous()
    return rmat_rot(r, v).numpy()


def rmat_transform_np(t, r, v):
    t = torch.from_numpy(t).contiguous()
    r = torch.from_numpy(r).contiguous()
    v = torch.from_numpy(v).contiguous()
    return rmat_transform(t, r, v).numpy()


def quaternion_to_rmat(quat):
    """quat: [4], (w, i, j, k)"""
    rmat = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    return rmat


def trans_rmat_to_pmat(trans, rmat):
    """Convert translation and rotation matrix to homogeneout matrix."""
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = rmat
    pose_mat[:3, -1] = trans
    return pose_mat


def trans_quat_to_pmat(trans, quat):
    """Convert translation and quaternion to homogeneous matrix."""
    # trans: [3]; quat: [4], (w, i, j, k)
    rmat = quaternion_to_rmat(quat)
    pose_mat = trans_rmat_to_pmat(trans, rmat)
    return pose_mat
