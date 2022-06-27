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
from functools import reduce
from scipy.spatial.transform import Rotation as R

import torch

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, \
    quaternion_to_axis_angle, matrix_to_axis_angle, \
    axis_angle_to_quaternion, axis_angle_to_matrix


class Rotation3D:
    """Class for different formats of 3D rotation."""

    ROT_TYPE = ['quat', 'rmat', 'axis']
    ROT_NAME = {
        'quat': 'quaternion',
        'rmat': 'matrix',
        'axis': 'axis_angle',
    }

    def __init__(self, rot, rot_type='quat'):
        self._rot = rot
        self._rot_type = rot_type

        self._check_valid()

    def _check_valid(self):
        """Check the shape of rotation."""
        assert isinstance(self._rot, torch.Tensor)
        assert self._rot_type in self.ROT_TYPE
        if self._rot_type == 'quat':
            assert self._rot.shape[-1] == 4
        elif self._rot_type == 'rmat':
            assert self._rot.shape[-1] == self._rot.shape[-2] == 3
        elif self._rot_type == 'axis':
            assert self._rot.shape[-1] == 3

    def to_rot(self, rot_type):
        assert rot_type in self.ROT_TYPE
        src_type = self.ROT_NAME[self._rot_type]
        dst_type = self.ROT_NAME[rot_type]
        if src_type == dst_type:
            return self.clone()
        new_rot = eval(f'{src_type}_to_{dst_type}')(self._rot)
        return Rotation3D(new_rot, rot_type)

    @property
    def rot(self):
        return self._rot

    @rot.setter
    def rot(self, rot):
        self._rot = rot
        self._check_valid()

    @property
    def rot_type(self):
        return self._rot_type

    def to(self, device):
        self._rot = self._rot.to(device)
        return self

    def type(self, dtype):
        self._rot = self._rot.type(dtype)
        return self

    def type_as(self, other):
        self._rot = self._rot.type_as(other)
        return self

    def clone(self):
        return Rotation3D(self._rot.clone(), self._rot_type)


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


# PyTorch-backed implementations
# quaternion-based transformations


def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1]).type_as(quaternion)
    return quaternion * scaling


def random_quaternions(shape):
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        shape: [N1, N2, ...]

    Returns:
        Quaternions as tensor of shape (N1, N2, ..., 4).
    """
    if isinstance(shape, int):
        shape = [shape]
    else:
        shape = list(shape)
    n = reduce(lambda x, y: x * y, shape)
    o = torch.randn((n, 4))
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o.view(shape + [4])


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qeuler(q, order, epsilon=0, to_degree=False):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    euler = torch.stack((x, y, z), dim=1).view(original_shape)
    if to_degree:
        euler = euler * 180. / np.pi
    return euler


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
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3

    # repeat to e.g. apply the same quat for all points in a point cloud
    # [4] --> [N, 4], [B, 4] --> [B, N, 4], [B, P, 4] --> [B, P, N, 4]
    if len(q.shape) == len(v.shape) - 1:
        q = q.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)

    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


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

    original_shape = list(v.shape)
    r = r.view(-1, 3, 3)
    v = v.view(-1, 3, 1)

    rv = torch.bmm(r, v).view(original_shape)
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


def rot_pc(rot, pc):
    """Rotate the 3D point cloud.

    Args:
        rot (Rotation3D): quat and rmat are supported now.
    """
    if rot.rot_type == 'quat':
        return qrot(rot.rot, pc)
    elif rot.rot_type == 'rmat':
        return rmat_rot(rot.rot, pc)
    else:
        raise NotImplementedError(f'{rot.rot_type} is not supported!')


def transform_pc(trans, rot, pc):
    """Rotate and translate the 3D point cloud.

    Args:
        rot (Rotation3D): quat and rmat are supported now.
    """
    if rot.rot_type == 'quat':
        return qtransform(trans, rot.rot, pc)
    elif rot.rot_type == 'rmat':
        return rmat_transform(trans, rot.rot, pc)
    else:
        raise NotImplementedError(f'{rot.rot_type} is not supported!')


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
