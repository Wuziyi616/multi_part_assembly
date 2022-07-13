"""Code borrowed from: https://github.com/FlyingGiraffe/vnn-pc"""

import torch
import torch.nn as nn

EPS = 1e-6


def conv1x1(in_channels, out_channels, dim):
    if dim == 3:
        return nn.Conv1d(in_channels, out_channels, 1, bias=False)
    elif dim == 4:
        return nn.Conv2d(in_channels, out_channels, 1, bias=False)
    elif dim == 5:
        return nn.Conv3d(in_channels, out_channels, 1, bias=False)
    else:
        raise NotImplementedError(f'{dim}D 1x1 Conv is not supported')


class VNLinear(nn.Module):

    def __init__(self, in_channels, out_channels, dim):
        super().__init__()

        self.map_to_feat = conv1x1(in_channels, out_channels, dim)

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C_in, 3, N, ...]

        Returns:
            [B, C_out, 3, N, ...]
        """
        x_out = self.map_to_feat(x)
        return x_out


class VNBatchNorm(nn.Module):

    def __init__(self, num_features, dim):
        super().__init__()

        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
        else:
            raise NotImplementedError(f'{dim}D is not supported')

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C, 3, N, ...]

        Returns:
            features of the same shape after BN along C-dim
        """
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        return x


class VNLeakyReLU(nn.Module):

    def __init__(
        self,
        in_channels,
        dim,
        share_nonlinearity=False,
        negative_slope=0.2,
    ):
        super().__init__()

        if share_nonlinearity:
            self.map_to_dir = conv1x1(in_channels, 1, dim=dim)
        else:
            self.map_to_dir = conv1x1(in_channels, in_channels, dim=dim)
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C, 3, N, ...]

        Returns:
            features of the same shape after LeakyReLU
        """
        d = self.map_to_dir(x)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = d.pow(2).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNReLU(VNLeakyReLU):

    def __init__(self, in_channels, dim, share_nonlinearity=False):
        super().__init__(
            in_channels,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=0.,
        )


class VNLinearBNLeakyReLU(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dim=5,
        share_nonlinearity=False,
        negative_slope=0.2,
    ):
        super().__init__()

        self.linear = VNLinear(in_channels, out_channels, dim=dim)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        self.leaky_relu = VNLeakyReLU(
            out_channels,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C_in, 3, N, ...]

        Returns:
            [B, C_out, 3, N, ...]
        """
        # Linear
        p = self.linear(x)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        p = self.leaky_relu(p)
        return p


class VNMaxPool(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.map_to_dir = conv1x1(in_channels, in_channels, dim=4)

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C, 3, N]

        Returns:
            [B, C, 3], features after max-pooling
        """
        d = self.map_to_dir(x)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j)
                                      for j in x.size()[:-1]]) + (idx, )
        x_max = x[index_tuple]
        return x_max


class VNLayerNorm(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.ln = nn.LayerNorm(num_features)

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C, 3, N, ...]

        Returns:
            features of the same shape after LN in each instance
        """
        norm = torch.norm(x, dim=2) + EPS  # [B, C, N, ...]
        norm_ln = self.ln(norm.transpose(1, -1)).transpose(1, -1)
        norm = norm.unsqueeze(2)
        norm_ln = norm_ln.unsqueeze(2)
        x = x / norm * norm_ln
        return x


class VNInFeature(nn.Module):
    """VN-Invariant layer."""

    def __init__(
        self,
        in_channels,
        dim=4,
        share_nonlinearity=False,
        negative_slope=0.2,
    ):
        super().__init__()

        self.dim = dim
        self.vn1 = VNLinearBNLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn2 = VNLinearBNLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn_lin = conv1x1(in_channels // 4, 3, dim=dim)

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C, 3, N, ...]

        Returns:
            rotation invariant features of the same shape
        """
        z = self.vn1(x)
        z = self.vn2(z)
        z = self.vn_lin(z)
        z = z.transpose(1, 2).contiguous()

        if self.dim == 4:
            x_in = torch.einsum('bijm,bjkm->bikm', x, z)
        elif self.dim == 3:
            x_in = torch.einsum('bij,bjk->bik', x, z)
        elif self.dim == 5:
            x_in = torch.einsum('bijmn,bjkmn->bikmn', x, z)
        else:
            raise NotImplementedError(f'dim={self.dim} is not supported')

        return x_in


class VNEqFeature(VNInFeature):
    """Map VN-IN features back to their original rotation."""

    def forward(self, x, x_in):
        """
        Args:
            x: point features of shape [B, C, 3, N, ...]
            x_in: rotation invariant features of shape [B, C, 3, N, ...]

        Returns:
            rotation equivariant features with x mapped from x_in
        """
        z = self.vn1(x)
        z = self.vn2(z)
        z = self.vn_lin(z)
        # z = z.transpose(1, 2).contiguous()

        if self.dim == 4:
            x_eq = torch.einsum('bijm,bjkm->bikm', x_in, z)
        elif self.dim == 3:
            x_eq = torch.einsum('bij,bjk->bik', x_in, z)
        elif self.dim == 5:
            x_eq = torch.einsum('bijmn,bjkmn->bikmn', x_in, z)
        else:
            raise NotImplementedError(f'dim={self.dim} is not supported')

        return x_eq


""" test code
import torch
from multi_part_assembly.models import VNLayerNorm
from multi_part_assembly.utils import random_rotation_matrixs
vn_ln = VNLayerNorm(16)
pc = torch.rand(2, 16, 3, 100)
rmat = random_rotation_matrixs(2)
rot_pc = rmat[:, None] @ pc
ln_pc = vn_ln(pc)
rot_ln_pc = rmat[:, None] @ ln_pc
ln_rot_pc = vn_ln(rot_pc)
(rot_ln_pc - ln_rot_pc).abs().max()
"""
