"""Code borrowed from: https://github.com/FlyingGiraffe/vnn-pc"""

import torch
import torch.nn as nn

EPS = 1e-6


class VNLinear(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C_in, 3, N, ...]

        Returns:
            [B, C_out, 3, N, ...]
        """
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNBatchNorm(nn.Module):

    def __init__(self, num_features, dim):
        super().__init__()

        self.dim = dim
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

    def __init__(self,
                 in_channels,
                 share_nonlinearity=False,
                 negative_slope=0.2):
        super().__init__()

        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C, 3, N, ...]

        Returns:
            features of the same shape after LeakyReLU
        """
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNReLU(VNLeakyReLU):

    def __init__(self, in_channels, share_nonlinearity=False):
        super().__init__(in_channels, share_nonlinearity, negative_slope=0.)


class VNLinearBNLeakyReLU(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dim=5,
                 share_nonlinearity=False,
                 negative_slope=0.2):
        super().__init__()

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        self.leaky_relu = VNLeakyReLU(
            out_channels,
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
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        p = self.leaky_relu(p)
        return p


class VNMaxPool(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C, 3, N]

        Returns:
            [B, C, 3], features after max-pooling
        """
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
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
        B, C = x.shape[0], x.shape[-2]
        ori_shape = x.shape
        x = x.transpose(-1, 1).reshape(B, -1, 3, C)
        x = self.ln(x)
        x = x.transpose(-1, 1).reshape(ori_shape)
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
        if dim != 3:
            in_channels *= 2  # we will concat x and x.mean()

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
        self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        """
        Args:
            x: point features of shape [B, C, 3, N, ...]

        Returns:
            rotation invariant features of the same shape
        """
        if self.dim in [4, 5]:
            dim = -1 if self.dim == 4 else (-1, -2)
            x_mean = x.mean(dim=dim, keepdim=True).expand(x.size())
            x = torch.cat((x, x_mean), dim=1)

        z = x
        z = self.vn1(z)
        z = self.vn2(z)
        z = self.vn_lin(z.transpose(1, -1)).transpose(1, -1)
        z = z.transpose(1, 2)

        if self.dim == 4:
            x_in = torch.einsum('bijm,bjkm->bikm', x, z)
        elif self.dim == 3:
            x_in = torch.einsum('bij,bjk->bik', x, z)
        elif self.dim == 5:
            x_in = torch.einsum('bijmn,bjkmn->bikmn', x, z)
        else:
            raise NotImplementedError(f'dim={self.dim} is not supported')

        return x_in


""" test code
from multi_part_assembly.utils import random_quaternions, qrot
pc = torch.randn(2, 1000, 3)
quat = random_quaternions(2)
rot_pc = qrot(quat, pc)
pc = pc.reshape(2, 10, 100, 3).permute(0, 1, 3, 2)  # [B, C, 3, N, ...]
rot_pc = rot_pc.reshape(2, 10, 100, 3).permute(0, 1, 3, 2)  # same

model = VNLinear(10, 20)

out = model(pc).transpose(-1, -2).reshape(2, 2000, 3)
rot_out = model(rot_pc).transpose(-1, -2).reshape(2, 2000, 3)
(qrot(quat, out) - rot_out).abs().max()
"""
