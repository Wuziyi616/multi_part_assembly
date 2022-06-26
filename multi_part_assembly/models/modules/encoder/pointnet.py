import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_part_assembly.models import VNLinear, VNBatchNorm, VNMaxPool, \
    VNLinearBNLeakyReLU


class PointNet(nn.Module):
    """PointNet feature extractor.

    Input point clouds [B, N, 3].
    Output per-point feature [B, N, feat_dim] or global feature [B, feat_dim].
    """

    def __init__(self, feat_dim, global_feat=True):
        super().__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, feat_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(feat_dim)

        self.global_feat = global_feat

    def forward(self, x):
        """x: [B, N, 3]"""
        x = x.transpose(2, 1).contiguous()  # [B, 3, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))  # [B, feat_dim, N]
        if self.global_feat:
            feat = x.max(dim=-1)[0]  # [B, feat_dim]
        else:
            feat = x.transpose(2, 1).contiguous()  # [B, N, feat_dim]
        return feat


def knn(x, k):
    """x: [B, C, N]"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [B, N, k]
    return idx


def vn_get_graph_feature(x, k=20, idx=None):
    """x: [B, C, 3, N]"""
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)  # [B, C*3, N]
    if idx is None:
        idx = knn(x, k=k)  # [B, N, k]
    device = x.device

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base  # [B, N, k]

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()  # [B, N, C*3]
    feature = x.view(batch_size * num_points, -1)[idx, :]  # [B, N, k, C*3]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)  # [B, N, k, C, 3]

    feature = torch.cat((feature - x, x, cross),
                        dim=3).permute(0, 3, 4, 1, 2).contiguous()
    # [B, 3C, 3, N, k]
    return feature


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNPointNet(nn.Module):
    """VNN-based rotation Equivariant PointNet feature extractor.

    Input point clouds [B, N, 3].
    Output per-point feature [B, N, feat_dim, 3] or
        global feature [B, feat_dim, 3].
    """

    def __init__(self, feat_dim, global_feat=True, **kwargs):
        super().__init__()

        self.conv1 = VNLinearBNLeakyReLU(3, 64, dim=5, negative_slope=0.0)
        self.conv2 = VNLinearBNLeakyReLU(64, 64, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearBNLeakyReLU(64, 64, dim=4, negative_slope=0.0)
        self.conv4 = VNLinearBNLeakyReLU(64, 128, dim=4, negative_slope=0.0)
        self.conv5 = VNLinear(128, feat_dim)
        self.bn5 = VNBatchNorm(feat_dim, dim=4)

        pool1 = kwargs.get('pool1', 'mean')  # in-knn pooling
        self.pool1 = self._build_pooling(pool1)
        pool2 = kwargs.get('pool2', 'max')  # final global_feats pooling
        self.pool2 = self._build_pooling(pool2)

        self.global_feat = global_feat

    @staticmethod
    def _build_pooling(pooling):
        if pooling == 'max':
            pool = VNMaxPool(64 // 3)
        elif pooling == 'mean':
            pool = mean_pool
        else:
            raise NotImplementedError(f'{pooling}-pooling not implemented')
        return pool

    def forward(self, x):
        """x: [B, N, 3]"""
        x = x.transpose(2, 1).contiguous()  # [B, 3, N]

        x = x.unsqueeze(1)  # [B, 1, 3, N]
        feat = vn_get_graph_feature(x)  # [B, 3, 3, N, k]
        x = self.conv1(feat)  # [B, C, 3, N, k]
        x = self.pool(x)  # [B, C, 3, N]

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.bn5(self.conv5(x))  # [B, feat_dim, 3, N]

        if self.global_feat:
            feat = self.pool(x)  # [B, feat_dim, 3]
        else:
            feat = x.permute(0, 3, 1, 2).contiguous()  # [B, N, feat_dim, 3]
        return feat
