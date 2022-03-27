"""Adopted from https://github.com/AnTao97/dgcnn.pytorch/blob/master/model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """x: [B, C, N]"""
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx


def get_graph_feature(x, k=20):
    """x: [B, C, N]"""
    idx = knn(x, k=k)   # (batch_size, num_points, k)

    batch_size, num_dims, num_points = x.size()
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = (idx + idx_base).view(-1)

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    # batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)


class DGCNN(nn.Module):
    """DGCNN feature extractor.

    Input point clouds [B, N, 3].
    Output per-point feature [B, N, feat_dim] or global feature [B, feat_dim].
    """

    def __init__(self, feat_dim, global_feat=True):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(feat_dim)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, feat_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.global_feat = global_feat
        if global_feat:
            self.out_fc = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, x):
        """x: [B, N, 3]"""
        x = x.transpose(2, 1).contiguous()  # [B, 3, N]
        batch_size = x.size(0)
        x = get_graph_feature(x)   # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)          # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1)[0]      # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)          # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1)[0]      # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)          # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1)[0]      # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)          # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1)[0]      # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)          # (batch_size, 64+64+128+256, num_points) -> (batch_size, feat_dim, num_points)

        if self.global_feat:
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (batch_size, feat_dim, num_points) -> (batch_size, feat_dim)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # (batch_size, feat_dim, num_points) -> (batch_size, feat_dim)
            x = torch.cat((x1, x2), 1)              # (batch_size, feat_dim*2)
            feat = self.out_fc(x)  # [B, feat_dim]
        else:
            feat = x.transpose(2, 1).contiguous()  # [B, N, feat_dim]

        return feat
