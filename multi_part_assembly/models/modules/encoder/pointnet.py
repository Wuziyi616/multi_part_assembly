import torch
import torch.nn as nn
import torch.nn.functional as F


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
