import torch
import torch.nn as nn


class MLP3(nn.Module):

    def __init__(self, feat_len):
        super().__init__()

        self.conv1 = nn.Conv1d(2 * feat_len, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, feat_len, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(feat_len)

    """
        Input: (B x P) x P x 2F
        Output: (B x P) x P x F
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)

        return x


class MLP4(nn.Module):

    def __init__(self, feat_len):
        super().__init__()

        self.conv1 = nn.Conv1d(2 * feat_len, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, feat_len, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(feat_len)

    """
        Input: (B x P) x P x 2F
        Output: (B x P) x P x F
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)

        return x


class RelationNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Linear(128 + 128, 256)
        self.mlp2 = nn.Linear(256, 512)
        self.mlp3 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = torch.relu(self.mlp2(x))
        x = torch.sigmoid(self.mlp3(x))
        return x


class PoseEncoder(nn.Module):

    def __init__(self, pose_dim):
        super().__init__()
        self.mlp1 = nn.Linear(pose_dim, 256)
        self.mlp2 = nn.Linear(256, 128)

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = torch.relu(self.mlp2(x))
        return x
