import torch
import torch.nn as nn


class MLP4(nn.Module):

    def __init__(self, feat_len):
        super().__init__()

        self.conv1 = nn.Conv1d(4 * feat_len, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, feat_len, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(feat_len)

    """
        Input: B x P x 4F
        Output: B x P x F
    """

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x.permute(0, 2, 1)

        return x
