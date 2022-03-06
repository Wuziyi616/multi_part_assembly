import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    """MLP-based regressor for translation and quaterion prediction."""

    def __init__(self, pc_feat_dim):
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(pc_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )

        # Rotation prediction head
        self.rot_head = nn.Linear(128, 4)

        # Translation prediction head
        self.trans_head = nn.Linear(128, 3)

    def forward(self, x):
        """x: [B, C] or [B, P, C]"""
        f = self.fc_layers(x)
        quat = self.rot_head(f)  # [B, 4] or [B, P, 4]
        quat = F.normalize(quat, p=2, dim=-1)
        trans = self.trans_head(f)  # [B, 3] or [B, P, 3]
        return quat, trans
