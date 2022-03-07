import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseRegressor(nn.Module):
    """MLP-based regressor for translation and quaterion prediction."""

    def __init__(self, feat_dim):
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(feat_dim, 256),
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
        view_shape = list(x.shape[:-1]) + [-1, ]
        f = self.fc_layers(x.view(-1, x.shape[-1])).view(view_shape)
        quat = self.rot_head(f)  # [B, 4] or [B, P, 4]
        quat = F.normalize(quat, p=2, dim=-1)
        trans = self.trans_head(f)  # [B, 3] or [B, P, 3]
        return quat, trans


class StocasticPoseRegressor(PoseRegressor):
    """Stochastic pose regressor with noise injection."""

    def __init__(self, feat_dim, noise_dim):
        super().__init__(feat_dim + noise_dim)

        self.noise_dim = noise_dim

    def forward(self, x):
        """x: [B, C] or [B, P, C]"""
        noise_shape = list(x.shape[:-1]) + [self.noise_dim, ]
        noise = torch.randn(noise_shape).type_as(x)
        x = torch.cat([x, noise], dim=-1)
        return super().forward(x)
