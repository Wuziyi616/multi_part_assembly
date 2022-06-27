import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseRegressor(nn.Module):
    """MLP-based regressor for translation and rotation prediction."""

    def __init__(self, feat_dim, rot_type='quat'):
        super().__init__()

        if rot_type == 'quat':
            rot_dim = 4
        elif rot_type == 'rmat':
            rot_dim = 6  # 6D representation from the CVPR'19 paper
        else:
            raise NotImplementedError(f'rotation {rot_type} is not supported')
        self.rot_type = rot_type

        self.fc_layers = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )

        # Rotation prediction head
        self.rot_head = nn.Linear(128, rot_dim)

        # Translation prediction head
        self.trans_head = nn.Linear(128, 3)

    def forward(self, x):
        """x: [B, C] or [B, P, C]"""
        view_shape = list(x.shape[:-1]) + [-1]
        f = self.fc_layers(x.view(-1, x.shape[-1])).view(view_shape)
        rot = self.rot_head(f)  # [B, 4/6] or [B, P, 4/6]
        if self.rot_type == 'quat':
            rot = F.normalize(rot, p=2, dim=-1)
        elif self.rot_type == 'rmat':
            # adopted from PyTorch3D's `rotation_6d_to_matrix`
            a1, a2 = rot[..., :3], rot[..., 3:]
            b1 = F.normalize(a1, p=2, dim=-1)
            b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
            b2 = F.normalize(b2, p=2, dim=-1)
            rot = torch.cat([b1, b2], dim=-1)  # back to [..., 6]
        trans = self.trans_head(f)  # [B, 3] or [B, P, 3]
        return rot, trans


class StocasticPoseRegressor(PoseRegressor):
    """Stochastic pose regressor with noise injection."""

    def __init__(self, feat_dim, noise_dim, rot_type='quat'):
        super().__init__(feat_dim + noise_dim, rot_type)

        self.noise_dim = noise_dim

    def forward(self, x):
        """x: [B, C] or [B, P, C]"""
        noise_shape = list(x.shape[:-1]) + [self.noise_dim]
        noise = torch.randn(noise_shape).type_as(x)
        x = torch.cat([x, noise], dim=-1)
        return super().forward(x)
