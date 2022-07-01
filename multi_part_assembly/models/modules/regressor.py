import torch
import torch.nn as nn
import torch.nn.functional as F

from .vnn import VNLinear, VNLeakyReLU, VNInFeature


def normalize_rot6d(rot):
    """Adopted from PyTorch3D.

    Args:
        rot: [..., 6] or [..., 2, 3]

    Returns:
        same shape where the first two 3-dim are normalized and orthogonal
    """
    if rot.shape[-1] == 3:
        unflatten = True
        rot = rot.flatten(-2, -1)
    else:
        unflatten = False
    a1, a2 = rot[..., :3], rot[..., 3:]
    b1 = F.normalize(a1, p=2, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, p=2, dim=-1)
    rot = torch.cat([b1, b2], dim=-1)  # back to [..., 6]
    if unflatten:
        rot = rot.unflatten(-1, (2, 3))
    return rot


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
        f = self.fc_layers(x)
        rot = self.rot_head(f)  # [B, 4/6] or [B, P, 4/6]
        if self.rot_type == 'quat':
            rot = F.normalize(rot, p=2, dim=-1)
        elif self.rot_type == 'rmat':
            rot = normalize_rot6d(rot)
        trans = self.trans_head(f)  # [B, 3] or [B, P, 3]
        return rot, trans


class VNPoseRegressor(nn.Module):
    """PoseRegressor for VN models.

    Target rotation should be rotation-equivariant, while target translation
        should be rotation-invariant.
    """

    def __init__(self, feat_dim, rot_type='rmat'):
        super().__init__()

        assert rot_type == 'rmat', 'VN model only supports rotation matrix'

        # for rotation
        self.vn_fc_layers = nn.Sequential(
            VNLinear(feat_dim, 256, dim=3),
            VNLeakyReLU(256, dim=3, negative_slope=0.2),
            VNLinear(256, 128, dim=3),
            VNLeakyReLU(128, dim=3, negative_slope=0.2),
        )

        # Rotation prediction head
        # we use the 6D representation from the CVPR'19 paper
        self.rot_head = VNLinear(128, 2, dim=3)  # [2, 3] --> 6

        # for translation
        self.in_feats = VNInFeature(feat_dim, dim=3)
        self.fc_layers = nn.Sequential(
            nn.Linear(feat_dim * 3, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )

        # Translation prediction head
        self.trans_head = nn.Linear(128, 3)

    def forward(self, x):
        """x: [B, C, 3] or [B, P, C, 3]"""
        unflatten = len(x.shape) == 4
        B, C = x.shape[0], x.shape[-2]
        x = x.view(-1, C, 3)
        # rotation
        rot_x = self.vn_fc_layers(x)  # [N, 128, 3]
        rot = self.rot_head(rot_x)  # [N, 2, 3]
        rot = normalize_rot6d(rot)  # [N, 2, 3]
        # translation
        trans_x = self.in_feats(x).flatten(-2, -1)  # [N, C*3]
        trans_x = self.fc_layers(trans_x)  # [N, 128]
        trans = self.trans_head(trans_x)  # [N, 3]
        # back to [B, P]
        if unflatten:
            rot = rot.unflatten(0, (B, -1))
            trans = trans.unflatten(0, (B, -1))
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
