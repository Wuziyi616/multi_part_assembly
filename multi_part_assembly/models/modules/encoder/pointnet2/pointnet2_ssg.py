import torch.nn as nn

from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2SSG(nn.Module):
    """PointNet++ feature extractor.

    Input point clouds [B, N, 3].
    Output global feature [B, feat_dim].
    """

    def __init__(self, feat_dim):
        super().__init__()

        self.feat_dim = feat_dim
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            ))
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            ))
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, self.feat_dim],
                use_xyz=True,
            ))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)  # [B, C, N]

        return features.squeeze(-1)  # [B, C]
