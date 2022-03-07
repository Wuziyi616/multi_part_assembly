import torch.nn as nn

from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG

from .pointnet2_ssg import PointNet2SSG


class PointNet2MSG(PointNet2SSG):
    """PointNet++ feature extractor.

    Input point clouds [B, N, 3].
    Output global feature [B, feat_dim].
    """

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[0, 32, 32, 64], [0, 64, 64, 128], [0, 64, 96, 128]],
                use_xyz=True,
            ))

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=True,
            ))
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, self.feat_dim],
                use_xyz=True,
            ))
