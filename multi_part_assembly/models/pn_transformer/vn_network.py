import torch

from multi_part_assembly.models import build_encoder, VNPoseRegressor

from .network import PNTransformer
from .transformer import CanonicalVNTransformerEncoder


class VNPNTransformer(PNTransformer):
    """SO(3) equivariant PointNet-Transformer based multi-part assembly model.

    This model should only be used in geometric assembly, and use rotation
        matrix as the rotation representation.
    This is because 1) the 6d rotation representation can be parametrized as
        (2, 3) matrix, which is compatible with the (C, 3) shape in VN models
    2) in semantic assembly we also input instance label and random noise,
        which cannot preserve rotation equivariance.

    Encoder: VNPointNet extracting per-part global point cloud features
    Correlator: VNTransformerEncoder perform part interactions
    Predictor: VN MLP for rotation and VN-In MLP for translation
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # see the above class docstring
        assert self.rot_type == 'rmat', 'VNPNTransformer should predict rmat'
        assert not self.semantic, 'VNPNTransformer is for geometric assembly'

    def _init_encoder(self):
        """Part point cloud encoder."""
        encoder = build_encoder(
            self.cfg.model.encoder,
            feat_dim=self.pc_feat_dim,
            global_feat=True,
            pool1=self.cfg.model.get('encoder_pool1', 'mean'),
            pool2=self.cfg.model.get('encoder_pool2', 'max'),
        )
        return encoder

    def _init_corr_module(self):
        """Part feature interaction module."""
        corr_module = CanonicalVNTransformerEncoder(
            d_model=self.pc_feat_dim,
            num_heads=self.cfg.model.transformer_heads,
            num_layers=self.cfg.model.transformer_layers,
            dropout=0.,
        )
        return corr_module

    def _init_pose_predictor(self):
        """Final pose estimator."""
        # only use feature as input to in VN models
        assert self.cfg.loss.noise_dim == 0
        pose_predictor = VNPoseRegressor(
            feat_dim=self.pc_feat_dim,
            rot_type=self.rot_type,
        )
        return pose_predictor

    def _extract_part_feats(self, part_pcs, part_valids):
        """Extract per-part point cloud features."""
        B, P, N, _ = part_pcs.shape  # [B, P, N, 3]
        valid_mask = (part_valids == 1)
        # shared-weight encoder
        valid_pcs = part_pcs[valid_mask]  # [n, N, 3]
        valid_feats = self.encoder(valid_pcs)  # [n, C, 3]
        pc_feats = torch.zeros(B, P, self.pc_feat_dim, 3).type_as(valid_feats)
        pc_feats[valid_mask] = valid_feats
        return pc_feats

    def forward(self, data_dict):
        """Forward pass to predict poses for each part.

        Args:
            data_dict should contains:
                - part_pcs: [B, P, N, 3]
                - part_valids: [B, P], 1 are valid parts, 0 are padded parts
            may contains:
                - pre_pose_feats: [B, P, C', 3] (reused) or None
        """
        feats = data_dict.get('pre_pose_feats', None)

        if feats is None:
            part_pcs = data_dict['part_pcs']
            part_valids = data_dict['part_valids']
            pc_feats = self._extract_part_feats(part_pcs, part_valids)
            # transformer feature fusion
            # [B, P, C, 3] --> [B, C, 3, P]
            pc_feats = pc_feats.permute(0, 2, 3, 1).contiguous()
            valid_mask = (part_valids == 1)  # [B, P]
            corr_feats = self.corr_module(pc_feats, valid_mask)  # [B, C, 3, P]
            # MLP predict poses
            # [B, C, 3, P] --> [B, P, C, 3]
            feats = corr_feats.permute(0, 3, 1, 2).contiguous()
        rot, trans = self.pose_predictor(feats)
        rot = self._wrap_rotation(rot)

        pred_dict = {
            'rot': rot,  # [B, P, 4/(3, 3)], Rotation3D
            'trans': trans,  # [B, P, 3]
            'pre_pose_feats': feats,  # [B, P, C', 3]
        }
        return pred_dict
