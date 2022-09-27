import torch

from multi_part_assembly.models import build_encoder, VNEqFeature, PoseRegressor

from .network import PNTransformer
from .transformer import TransformerEncoder


class VNPNTransformer(PNTransformer):
    """SO(3) equivariant PointNet-Transformer based multi-part assembly model.

    This model should only be used in geometric assembly, and use rotation
        matrix as the rotation representation.
    This is because 1) the 6d rotation representation can be parametrized as
        (2, 3) matrix, which is compatible with the (C, 3) shape in VN models
    2) in semantic assembly we also input instance label and random noise,
        which cannot preserve rotation equivariance.

    Encoder: VNPointNet extracting per-part global point cloud features. Then,
        we canonicalize part features and store the canonicalization factor.
    Correlator: vanilla TransformerEncoder perform part interactions. Since the
        input features are canonicalized (i.e. invariant to transformation),
        the output features are also canonicalized.
    Predictor: vanilla PoseRegressor predicting rotation and translation.
        The predicted translation is thus invariant, which is what we want.
        The predicted rotation is also invariant, but we want it to be
        equivariant to input part rotation, so we need to apply the inverse
        of the stored canonicalization factor to it.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # see the above class docstring
        assert self.rot_type == 'rmat', 'VNPNTransformer should predict rmat'
        assert not self.semantic, 'VNPNTransformer is for geometric assembly'

        self.feats_can = VNEqFeature(
            self.pc_feat_dim, dim=4, use_rmat=self.cfg.model.rmat_can)

    def _init_encoder(self):
        """Part point cloud encoder."""
        encoder = build_encoder(
            self.cfg.model.encoder,
            feat_dim=self.pc_feat_dim,
            global_feat=True,
            pool1='mean',
            pool2='max',
        )
        return encoder

    def _init_corr_module(self):
        """Part feature interaction module."""
        corr_module = TransformerEncoder(
            d_model=self.pc_feat_dim * 3,
            num_heads=self.cfg.model.transformer_heads,
            ffn_dim=self.pc_feat_dim * 3 * 4,
            num_layers=self.cfg.model.transformer_layers,
            norm_first=True,
            dropout=0.,
        )
        return corr_module

    def _init_pose_predictor(self):
        """Final pose estimator."""
        # only use feature as input to in VN models
        assert self.cfg.loss.noise_dim == 0
        pose_predictor = PoseRegressor(
            feat_dim=self.pc_feat_dim * 3,
            rot_type=self.rot_type,
            norm_rot=self.cfg.model.rmat_can,  # use rotation matrix in can
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
                - pre_pose_feats: [B, P, C'*3] (reused) or None
        """
        feats = data_dict.get('pre_pose_feats', None)
        assert feats is None

        part_pcs = data_dict['part_pcs']
        part_valids = data_dict['part_valids']
        pc_feats = self._extract_part_feats(part_pcs, part_valids)
        # [B, P, C, 3] --> [B, C, 3, P]
        pc_feats = pc_feats.permute(0, 2, 3, 1).contiguous()
        pc_feats = self.feats_can(pc_feats)  # [B, C, 3, P], invariant
        # to [B, P, C*3]
        pc_feats = pc_feats.flatten(1, 2).transpose(1, 2).contiguous()
        # transformer feature fusion
        valid_mask = (part_valids == 1)  # [B, P]
        feats = self.corr_module(pc_feats, valid_mask)  # [B, P, C*3]
        rot, trans = self.pose_predictor(feats)  # [B, P, 6], [B, P, 3]
        # translation prediction is invariant, which is what we want
        # we need to make rotation prediction equivariant
        # to [B, 2, 3, P]
        rot = rot.transpose(1, 2).unflatten(1, (2, 3)).contiguous()
        rot = self.feats_can(rot)  # [B, 2, 3, P], equivariant
        rot = rot.permute(0, 3, 1, 2).contiguous()  # [B, P, 2, 3]
        rot = self._wrap_rotation(rot)

        pred_dict = {
            'rot': rot,  # [B, P, 4/(3, 3)], Rotation3D
            'trans': trans,  # [B, P, 3]
            'pre_pose_feats': feats,  # [B, P, C', 3]
        }
        return pred_dict
