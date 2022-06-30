import torch

from multi_part_assembly.models import BaseModel
from multi_part_assembly.models import build_encoder, StocasticPoseRegressor

from .transformer import TransformerEncoder


class PNTransformer(BaseModel):
    """PointNet-Transformer based multi-part assembly model.

    Encoder: PointNet extracting per-part global point cloud features
    Correlator: TransformerEncoder perform part interactions
    Predictor: MLP-based pose predictor
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.encoder = self._init_encoder()
        self.corr_module = self._init_corr_module()
        self.pose_predictor = self._init_pose_predictor()

    def _init_encoder(self):
        """Part point cloud encoder."""
        encoder = build_encoder(
            self.cfg.model.encoder,
            feat_dim=self.pc_feat_dim,
            global_feat=True,
        )
        return encoder

    def _init_corr_module(self):
        """Part feature interaction module."""
        corr_module = TransformerEncoder(
            d_model=self.pc_feat_dim,
            num_heads=self.cfg.model.transformer_heads,
            ffn_dim=self.cfg.model.transformer_feat_dim,
            num_layers=self.cfg.model.transformer_layers,
            norm_first=self.cfg.model.transformer_pre_ln,
        )
        return corr_module

    def _init_pose_predictor(self):
        """Final pose estimator."""
        # concat feature, instance_label and noise as input
        dim = self.pc_feat_dim
        if self.semantic:  # instance_label in semantic assembly
            dim += self.max_num_part
        if self.use_part_label:
            dim += self.cfg.data.num_part_category
        pose_predictor = StocasticPoseRegressor(
            feat_dim=dim,
            noise_dim=self.cfg.loss.noise_dim,
            rot_type=self.rot_type,
        )
        return pose_predictor

    def _extract_part_feats(self, part_pcs, part_valids):
        """Extract per-part point cloud features."""
        B, P, N, _ = part_pcs.shape  # [B, P, N, 3]
        valid_mask = (part_valids == 1)
        # shared-weight encoder
        valid_pcs = part_pcs[valid_mask]  # [n, N, 3]
        valid_feats = self.encoder(valid_pcs)  # [n, C]
        pc_feats = torch.zeros(B, P, self.pc_feat_dim).type_as(valid_feats)
        pc_feats[valid_mask] = valid_feats
        return pc_feats

    def forward(self, data_dict):
        """Forward pass to predict poses for each part.

        Args:
            data_dict should contains:
                - part_pcs: [B, P, N, 3]
                - part_valids: [B, P], 1 are valid parts, 0 are padded parts
                - part_label: [B, P, NUM_PART_CATEGORY] when using as input
                    otherwise [B, P, 0] just a placeholder for compatibility
                - instance_label: [B, P, P (0 in geometric assembly)]
            may contains:
                - pre_pose_feats: [B, P, C'] (reused) or None
        """
        feats = data_dict.get('pre_pose_feats', None)

        if feats is None:
            part_pcs = data_dict['part_pcs']
            part_valids = data_dict['part_valids']
            pc_feats = self._extract_part_feats(part_pcs, part_valids)
            # transformer feature fusion
            valid_mask = (part_valids == 1)
            corr_feats = self.corr_module(pc_feats, valid_mask)  # [B, P, C]
            # MLP predict poses
            part_label = data_dict['part_label'].type_as(corr_feats)
            inst_label = data_dict['instance_label'].type_as(corr_feats)
            feats = torch.cat([corr_feats, part_label, inst_label], dim=-1)
        rot, trans = self.pose_predictor(feats)
        rot = self._wrap_rotation(rot)

        pred_dict = {
            'rot': rot,  # [B, P, 4/(3, 3)], Rotation3D
            'trans': trans,  # [B, P, 3]
            'pre_pose_feats': feats,  # [B, P, C']
        }
        return pred_dict

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        """Predict poses and calculate loss.

        Since there could be several parts that are the same in one shape, we
            need to do Hungarian matching to find the min loss values.

        Args:
            data_dict: the data loaded from dataloader
            pre_pose_feats: because the stochasticity is only in the final pose
                regressor, we can reuse all the computed features before

        Returns a dict of loss, each is a [B] shape tensor for later selection.
        See GNN Assembly paper Sec 3.4, the MoN loss is sampling prediction
            several times and select the min one as final loss.
            Also returns computed features before pose regressing for reusing.
        """
        part_pcs, valids = data_dict['part_pcs'], data_dict['part_valids']
        forward_dict = {
            'part_pcs': part_pcs,
            'part_valids': valids,
            'part_label': data_dict['part_label'],
            'instance_label': data_dict['instance_label'],
            'pre_pose_feats': out_dict.get('pre_pose_feats', None),
        }

        # prediction
        out_dict = self.forward(forward_dict)
        pre_pose_feats = out_dict['pre_pose_feats']

        # loss computation
        loss_dict, out_dict = self._calc_loss(out_dict, data_dict)
        out_dict['pre_pose_feats'] = pre_pose_feats

        return loss_dict, out_dict
