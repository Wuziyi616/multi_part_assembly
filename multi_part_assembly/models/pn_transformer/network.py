import torch

from multi_part_assembly.models import BaseModel
from multi_part_assembly.models import build_encoder, StocasticPoseRegressor
from multi_part_assembly.utils import trans_l2_loss, rot_points_cd_loss, \
    shape_cd_loss, calc_part_acc, calc_connectivity_acc

from .transformer import TransformerEncoder


class PNTransformer(BaseModel):
    """Baseline multi-part assembly model.

    Encoder: PointNet extracting per-part global point cloud features
    Correlator: TransformerEncoder perform part interactions
    Predictor: MLP-based pose predictor
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.max_num_part = self.cfg.data.max_num_part
        self.pc_feat_dim = self.cfg.model.pc_feat_dim

        # loss configs
        self.sample_iter = self.cfg.loss.sample_iter

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
        pose_predictor = StocasticPoseRegressor(
            feat_dim=self.pc_feat_dim + self.max_num_part,
            noise_dim=self.cfg.model.noise_dim,
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
            data_dict shoud contains:
                - part_pcs: [B, P, N, 3]
                - part_valids: [B, P], 1 are valid parts, 0 are padded parts
                - instance_label: [B, P, P]
            may contains:
                - pre_pose_feats: [B, P, C'] (reused) or None
        """
        feats = data_dict.get('pre_pose_feats', None)

        if feats is None:
            part_pcs = data_dict['part_pcs']
            part_valids = data_dict['part_valids']
            inst_label = data_dict['instance_label']
            pc_feats = self._extract_part_feats(part_pcs, part_valids)
            # transformer feature fusion
            valid_mask = (part_valids == 1)
            corr_feats = self.corr_module(pc_feats, valid_mask)  # [B, P, C]
            # MLP predict poses
            inst_label = inst_label.type_as(corr_feats)
            feats = torch.cat([corr_feats, inst_label], dim=-1)
        quat, trans = self.pose_predictor(feats)

        pred_dict = {
            'quat': quat,  # [B, P, 4]
            'trans': trans,  # [B, P, 3]
            'pre_pose_feats': feats,  # [B, P, C']
        }
        return pred_dict

    def _calc_loss(self, out_dict, data_dict):
        """Calculate loss by matching GT to prediction."""
        pred_trans, pred_quat = out_dict['trans'], out_dict['quat']

        # matching GT with predictions for lowest loss
        part_pcs, valids = data_dict['part_pcs'], data_dict['part_valids']
        gt_trans, gt_quat = data_dict['part_trans'], data_dict['part_quat']
        match_ids = data_dict['match_ids']
        new_trans, new_quat = self._match_parts(part_pcs, pred_trans,
                                                pred_quat, gt_trans, gt_quat,
                                                match_ids)

        # computing loss
        trans_loss = trans_l2_loss(pred_trans, new_trans, valids)
        rot_pt_cd_loss = rot_points_cd_loss(part_pcs, pred_quat, new_quat,
                                            valids)
        transform_pt_cd_loss, gt_trans_pts, pred_trans_pts = \
            shape_cd_loss(
                part_pcs,
                pred_trans,
                new_trans,
                pred_quat,
                new_quat,
                valids,
                ret_pts=True)
        loss_dict = {
            'trans_loss': trans_loss,
            'rot_pt_cd_loss': rot_pt_cd_loss,
            'transform_pt_cd_loss': transform_pt_cd_loss,
        }  # all loss are of shape [B]

        # in eval, we also want to compute part_acc and connectivity_acc
        if not self.training:
            loss_dict['part_acc'] = calc_part_acc(part_pcs, pred_trans,
                                                  new_trans, pred_quat,
                                                  new_quat, valids)
            if 'contact_points' in data_dict.keys():
                loss_dict['connectivity_acc'] = calc_connectivity_acc(
                    pred_trans, pred_quat, data_dict['contact_points'])

        # return some intermediate variables for reusing
        out_dict = {
            'pred_trans': pred_trans,  # [B, P, 3]
            'pred_quat': pred_quat,  # [B, P, 4]
            'gt_trans_pts': gt_trans_pts,  # [B, P, N, 3]
            'pred_trans_pts': pred_trans_pts,  # [B, P, N, 3]
        }

        return loss_dict, out_dict

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
        instance_label = data_dict['instance_label']
        forward_dict = {
            'part_pcs': part_pcs,
            'part_valids': valids,
            'instance_label': instance_label,
            'pre_pose_feats': out_dict.get('pre_pose_feats', None),
        }

        # prediction
        out_dict = self.forward(forward_dict)
        pre_pose_feats = out_dict['pre_pose_feats']

        # loss computation
        loss_dict, out_dict = self._calc_loss(out_dict, data_dict)
        out_dict['pre_pose_feats'] = pre_pose_feats

        return loss_dict, out_dict
