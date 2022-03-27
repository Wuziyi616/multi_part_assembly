import copy

import torch
import torch.nn as nn

from multi_part_assembly.models import StocasticPoseRegressor

from .network import PNTransformer
from .transformer import TransformerEncoder


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PosEncoder(nn.Module):
    """Learnable positional encoding model."""

    def __init__(self, dims):
        super().__init__()

        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        pos_enc = self.layers(x)
        return pos_enc


class PNTransformerRefine(PNTransformer):
    """Baseline multi-part assembly model with iterative refinement."""

    def __init__(self, cfg):
        self.refine_steps = cfg.model.refine_steps
        self.pose_pc_feat = cfg.model.pose_pc_feat

        super().__init__(cfg)

        self.corr_pos_enc = self._init_corr_pos_enc()

    def _init_corr_pos_enc(self):
        """Positional encoding for `corr_module`."""
        # as in DGL, we reuse the same PE for all refinement steps
        pos_enc_dims = self.cfg.model.transformer_pos_enc
        corr_pos_enc = PosEncoder(pos_enc_dims)
        return corr_pos_enc

    def _init_corr_module(self):
        """Part feature interaction module."""
        # as in DGL, different model for each refinement step
        corr_module = TransformerEncoder(
            d_model=self.pc_feat_dim,
            num_heads=self.cfg.model.transformer_heads,
            ffn_dim=self.cfg.model.transformer_feat_dim,
            num_layers=self.cfg.model.transformer_layers,
            norm_first=self.cfg.model.transformer_pre_ln,
            out_dim=self.pc_feat_dim,
        )
        corr_modules = _get_clones(corr_module, self.refine_steps)
        return corr_modules

    def _init_pose_predictor(self):
        """Final pose estimator."""
        # concat feature, instance_label and noise as input
        dim = self.pc_feat_dim + self.max_num_part + 7
        if self.pose_pc_feat:
            dim += self.pc_feat_dim
        pose_predictor = StocasticPoseRegressor(
            feat_dim=dim,
            noise_dim=self.cfg.model.noise_dim,
        )
        pose_predictors = _get_clones(pose_predictor, self.refine_steps)
        return pose_predictors

    def forward(self, data_dict):
        """Forward pass to predict poses for each part.

        Args:
            data_dict shoud contains:
                - part_pcs: [B, P, N, 3]
                - part_valids: [B, P], 1 are valid parts, 0 are padded parts
                - instance_label: [B, P, P]
            may contains:
                - pc_feats: [B, P, C] (reused) or None
        """
        pc_feats = data_dict.get('pc_feats', None)
        part_pcs, part_valids = data_dict['part_pcs'], data_dict['part_valids']
        if pc_feats is None:
            pc_feats = self._extract_part_feats(part_pcs, part_valids)

        part_feats = pc_feats
        inst_label = data_dict['instance_label'].type_as(pc_feats)
        B, P, _ = inst_label.shape
        pose = torch.cat([torch.ones(B, P, 1), torch.zeros(B, P, 6)], dim=-1)
        pose = pose.type_as(pc_feats)

        pred_quat, pred_trans = [], []
        for i in range(self.refine_steps):
            # transformer feature fusion
            # positional encoding from predicted pose
            pos_enc = self.corr_pos_enc(pose)
            # directly add positional encoding as in ViT
            in_feats = part_feats + pos_enc
            # transformer feature fusion, [B, P, C]
            valid_mask = (part_valids == 1)
            corr_feats = self.corr_module[i](in_feats, valid_mask)
            # MLP predict poses
            if self.pose_pc_feat:
                feats = torch.cat([pc_feats, corr_feats, inst_label, pose],
                                  dim=-1)
            else:
                feats = torch.cat([corr_feats, inst_label, pose], dim=-1)
            quat, trans = self.pose_predictor[i](feats)
            pred_quat.append(quat)
            pred_trans.append(trans)

            # update for next iteration
            pose = torch.cat([quat, trans], dim=-1)
            part_feats = corr_feats

        if self.training:
            pred_quat = torch.stack(pred_quat, dim=0)
            pred_trans = torch.stack(pred_trans, dim=0)
        else:
            # directly take the last step results
            pred_quat = pred_quat[-1]
            pred_trans = pred_trans[-1]

        pred_dict = {
            'quat': pred_quat,  # [(T, )B, P, 4]
            'trans': pred_trans,  # [(T, )B, P, 3]
            'pc_feats': pc_feats,  # [B, P, C]
        }
        return pred_dict

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        """Predict poses and calculate loss."""
        part_pcs, valids = data_dict['part_pcs'], data_dict['part_valids']
        instance_label = data_dict['instance_label']
        forward_dict = {
            'part_pcs': part_pcs,
            'part_valids': valids,
            'instance_label': instance_label,
            'pc_feats': out_dict.get('pc_feats', None),
        }

        # prediction
        out_dict = self.forward(forward_dict)
        pc_feats = out_dict['pc_feats']

        # loss computation
        if not self.training:
            loss_dict, out_dict = self._calc_loss(out_dict, data_dict)
            out_dict['pc_feats'] = pc_feats
            return loss_dict, out_dict

        pred_trans, pred_quat = out_dict['trans'], out_dict['quat']
        all_loss_dict = None
        for i in range(self.refine_steps):
            pred_dict = {'quat': pred_quat[i], 'trans': pred_trans[i]}
            loss_dict, out_dict = self._calc_loss(pred_dict, data_dict)
            if all_loss_dict is None:
                all_loss_dict = {k: 0. for k in loss_dict.keys()}
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict[k] + v
                all_loss_dict[f'{k}_{i}'] = v

        return all_loss_dict, out_dict
