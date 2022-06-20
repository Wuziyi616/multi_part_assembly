"""Code borrowed from https://github.com/hyperplane-lab/Generative-3D-Part-Assembly/blob/main/exps/Our_Method-dynamic_graph_learning/models/model_dynamic.py"""

import numpy as np

import torch
import torch.nn as nn

from multi_part_assembly.utils import _get_clones
from multi_part_assembly.models import BaseModel
from multi_part_assembly.models import build_encoder, StocasticPoseRegressor

from .modules import MLP3, MLP4, RelationNet, PoseEncoder


class DGLModel(BaseModel):
    """Dynamic GNN based multi-part assembly model (`DGL`).

    From paper: https://arxiv.org/pdf/2006.07793.pdf

    Encoder: PointNet extracting global & part point cloud feature
    GNN: Dynamic GNN performs message passing, relation reasoning
    Predictor: MLP-based pose predictor
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.iter = self.cfg.model.gnn_iter

        self.encoder = self._init_encoder()
        self.mlp3s = self._init_mlp3s()
        self.mlp4s = self._init_mlp4s()
        self.pose_predictors = self._init_pose_predictor()
        self.relation_predictor = RelationNet()
        self.relation_predictor_dense = RelationNet()
        self.pose_extractor = PoseEncoder()

    def _init_encoder(self):
        """Part point cloud encoder."""
        encoder = build_encoder(
            self.cfg.model.encoder,
            feat_dim=self.pc_feat_dim,
            global_feat=True,
        )
        return encoder

    def _init_mlp3s(self):
        """MLP3 in GNN message passing."""
        mlp3s = nn.ModuleList(
            [MLP3(self.pc_feat_dim) for _ in range(self.iter)])
        return mlp3s

    def _init_mlp4s(self):
        """MLP3 in GNN node feature aggregation."""
        mlp4s = nn.ModuleList(
            [MLP4(self.pc_feat_dim) for _ in range(self.iter)])
        return mlp4s

    def _init_pose_predictor(self):
        """Final pose estimator."""
        # concat feature, instance_label, last_pose and noise as input
        dim = self.pc_feat_dim + 7
        if self.semantic:  # instance_label in semantic assembly
            dim += self.max_num_part
        if self.use_part_label:
            dim += self.cfg.data.num_part_category
        pose_predictor = StocasticPoseRegressor(
            feat_dim=dim,
            noise_dim=self.cfg.loss.noise_dim,
        )
        pose_predictors = _get_clones(pose_predictor, self.iter)
        return pose_predictors

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
                - instance_label: [B, P, P (0 in geometry assembly)]
                - part_ids: [B, P]
                - valid_matrix: [B, P, P]
            may contains:
                - part_feats: [B, P, C'] (reused) or None
                - class_list: batch of list of list (reused) or None
        """
        part_feats = data_dict.get('part_feats', None)

        if part_feats is None:
            part_pcs = data_dict['part_pcs']
            part_valids = data_dict['part_valids']
            part_feats = self._extract_part_feats(part_pcs, part_valids)
        local_feats = part_feats

        part_label = data_dict['part_label'].type_as(part_feats)
        instance_label = data_dict['instance_label'].type_as(part_feats)
        B, P = instance_label.shape[:2]
        # initialize a fully connected graph
        relation_matrix = data_dict['valid_matrix'].double()
        valid_matrix = relation_matrix  # 1 indicates valid relation
        pred_pose = torch.zeros((B, P, 7)).type_as(part_feats).detach()
        pred_pose[..., 0] = 1.

        # construct same_class_list for GNN node aggregation/separation
        class_list = data_dict.get('class_list', None)
        if class_list is None and self.semantic:
            part_valids = data_dict['part_valids']
            part_ids = data_dict['part_ids']
            class_list = [[] for _ in range(B)]
            for i in range(B):
                class_ids = part_ids[i][part_valids[i] == 1].cpu().numpy()
                for lbl in np.unique(class_ids):
                    class_list[i].append(np.where(class_ids == lbl)[0])

        all_pred_quat, all_pred_trans = [], []
        for iter_ind in range(self.iter):
            # adjust relations
            if iter_ind >= 1:
                pose_feats = self.pose_extractor(pred_pose)  # [B, P, C]
                # merge features of parts in the same class
                if iter_ind % 2 == 1 and self.semantic:
                    pose_feat = pose_feats.clone()
                    part_feats_copy = part_feats.clone()
                    for i in range(B):
                        for cls_lst in class_list[i]:
                            if len(cls_lst) <= 1:
                                continue
                            pose_feat[i, cls_lst] = pose_feats[i, cls_lst].\
                                max(dim=-2, keepdim=True)[0]
                            part_feats_copy[i, cls_lst] = part_feats[
                                i, cls_lst].max(
                                    dim=-2, keepdim=True)[0]
                    # the official implementation performs stop gradient
                    # see https://github.com/hyperplane-lab/Generative-3D-Part-Assembly/blob/main/exps/Our_Method-dynamic_graph_learning/models/model_dynamic.py#L236
                    # we discover that detach sometimes cause unstable training
                    # so we disable it here
                    # part_feats_copy = part_feats_copy.detach()
                else:
                    pose_feat = pose_feats
                    part_feats_copy = part_feats

                # predict new graph relations
                pose_feat1 = pose_feat.unsqueeze(1).repeat(1, P, 1, 1)
                pose_feat2 = pose_feat.unsqueeze(2).repeat(1, 1, P, 1)
                input_relation = torch.cat([pose_feat1, pose_feat2], dim=-1)
                if iter_ind % 2 == 0:
                    new_relation = self.relation_predictor_dense(
                        input_relation.view(B, P * P, -1)).view(B, P, P)
                else:
                    new_relation = self.relation_predictor(
                        input_relation.view(B, P * P, -1)).view(B, P, P)
                relation_matrix = new_relation.double() * valid_matrix
            else:
                part_feats_copy = part_feats

            # mlp3, GNN nodes pairwise interaction
            part_feat1 = part_feats_copy.unsqueeze(2).repeat(1, 1, P, 1)
            part_feat2 = part_feats_copy.unsqueeze(1).repeat(1, P, 1, 1)
            input_3 = torch.cat([part_feat1, part_feat2], dim=-1)
            part_relation = self.mlp3s[iter_ind](input_3.view(B * P, P, -1))
            part_relation = part_relation.view(B, P, P, -1).double()

            # pooling over connected nodes
            part_message = part_relation * relation_matrix.unsqueeze(-1)
            part_message = part_message.sum(dim=2)  # B x P x F
            norm = relation_matrix.sum(dim=-1)  # B x P
            normed_part_message = \
                part_message / (norm.unsqueeze(dim=-1) + 1e-6)

            # mlp4, node aggregation
            input_4 = torch.cat(
                [normed_part_message.type_as(part_feats), part_feats], dim=-1)
            part_feats = self.mlp4s[iter_ind](input_4)  # B x P x F

            # mlp5, pose prediction
            input_5 = torch.cat(
                [part_feats, part_label, instance_label, pred_pose], dim=-1)
            pred_quat, pred_trans = self.pose_predictors[iter_ind](input_5)
            pred_pose = torch.cat([pred_quat, pred_trans], dim=-1)

            # save poses
            all_pred_quat.append(pred_quat)
            all_pred_trans.append(pred_trans)

        if self.training:
            pred_quat = torch.stack(all_pred_quat, dim=0)
            pred_trans = torch.stack(all_pred_trans, dim=0)
        else:
            # directly take the last step results
            pred_quat = all_pred_quat[-1]
            pred_trans = all_pred_trans[-1]

        pred_dict = {
            'quat': pred_quat,  # [(T, )B, P, 4]
            'trans': pred_trans,  # [(T, )B, P, 3]
            'part_feats': local_feats,  # [B, P, C]
            'class_list': class_list,  # batch of list of list
        }
        return pred_dict

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        """Predict poses and calculate loss.

        Since there could be several parts that are the same in one shape, we
            need to do Hungarian matching to find the min loss values.

        Args:
            data_dict: the data loaded from dataloader
            part_feats: reuse part point cloud features
            class_list: pre-computed same class list

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
            'part_ids': data_dict['part_ids'],
            'valid_matrix': data_dict['valid_matrix'],
            'part_feats': out_dict.get('part_feats', None),
            'class_list': out_dict.get('class_list', None),
        }

        # prediction
        out_dict = self.forward(forward_dict)
        part_feats, class_list = out_dict['part_feats'], out_dict['class_list']

        # loss computation
        if not self.training:
            loss_dict, out_dict = self._calc_loss(out_dict, data_dict)
            out_dict['part_feats'] = part_feats
            out_dict['class_list'] = class_list
            return loss_dict, out_dict

        pred_trans, pred_quat = out_dict['trans'], out_dict['quat']
        all_loss_dict = None
        for i in range(self.iter):
            pred_dict = {'quat': pred_quat[i], 'trans': pred_trans[i]}
            loss_dict, out_dict = self._calc_loss(pred_dict, data_dict)
            if all_loss_dict is None:
                all_loss_dict = {k: 0. for k in loss_dict.keys()}
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict[k] + v
                all_loss_dict[f'{k}_{i}'] = v
        out_dict['part_feats'] = part_feats
        out_dict['class_list'] = class_list

        return all_loss_dict, out_dict
