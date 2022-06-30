"""Code borrowed from https://github.com/hyperplane-lab/Generative-3D-Part-Assembly/blob/main/exps/Our_Method-dynamic_graph_learning/models/model_dynamic.py"""

import numpy as np

import torch

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
        self.merge_node = self.cfg.model.merge_node

        self.encoder = self._init_encoder()
        self.edge_mlps = self._init_edge_mlps()
        self.node_mlps = self._init_node_mlps()
        self.pose_predictors = self._init_pose_predictor()
        self.relation_predictor_dense = RelationNet()
        if self.merge_node:
            self.relation_predictor = RelationNet()
        self.pose_extractor = PoseEncoder(self.pose_dim)

    def _init_encoder(self):
        """Part point cloud encoder."""
        encoder = build_encoder(
            self.cfg.model.encoder,
            feat_dim=self.pc_feat_dim,
            global_feat=True,
        )
        return encoder

    def _init_edge_mlps(self):
        """MLP in GNN calculating edge features."""
        edge_mlp = MLP3(self.pc_feat_dim)
        edge_mlps = _get_clones(edge_mlp, self.iter)
        return edge_mlps

    def _init_node_mlps(self):
        """MLP in GNN performing node feature aggregation."""
        node_mlp = MLP4(self.pc_feat_dim)
        node_mlps = _get_clones(node_mlp, self.iter)
        return node_mlps

    def _init_pose_predictor(self):
        """Final pose estimator."""
        # concat feature, instance_label, last_pose and noise as input
        dim = self.pc_feat_dim + self.pose_dim
        if self.semantic:  # instance_label in semantic assembly
            dim += self.max_num_part
        if self.use_part_label:
            dim += self.cfg.data.num_part_category
        pose_predictor = StocasticPoseRegressor(
            feat_dim=dim,
            noise_dim=self.cfg.loss.noise_dim,
            rot_type=self.rot_type,
        )
        pose_predictors = _get_clones(pose_predictor, self.iter)
        return pose_predictors

    def _gather_same_class(self, data_dict):
        """Construct same_class_list for GNN node aggregation/separation."""
        class_list = data_dict.get('class_list', None)  # pre-computed
        if self.merge_node and self.semantic and class_list is None:
            part_valids = data_dict['part_valids']
            part_ids = data_dict['part_ids']
            B = part_valids.shape[0]
            class_list = [[] for _ in range(B)]
            for i in range(B):
                class_ids = part_ids[i][part_valids[i] == 1].cpu().numpy()
                for lbl in np.unique(class_ids):
                    class_list[i].append(np.where(class_ids == lbl)[0])
        return class_list

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

    def _merge_nodes(self, part_feats, pose_feats, class_list):
        """Merge geometrically equivalent nodes in GNN."""
        B = part_feats.shape[0]
        pose_feats_copy = pose_feats.clone()
        part_feats_copy = part_feats.clone()
        for i in range(B):
            for cls_lst in class_list[i]:
                if len(cls_lst) <= 1:
                    continue
                pose_feats_copy[i, cls_lst] = pose_feats[i, cls_lst].\
                    max(dim=-2, keepdim=True)[0]
                part_feats_copy[i, cls_lst] = \
                    part_feats[i, cls_lst].max(dim=-2, keepdim=True)[0]
        # the official implementation performs stop gradient
        # see https://github.com/hyperplane-lab/Generative-3D-Part-Assembly/blob/main/exps/Our_Method-dynamic_graph_learning/models/model_dynamic.py#L236
        # we discover that detach sometimes cause unstable training
        # so we disable it here
        # part_feats_copy = part_feats_copy.detach()
        return part_feats_copy, pose_feats_copy

    def _update_relation(self, pose_feats, iter_ind):
        """Update GNN nodes relation using pose features."""
        B, P, _ = pose_feats.shape
        pose_feat1 = pose_feats.unsqueeze(1).repeat(1, P, 1, 1)
        pose_feat2 = pose_feats.unsqueeze(2).repeat(1, 1, P, 1)
        input_relation = torch.cat([pose_feat1, pose_feat2], dim=-1)
        if self.merge_node and iter_ind % 2 == 1:
            new_relation = self.relation_predictor(
                input_relation.view(B, P * P, -1)).view(B, P, P)
        else:
            new_relation = self.relation_predictor_dense(
                input_relation.view(B, P * P, -1)).view(B, P, P)
        return new_relation

    def _message_passing(self, part_feats, relation_matrix, iter_ind):
        """Perform one step of message passing, get per-node messages."""
        B, P, _ = part_feats.shape
        # GNN nodes pairwise interaction
        part_feat1 = part_feats.unsqueeze(2).repeat(1, 1, P, 1)
        part_feat2 = part_feats.unsqueeze(1).repeat(1, P, 1, 1)
        pairwise_feats = torch.cat([part_feat1, part_feat2], dim=-1)
        part_relation = \
            self.edge_mlps[iter_ind](pairwise_feats.view(B * P, P, -1))
        part_relation = part_relation.view(B, P, P, -1)

        # compute message as weighted sum over edge features
        part_message = part_relation * relation_matrix.unsqueeze(-1)
        part_message = part_message.sum(dim=2)  # B x P x F
        norm = relation_matrix.sum(dim=-1, keepdim=True)  # B x P x 1
        normed_part_message = part_message / (norm + 1e-6)

        return normed_part_message

    def forward(self, data_dict):
        """Forward pass to predict poses for each part.

        Args:
            data_dict should contains:
                - part_pcs: [B, P, N, 3]
                - part_valids: [B, P], 1 are valid parts, 0 are padded parts
                - part_label: [B, P, NUM_PART_CATEGORY] when using as input
                    otherwise [B, P, 0] just a placeholder for compatibility
                - instance_label: [B, P, P (0 in geometric assembly)]
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

        # initialize a fully connected graph
        valid_matrix = data_dict['valid_matrix']  # 1 indicates valid relation
        part_label = data_dict['part_label'].type_as(part_feats)
        instance_label = data_dict['instance_label'].type_as(part_feats)
        B, P = instance_label.shape[:2]
        # init pose as identity
        pred_pose = self.zero_pose.repeat(B, P, 1).type_as(part_feats).detach()

        # construct same_class_list for GNN node aggregation/separation
        class_list = self._gather_same_class(data_dict)

        all_pred_rot, all_pred_trans = [], []
        for iter_ind in range(self.iter):
            # adjust relations
            if iter_ind >= 1:
                pose_feats = self.pose_extractor(pred_pose)  # B x P x F
                # merge features of parts in the same class
                if self.merge_node and self.semantic and iter_ind % 2 == 1:
                    part_feats_copy, pose_feats_copy = self._merge_nodes(
                        part_feats, pose_feats, class_list)
                else:
                    part_feats_copy = part_feats
                    pose_feats_copy = pose_feats

                # predict new graph relations
                new_relation = self._update_relation(pose_feats_copy, iter_ind)
                relation_matrix = new_relation * valid_matrix
            # first iter, use fully-connected graph
            else:  # iter_ind == 0
                part_feats_copy = part_feats
                relation_matrix = valid_matrix

            # perform message passing
            messages = self._message_passing(part_feats_copy, relation_matrix,
                                             iter_ind)  # B x P x F

            # GNN node aggregation
            node_feats = torch.cat([messages.type_as(part_feats), part_feats],
                                   dim=-1)
            part_feats = self.node_mlps[iter_ind](node_feats)  # B x P x F

            # pose prediction
            pose_feats = torch.cat(
                [part_feats, part_label, instance_label, pred_pose], dim=-1)
            pred_rot, pred_trans = self.pose_predictors[iter_ind](pose_feats)
            pred_pose = torch.cat([pred_rot, pred_trans], dim=-1)

            # save poses
            all_pred_rot.append(pred_rot)
            all_pred_trans.append(pred_trans)

        if self.training:
            pred_rot = self._wrap_rotation(torch.stack(all_pred_rot, dim=0))
            pred_trans = torch.stack(all_pred_trans, dim=0)
        else:
            # directly take the last step results
            pred_rot = self._wrap_rotation(all_pred_rot[-1])
            pred_trans = all_pred_trans[-1]

        pred_dict = {
            'rot': pred_rot,  # [(T, )B, P, 4/(3, 3)], Rotation3D
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

        pred_trans, pred_rot = out_dict['trans'], out_dict['rot']
        all_loss_dict = None
        for i in range(self.iter):
            pred_dict = {'rot': pred_rot[i], 'trans': pred_trans[i]}
            loss_dict, out_dict = self._calc_loss(pred_dict, data_dict)
            if all_loss_dict is None:
                all_loss_dict = {k: 0. for k in loss_dict.keys()}
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict[k] + v
                all_loss_dict[f'{k}_{i}'] = v
        out_dict['part_feats'] = part_feats
        out_dict['class_list'] = class_list

        return all_loss_dict, out_dict
