"""Re-implementation according to the paper."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_part_assembly.utils import _get_clones
from multi_part_assembly.models import BaseModel, DGLModel, RNNWrapper

from .modules import MLP4, RelationNet, PoseEncoder


class RGLNet(DGLModel):
    """Recurrent GNN based multi-part assembly model (`RGL-Net`).

    From paper: https://arxiv.org/pdf/2107.12859.pdf

    Encoder: PointNet extracting global & part point cloud feature
    GNN: GNN performs message passing, relation reasoning
    GRU: Progressive message encoding
    Predictor: MLP-based pose predictor
    """

    def __init__(self, cfg):
        BaseModel.__init__(self, cfg)

        self.iter = self.cfg.model.gnn_iter

        self.encoder = self._init_encoder()
        self.edge_mlps = self._init_edge_mlps()
        self.node_mlps = self._init_node_mlps()
        self.pose_predictors = self._init_pose_predictor()
        self.relation_predictor = RelationNet()
        self.pose_extractor = PoseEncoder()
        self.grus = self._init_grus()

    def _init_node_mlps(self):
        """MLP in GNN performing node feature aggregation."""
        node_mlp = MLP4(self.pc_feat_dim)
        node_mlps = _get_clones(node_mlp, self.iter)
        return node_mlps

    def _init_grus(self):
        """GRU module for progressive message encoding."""
        gru = nn.GRU(
            input_size=self.pc_feat_dim * 2,
            hidden_size=self.pc_feat_dim * 2,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        gru = RNNWrapper(gru, batch_first=True)
        grus = _get_clones(gru, self.iter)
        return grus

    def _rand_gru_hidden(self, B):
        """Random initialize the GRU hidden state."""
        # init forward and reverse hidden states are the same
        rand_vec = torch.randn((1, B, self.pc_feat_dim)).repeat(2, 1, 1)
        zero_vec = torch.randn((2, B, self.pc_feat_dim))
        init_hidden = torch.cat([rand_vec, zero_vec], dim=-1)
        return init_hidden

    def forward(self, data_dict):
        """Forward pass to predict poses for each part.

        Args:
            data_dict shoud contains:
                - part_pcs: [B, P, N, 3]
                - part_valids: [B, P], 1 are valid parts, 0 are padded parts
                - part_label: [B, P, NUM_PART_CATEGORY] when using as input
                    otherwise [B, P, 0] just a placeholder for compatibility
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

        valid_matrix = data_dict['valid_matrix']
        part_label = data_dict['part_label'].type_as(part_feats)
        instance_label = data_dict['instance_label'].type_as(part_feats)
        B, P = instance_label.shape[:2]
        # initialize identity poses
        pred_pose = torch.zeros((B, P, 7)).type_as(part_feats).detach()
        pred_pose[..., 0] = 1.

        all_pred_quat, all_pred_trans = [], []
        for iter_ind in range(self.iter):
            # compute weights between pairs of parts
            if iter_ind >= 1:
                pose_feats = self.pose_extractor(pred_pose)  # B x P x F
                pose_feat1 = pose_feats.unsqueeze(2).repeat(1, 1, P, 1)
                pose_feat2 = pose_feats.unsqueeze(1).repeat(1, P, 1, 1)
                input_relation = torch.cat([pose_feat1, pose_feat2], dim=-1)
                edge_weights = self.relation_predictor(
                    input_relation.view(B, P * P, -1)).view(B, P, P)
            else:
                edge_weights = torch.ones((B, P, P)).type_as(part_feats)
            # masked out padded parts
            edge_weights = edge_weights.masked_fill(valid_matrix == 0,
                                                    float('-inf'))
            edge_weights = F.softmax(edge_weights, dim=-1)  # B x P x P
            # avoid nan
            edge_weights = edge_weights.masked_fill(valid_matrix == 0, 0.)

            # GNN nodes pairwise interaction
            part_feat1 = part_feats.unsqueeze(2).repeat(1, 1, P, 1)
            part_feat2 = part_feats.unsqueeze(1).repeat(1, P, 1, 1)
            pairwise_feats = torch.cat([part_feat1, part_feat2], dim=-1)
            part_relation = \
                self.edge_mlps[iter_ind](pairwise_feats.view(B * P, P, -1))
            part_relation = part_relation.view(B, P, P, -1)  # B x P x P x F

            # compute message as weighted sum over edge features, B x P x F
            messages = (edge_weights.unsqueeze(-1) * part_relation).sum(2)

            # GRU progressive message passing
            # TODO: pack part sequences to handle padded parts?
            gru_inputs = \
                torch.cat([part_feats, messages], dim=-1)  # B x P x 2F
            init_hidden = \
                self._rand_gru_hidden(B).type_as(gru_inputs)  # 2 x B x 2F
            gru_outputs, _ = self.grus[iter_ind](
                gru_inputs,
                init_hidden,
                valids=data_dict['part_valids'],
            )  # B x P x 4F

            # node feature update
            part_feats = self.node_mlps[iter_ind](gru_outputs)  # B x P x F

            # pose prediction
            pose_feats = torch.cat(
                [part_feats, part_label, instance_label, pred_pose], dim=-1)
            pred_quat, pred_trans = self.pose_predictors[iter_ind](pose_feats)
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
            'class_list': None,  # keep for compatibility
        }
        return pred_dict
