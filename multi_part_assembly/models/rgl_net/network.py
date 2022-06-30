"""Code borrowed from https://github.com/absdnd/RGL_NET_Progressive_Part_Assembly/blob/main/models/model_Ours.py"""

import torch
import torch.nn as nn

from multi_part_assembly.utils import _get_clones
from multi_part_assembly.models import DGLModel, RNNWrapper

from .modules import MLP4


class RGLNet(DGLModel):
    """Recurrent GNN based multi-part assembly model (`RGL-Net`).

    From paper: https://arxiv.org/pdf/2107.12859.pdf

    Encoder: PointNet extracting global & part point cloud feature
    GNN: GNN performs message passing, relation reasoning
    GRU: Progressive message encoding
    Predictor: MLP-based pose predictor
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.grus = self._init_grus()

    def _init_node_mlps(self):
        """MLP in GNN performing node feature aggregation."""
        # input dim is different from DGL's node_mlp
        # because here it also inputs hidden states from GRU
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

    def _init_gru_hidden(self, B):
        """Random initialize the GRU hidden state."""
        # init forward and reverse hidden states are the same
        rand_vec = torch.randn((1, B, self.pc_feat_dim)).repeat(2, 1, 1)
        zero_vec = torch.randn((2, B, self.pc_feat_dim))
        init_hidden = torch.cat([rand_vec, zero_vec], dim=-1)
        return init_hidden

    def _run_gru(self, part_feats, messages, valids, iter_ind):
        """Run GRU over part features and GNN node messahes."""
        B = part_feats.shape[0]
        gru_inputs = torch.cat([part_feats, messages], dim=-1)  # B x P x 2F
        init_hidden = self._init_gru_hidden(B).type_as(messages)  # 2 x B x 2F
        gru_outputs, _ = self.grus[iter_ind](
            gru_inputs,
            init_hidden,
            valids=valids,
        )  # B x P x 4F
        return gru_outputs

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
                                             iter_ind)

            # GRU progressive message passing
            gru_outputs = self._run_gru(part_feats, messages,
                                        data_dict['part_valids'],
                                        iter_ind)  # B x P x 4F

            # node feature update
            part_feats = self.node_mlps[iter_ind](gru_outputs)  # B x P x F

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
