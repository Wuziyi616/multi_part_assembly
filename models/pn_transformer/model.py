import torch
import torch.nn as nn

from utils import qrot, qtransform, chamfer_distance


class PNTransformerMLP(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data_dict):
        """
        data_dict = {
            'part_pcs': [B, P, N, 3]
            'part_trans': [B, P, 3]
            'part_quat': [B, P, 4]
            'part_valids': [B, P]
            'shape_id': int
            'part_ids': [B, P]
            'instance_label': [B, P, P]
            'match_ids': [B, P]
            'contact_points': [B, P, P, 4]
            'sym': [B, P, 3]
        }
        """

    def train_loss_function(self, data_dict):
        out_dict = self.forward(data_dict)
