import torch
import torch.nn as nn


class PNTransformerMLP(nn.Module):

    def train_loss_function(self, data_dict):
        out_dict = self.forward(data_dict)

