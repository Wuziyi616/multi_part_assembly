import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNWrapper(nn.Module):

    def __init__(self, rnn, batch_first=True):
        super().__init__()

        self.rnn = rnn
        self.batch_first = batch_first

    def forward(self, x, hidden=None, valids=None):
        """Forward pass.

        Args:
            x: [B, T, C] or [T, B, C] depending on self.batch_first
            hidden: initial hidden state
            valids: [B, P], 1 for input parts, 0 for padded parts
                always batch_first because we only need to compute length

        Returns:
            output of the original RNN
        """
        # compute lengths
        if valids is not None:
            lengths = torch.sum(valids, dim=1).cpu()
            x = pack_padded_sequence(
                x,
                lengths,
                batch_first=self.batch_first,
                enforce_sorted=False,
            )

        output, hidden = self.rnn(x, hidden)

        # unpack outputs
        if valids is not None:
            output, _ = pad_packed_sequence(
                output,
                batch_first=self.batch_first,
                total_length=valids.shape[1],
            )

        return output, hidden
