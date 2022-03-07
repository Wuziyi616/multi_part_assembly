import torch.nn as nn


def build_transformer_encoder(d_model,
                              num_heads,
                              ffn_dim,
                              num_layers,
                              norm_first=True):
    """Build the Transformer Encoder.

    Input tokens [B, N, C], output features after interaction [B, N, C].

    Args:
        d_model: input token feature dim
        num_heads: head number in multi-head self-attention
        ffn_dim: MLP hidden size in FFN
        num_layers: stack TransformerEncoder layers number
        norm_first: whether apply pre-LN
    """
    transformer_enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        norm_first=norm_first,
        batch_first=True)
    # TODO: sometimes this final-norm is unnecessary even in pre-LN cases
    norm = nn.LayerNorm(d_model) if norm_first else None
    transformer_encoder = nn.TransformerEncoder(
        encoder_layer=transformer_enc_layer, num_layers=num_layers, norm=norm)
    return transformer_encoder


class TransformerEncoder(nn.Module):
    """Transformer encoder with padding_mask."""

    def __init__(self,
                 d_model,
                 num_heads,
                 ffn_dim,
                 num_layers,
                 norm_first=True):
        super().__init__()

        self.transformer_encoder = build_transformer_encoder(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            norm_first=norm_first)

    def forward(self, tokens, valid_masks):
        """Forward pass.

        Args:
            tokens: [B, N, C]
            valid_masks: [B, N], 1 for valid, 0 for padded

        Returns:
            torch.Tensor: [B, N, C]
        """
        if valid_masks is not None:
            assert valid_masks.shape == tokens.shape[:2]
            pad_masks = (1 - valid_masks).bool()  # 1 --> padding
        else:
            pad_masks = None
        out = self.transformer_encoder(tokens, src_key_padding_mask=pad_masks)
        return out
