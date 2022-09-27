import torch.nn as nn


def build_transformer_encoder(
    d_model,
    num_heads,
    ffn_dim,
    num_layers,
    norm_first=True,
    dropout=0.1,
):
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
        dropout=dropout,
        norm_first=norm_first,
        batch_first=True,
    )
    norm = nn.LayerNorm(d_model) if norm_first else None
    transformer_encoder = nn.TransformerEncoder(
        encoder_layer=transformer_enc_layer, num_layers=num_layers, norm=norm)
    return transformer_encoder


class TransformerEncoder(nn.Module):
    """Transformer encoder with padding_mask."""

    def __init__(
        self,
        d_model,
        num_heads,
        ffn_dim,
        num_layers,
        norm_first=True,
        dropout=0.1,
        out_dim=None,
    ):
        super().__init__()

        self.transformer_encoder = build_transformer_encoder(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            norm_first=norm_first,
            dropout=dropout,
        )
        self.out_fc = nn.Linear(d_model, out_dim) if \
            out_dim is not None else nn.Identity()

    def forward(self, tokens, valid_masks):
        """Forward pass.

        Args:
            tokens: [B, N, C]
            valid_masks: [B, N], True for valid, False for padded

        Returns:
            torch.Tensor: [B, N, C]
        """
        if valid_masks is not None:
            assert valid_masks.shape == tokens.shape[:2]
            pad_masks = (~valid_masks)  # True --> padding
        else:
            pad_masks = None
        out = self.transformer_encoder(tokens, src_key_padding_mask=pad_masks)
        return self.out_fc(out)
