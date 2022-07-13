import torch.nn as nn

from multi_part_assembly.models import VNTransformerEncoderLayer, \
    VNLayerNorm, VNLinear, VNInFeature, VNEqFeature


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


class CanonicalVNTransformerEncoder(TransformerEncoder):
    """VNTransformer encoder with padding_mask.

    It first maps tokens to invariant features.
    Then, it applies the normal TransformerEncoder to perform interactions.
    Finally, it maps the invariant features back to the rotation of tokens.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        dropout=0.,
        out_dim=None,
    ):
        super().__init__(
            d_model=d_model * 3,
            num_heads=num_heads,
            ffn_dim=d_model * 3 * 4,
            num_layers=num_layers,
            norm_first=True,
            dropout=dropout,
            out_dim=out_dim,
        )

        self.feats_in = VNInFeature(d_model, dim=4)
        self.feats_eq = VNEqFeature(d_model, dim=4)

    def forward(self, tokens, valid_masks):
        """Forward pass.

        Args:
            tokens: [B, C, 3, N]
            valid_masks: [B, N], True for valid, False for padded

        Returns:
            torch.Tensor: [B, C, 3, N]
        """
        # map tokens to invariant features
        tokens_in = self.feats_in(tokens).flatten(1, 2)  # [B, C*3, N]
        tokens_in = tokens_in.transpose(1, 2).contiguous()  # [B, N, C*3]
        out_in = super().forward(tokens_in, valid_masks)  # [B, N, C*3]
        # back to [B, C, 3, N]
        out_in = out_in.transpose(1, 2).unflatten(1, (-1, 3)).contiguous()
        out_eq = self.feats_eq(tokens, out_in)
        return out_eq


""" test code
import torch
from multi_part_assembly.models import CanonicalVNTransformerEncoder, VNTransformerEncoder
from multi_part_assembly.utils import random_rotation_matrixs
vn_trans = CanonicalVNTransformerEncoder(16, 4, 2, 0.)
pc = torch.rand(2, 16, 3, 100)
rmat = random_rotation_matrixs((2, 16))  # [2, 16, 3, 3]
rot_pc = rmat @ pc
trans_pc = vn_trans(pc)
rot_trans_pc = rmat @ trans_pc
trans_rot_pc = vn_trans(rot_pc)
(rot_trans_pc - trans_rot_pc).abs().max()
"""


def build_vn_transformer_encoder(
    d_model,
    num_heads,
    num_layers,
    relu=True,
    dropout=0.,
):
    """Build the SO(3) rotation equivariant Transformer Encoder.

    Input tokens [B, C, 3, N], output features after interaction [B, C, 3, N].

    Args:
        d_model: input token feature dim
        num_heads: head number in multi-head self-attention
        ffn_dim: MLP hidden size in FFN
        num_layers: stack TransformerEncoder layers number
    """
    transformer_enc_layer = VNTransformerEncoderLayer(
        d_model=d_model,
        n_head=num_heads,
        relu=relu,
        dropout=dropout,
    )
    norm = VNLayerNorm(d_model)
    transformer_encoder = nn.TransformerEncoder(
        encoder_layer=transformer_enc_layer, num_layers=num_layers, norm=norm)
    return transformer_encoder


class VNTransformerEncoder(nn.Module):
    """VNTransformer encoder with padding_mask."""

    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        dropout=0.,
        out_dim=None,
    ):
        super().__init__()

        self.transformer_encoder = build_vn_transformer_encoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.out_fc = VNLinear(d_model, out_dim, dim=4) if \
            out_dim is not None else nn.Identity()

    def forward(self, tokens, valid_masks):
        """Forward pass.

        Args:
            tokens: [B, C, 3, N]
            valid_masks: [B, N], True for valid, False for padded

        Returns:
            torch.Tensor: [B, C, 3, N]
        """
        if valid_masks is not None:
            assert valid_masks.shape[0] == tokens.shape[0]
            assert valid_masks.shape[1] == tokens.shape[3]
            pad_masks = (~valid_masks)  # True --> padding
        else:
            pad_masks = None
        out = self.transformer_encoder(tokens, src_key_padding_mask=pad_masks)
        return self.out_fc(out)
