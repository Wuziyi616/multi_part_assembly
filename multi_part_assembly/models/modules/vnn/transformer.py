"""Code borrowed from https://github.com/karpathy/minGPT"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import VNLinear, VNLayerNorm, VNReLU, VNLeakyReLU


class VNSelfAttention(nn.Module):
    """Inspired by VNT-Net: https://arxiv.org/pdf/2205.09690.pdf.

    Note that, we cannot use dropout in VN networks.
    """

    def __init__(self, d_model, n_head, dropout=0.):
        super().__init__()

        assert d_model % n_head == 0
        assert dropout == 0.
        self.n_head = n_head

        # key, query, value projections for all heads
        self.key = VNLinear(d_model, d_model)
        self.query = VNLinear(d_model, d_model)
        self.value = VNLinear(d_model, d_model)

        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # output projection
        self.proj = VNLinear(d_model, d_model)

    def forward(self, x, src_key_padding_mask=None):
        """Forward pass.

        Args:
            x: [B, C, 3, N]
            src_key_padding_mask: None or [B, N], True means padded tokens

        Returns:
            [B, C, 3, N]
        """
        B, C, _, N = x.size()

        # [B, nh, N, hs*3]
        k = self.key(x).reshape(B, self.n_head, C // self.n_head, 3, N).\
            permute(0, 1, 4, 2, 3).flatten(-2, -1)
        q = self.query(x).reshape(B, self.n_head, C // self.n_head, 3, N).\
            permute(0, 1, 4, 2, 3).flatten(-2, -1)
        v = self.value(x).reshape(B, self.n_head, C // self.n_head, 3, N).\
            permute(0, 1, 4, 2, 3).flatten(-2, -1)

        # [B, nh, N, hs*3] x [B, nh, N, hs*3] --> [B, nh, N, N]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if src_key_padding_mask is not None:
            assert src_key_padding_mask.dtype == torch.bool
            mask = src_key_padding_mask[:, None, None, :]  # [B, 1, 1, N]
            att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # [B, nh, N, N] x [B, nh, N, hs*3] --> [B, nh, N, hs*3]
        # to [B, C, 3, N]
        y = y.transpose(2, 3).unflatten(2, (-1, 3)).flatten(1, 2).contiguous()

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class VNTransformerEncoderLayer(nn.Module):
    """VN Transformer block."""

    def __init__(self, d_model, n_head, relu=True, dropout=0.):
        super().__init__()

        assert dropout == 0.
        self.ln1 = VNLayerNorm(d_model)
        self.ln2 = VNLayerNorm(d_model)
        self.attn = VNSelfAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            VNLinear(d_model, 4 * d_model),
            VNReLU(4 * d_model) if relu else VNLeakyReLU(4 * d_model),
            VNLinear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, src_key_padding_mask=None, src_mask=None):
        """Forward pass.

        Args:
            x: [B, C, 3, N]
            src_key_padding_mask: None or [B, N], True means padded tokens
            src_mask: useless, to be compatible with nn.TransformerEncoderLayer

        Returns:
            [B, C, 3, N]
        """
        x = x + self.attn(self.ln1(x), src_key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


""" test code
import torch
from multi_part_assembly.models import VNTransformerEncoderLayer
from multi_part_assembly.utils import random_rotation_matrixs
vn_attn = VNTransformerEncoderLayer(16, 4, True, 0)
pc = torch.rand(2, 16, 3, 100)
rmat = random_rotation_matrixs(2)
rot_pc = rmat[:, None] @ pc
attn_pc = vn_attn(pc)
rot_attn_pc = rmat[:, None] @ attn_pc
attn_rot_pc = vn_attn(rot_pc)
"""
