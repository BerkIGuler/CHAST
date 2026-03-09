"""
CHAST: Channel Attention eSTimator (NeurIPS 2025 Workshop - AI4NextG).

Operates on sparse pilot LS estimates (full K×L grid, non-zero only at DM-RS positions).
Conv2D is applied as standard 2D convolution on this zero-filled grid — no special sparse op.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


def sparse_ls_grid_to_tensor(ls_grid: torch.Tensor) -> torch.Tensor:
    """
    Build the CHAST input from the sparse LS estimate grid.

    Expects the full (K x L) grid with non-zero only at pilot locations (dataset
    sparse format). Stacks real and imaginary parts into the 2-channel tensor
    CHAST expects.

    Args:
        ls_grid: Complex tensor (..., K, L) — sparse LS grid from dataset.

    Returns:
        (..., 2, K, L) float tensor: channel 0 = real, channel 1 = imag.
    """
    return torch.stack([ls_grid.real, ls_grid.imag], dim=-3)


class CHAST(nn.Module):
    """
    CHAST: CNN front-end (on sparse grid) + patch tokenization + 1 Transformer block + recon.
    """

    def __init__(
        self,
        num_subcarriers: int = 120,
        num_symbols: int = 14,
        in_channels: int = 2,
        cnn_filters_1: int = 32,
        cnn_filters_2: int = 2,
        patch_h: int = 12,
        patch_w: int = 14,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.embed_dim = embed_dim

        # (i) Local feature extraction: standard Conv2D on sparse (zero-filled) grid.
        #     Input (B, 2, K, L): zeros at non-pilot positions. Convolution is applied
        #     everywhere; outputs are dense because each position sees a neighborhood
        #     that may include pilot locations — so the CNN "interpolates in latent space".
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, cnn_filters_1, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(cnn_filters_1, cnn_filters_2, kernel_size, padding=pad)
        self.act = nn.GELU()

        # (ii) Patch tokenization: one patch = one RB (12×14). Conv2d with kernel=stride=(12,14).
        self.patch_embed = nn.Conv2d(cnn_filters_2, embed_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        num_patches_h = num_subcarriers // patch_h
        num_patches_w = num_symbols // patch_w
        self.num_patches = num_patches_h * num_patches_w

        # (iii) Single Transformer encoder block
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

        # (iv) Reconstruct full grid from patches
        self.recon = nn.ConvTranspose2d(embed_dim, 2, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))

    def forward(
        self,
        x: torch.Tensor,
        sparse_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input (B, 2, K, L) — LS estimate grid, zeros at non-pilot positions.
            sparse_input: If provided, added as residual to output (B, 2, K, L).

        Returns:
            (B, 2, K, L) channel estimate (real and imag in channel dim).
        """
        # (i) CNN on sparse grid
        h = self.act(self.conv1(x))
        h = self.conv2(h)  # (B, 2, K, L)

        # (ii) Patch embed
        h = self.patch_embed(h)  # (B, embed_dim, num_patches_h, num_patches_w)
        B, C, Ph, Pw = h.shape
        h = h.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # (iii) Transformer block
        residual = h
        h = self.ln1(h)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.dropout(h) + residual
        residual = h
        h = self.ln2(h)
        h = self.mlp(h) + residual

        # (iv) Reconstruct
        h = h.transpose(1, 2).reshape(B, C, Ph, Pw)
        out = self.recon(h)  # (B, 2, K, L)

        if sparse_input is not None:
            out = out + sparse_input
        return out
