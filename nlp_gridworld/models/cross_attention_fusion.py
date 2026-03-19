"""
models/cross_attention_fusion.py
---------------------------------
Cross-attention module: language vector attends over CNN spatial features.

  Query  = language hidden state  (B, lang_dim)
  Keys   = CNN spatial features   (B, HW, vis_dim)
  Values = CNN spatial features   (B, HW, vis_dim)
  Output = attended summary       (B, out_dim)

This lets the model learn WHICH spatial locations are relevant for a given
instruction rather than blindly concatenating all locations.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Args:
        lang_dim  : dimension of the language vector
        vis_dim   : number of channels in the CNN feature map
        num_heads : number of attention heads
        dropout   : attention dropout
    """

    def __init__(self, lang_dim: int, vis_dim: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Project visual features to lang_dim so MHA has consistent dims
        self.k_proj  = nn.Linear(vis_dim, lang_dim)
        self.v_proj  = nn.Linear(vis_dim, lang_dim)

        self.attn    = nn.MultiheadAttention(
            embed_dim=lang_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(lang_dim * 2, lang_dim)
        self.norm     = nn.LayerNorm(lang_dim)

    def forward(self, lang: torch.Tensor,
                vis_map: torch.Tensor) -> torch.Tensor:
        """
        lang    : (B, lang_dim)
        vis_map : (B, C, H, W)
        returns : (B, lang_dim)
        """
        B, C, H, W = vis_map.shape
        vis_flat = vis_map.flatten(2).permute(0, 2, 1)    # (B, HW, C)

        q = lang.unsqueeze(1)                              # (B, 1, lang_dim)
        k = self.k_proj(vis_flat)                          # (B, HW, lang_dim)
        v = self.v_proj(vis_flat)                          # (B, HW, lang_dim)

        attn_out, _ = self.attn(q, k, v)                  # (B, 1, lang_dim)
        attn_out    = attn_out.squeeze(1)                  # (B, lang_dim)

        # Residual concatenation
        fused = torch.cat([lang, attn_out], dim=-1)        # (B, lang_dim*2)
        out   = self.out_proj(fused)                       # (B, lang_dim)
        return self.norm(out + lang)                       # residual
