"""
models/agent_v3.py  —  Stage 2: Cross-attention fusion
--------------------------------------------------------
Improvements over AgentV2:
  E. Language vector attends over CNN spatial feature maps
     (replaces naive concatenation with cross-attention).

Retains all Stage 1 improvements (A, B, D).
"""

import torch
import torch.nn as nn

from models.cross_attention_fusion import CrossAttentionFusion

GRID_SIZE     = 8
GRID_CHANNELS = 5
NUM_ACTIONS   = 6


class AgentV3(nn.Module):
    """
    Args:
        vocab_size  : tokenizer vocabulary size
        embed_dim   : word embedding dimension
        hidden_dim  : GRU hidden dimension  (also used as lang_dim in cross-attn)
        cnn_channels: output channels of CNN (used as vis_dim)
        num_heads   : number of attention heads in cross-attention
        dropout     : dropout on policy head
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, cnn_channels: int = 64,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()

        # ── Language encoder (same as V2) ─────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru       = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # ── CNN encoder — returns feature MAP not flat vector ─────────
        self.conv1 = nn.Conv2d(GRID_CHANNELS,   32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,              64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, cnn_channels,    kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        # NOTE: we do NOT flatten here — cross-attention needs the spatial map

        # ── E. Cross-attention fusion ──────────────────────────────────
        self.fusion = CrossAttentionFusion(
            lang_dim=hidden_dim, vis_dim=cnn_channels,
            num_heads=num_heads, dropout=dropout
        )

        # ── Policy head ───────────────────────────────────────────────
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, NUM_ACTIONS)

    # ─────────────────────────────────────────────────────────────────

    def encode_instruction(self, tokens: torch.Tensor) -> torch.Tensor:
        emb  = self.embedding(tokens)
        _, h = self.gru(emb)
        return h.squeeze(0)   # (B, hidden_dim)

    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: (B,5,8,8) → feature map (B, cnn_channels, 8, 8)"""
        x = self.relu(self.conv1(grid))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x

    def forward(self, tokens: torch.Tensor,
                grid: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, T)
        grid   : (B, 5, 8, 8)
        returns: (B, 6)
        """
        lang    = self.encode_instruction(tokens)   # (B, hidden_dim)
        vis_map = self.encode_grid(grid)             # (B, cnn_ch, 8, 8)
        fused   = self.fusion(lang, vis_map)         # (B, hidden_dim)
        fused   = self.dropout(self.norm(fused))
        return self.fc(fused)
