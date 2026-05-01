"""
models/model_attention.py
--------------------------
Model: CNN + Attentive-GRU + Cross-Attention fusion

Changes from v1:
  - AttentiveGRUEncoder instead of plain GRU final-state
  - AuxColourHead auxiliary loss
  - hidden_dim 256
"""

import torch
import torch.nn as nn

from grid_env_simple import GRID_CHANNELS, NUM_ACTIONS
from models.components import (AttentiveGRUEncoder, AuxColourHead,
                                CrossAttentionFusion)


class AttentionAgent(nn.Module):

    MODEL_NAME = "attention"

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 256, cnn_channels: int = 64,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()

        self.lang_enc = AttentiveGRUEncoder(vocab_size, embed_dim,
                                            hidden_dim, dropout)

        self.conv1 = nn.Conv2d(GRID_CHANNELS, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, cnn_channels, 3, padding=1)
        self.relu  = nn.ReLU()

        self.fusion     = CrossAttentionFusion(hidden_dim, cnn_channels,
                                               num_heads, dropout)
        self.norm       = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.aux_colour = AuxColourHead(hidden_dim)

    def _encode_grid(self, grid):
        x = self.relu(self.conv1(grid))
        x = self.relu(self.conv2(x))
        return self.relu(self.conv3(x))

    def forward(self, tokens: torch.Tensor, grid: torch.Tensor) -> dict:
        lang    = self.lang_enc(tokens)
        vis_map = self._encode_grid(grid)
        fused   = self.dropout(self.norm(self.fusion(lang, vis_map)))
        return {"action": self.fc(fused), "colour": self.aux_colour(lang)}

    def predict(self, tokens, grid, hx=None):
        with torch.no_grad():
            out = self.forward(tokens, grid)
        return out["action"].argmax(1).item(), None
