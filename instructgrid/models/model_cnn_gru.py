"""
models/model_cnn_gru.py
------------------------
Model: CNN + Attentive-GRU  (baseline with all core fixes)

Changes from v1:
  - AttentiveGRUEncoder: pools over ALL hidden states via learned
    attention — colour words early in the instruction are not diluted
  - AuxColourHead: auxiliary loss forces the encoder to explicitly
    represent which box colour is relevant
  - hidden_dim: 256 (was 128) for more expressive colour routing
"""

import torch
import torch.nn as nn

from grid_env_simple import GRID_CHANNELS, NUM_ACTIONS
from models.components import AttentiveGRUEncoder, AuxColourHead

GRID_SIZE = 8


class CNNGRUAgent(nn.Module):

    MODEL_NAME = "cnn_gru"

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        # Attentive language encoder
        self.lang_enc = AttentiveGRUEncoder(vocab_size, embed_dim,
                                            hidden_dim, dropout)

        # 3-layer CNN visual encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 64 * GRID_SIZE * GRID_SIZE   # 4096

        # Policy head
        fused_dim    = hidden_dim + cnn_out
        self.norm    = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(fused_dim, NUM_ACTIONS)

        # Auxiliary colour head
        self.aux_colour = AuxColourHead(hidden_dim)

    def _encode(self, tokens, grid):
        lang  = self.lang_enc(tokens)
        vis   = self.cnn(grid)
        fused = self.dropout(self.norm(torch.cat([lang, vis], dim=-1)))
        return lang, fused

    def forward(self, tokens: torch.Tensor, grid: torch.Tensor) -> dict:
        lang, fused = self._encode(tokens, grid)
        return {"action": self.fc(fused), "colour": self.aux_colour(lang)}

    def predict(self, tokens, grid, hx=None):
        with torch.no_grad():
            out = self.forward(tokens, grid)
        return out["action"].argmax(1).item(), None
