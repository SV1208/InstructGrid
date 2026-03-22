"""
models/model_lstm_attention.py
-------------------------------
Model: LSTM + Attentive-GRU + Cross-Attention

Changes from v1:
  - AttentiveGRUEncoder instead of plain GRU
  - AuxColourHead auxiliary loss
  - hidden_dim 256
  - LSTM maintains state across timesteps at eval time
"""

import torch
import torch.nn as nn

from grid_env_simple import GRID_CHANNELS, NUM_ACTIONS
from models.components import (AttentiveGRUEncoder, AuxColourHead,
                                CrossAttentionFusion)


class LSTMAttentionAgent(nn.Module):

    MODEL_NAME = "lstm_attention"

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 256, cnn_channels: int = 64,
                 num_heads: int = 4, lstm_hidden: int = 256,
                 num_lstm_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm_hidden     = lstm_hidden
        self.num_lstm_layers = num_lstm_layers

        self.lang_enc = AttentiveGRUEncoder(vocab_size, embed_dim,
                                            hidden_dim, dropout)

        self.conv1 = nn.Conv2d(GRID_CHANNELS, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, cnn_channels, 3, padding=1)
        self.relu  = nn.ReLU()

        self.fusion = CrossAttentionFusion(hidden_dim, cnn_channels,
                                           num_heads, dropout)

        self.lstm = nn.LSTM(hidden_dim, lstm_hidden,
                            num_layers=num_lstm_layers, batch_first=True)

        self.norm       = nn.LayerNorm(lstm_hidden)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(lstm_hidden, NUM_ACTIONS)
        self.aux_colour = AuxColourHead(hidden_dim)

    def _encode_grid(self, grid):
        x = self.relu(self.conv1(grid))
        x = self.relu(self.conv2(x))
        return self.relu(self.conv3(x))

    def forward(self, tokens: torch.Tensor,
                grid: torch.Tensor, hx=None) -> dict:
        lang    = self.lang_enc(tokens)
        vis_map = self._encode_grid(grid)
        fused   = self.fusion(lang, vis_map)             # (B, hidden_dim)

        lstm_out, _ = self.lstm(fused.unsqueeze(1), hx)  # (B,1,lstm_hidden)
        out = self.dropout(self.norm(lstm_out.squeeze(1)))

        return {"action": self.fc(out), "colour": self.aux_colour(lang)}

    def predict(self, tokens, grid, hx=None):
        with torch.no_grad():
            lang    = self.lang_enc(tokens)
            vis_map = self._encode_grid(grid)
            fused   = self.fusion(lang, vis_map)
            lstm_out, hx_out = self.lstm(fused.unsqueeze(1), hx)
            out    = self.dropout(self.norm(lstm_out.squeeze(1)))
            action = self.fc(out).argmax(1).item()
        return action, hx_out
