"""
models/agent_v3_lstm.py  —  Stage 2: LSTM temporal policy head
---------------------------------------------------------------
Improvement F: wraps AgentV3 with an LSTM that gives the agent memory
of its own previous actions and observations across a trajectory.

During TRAINING on the existing single-step dataset, set T=1 and pass
hx=None — fully backward-compatible.

During EVALUATION, maintain (h, c) across timesteps:
    logits, hx = model(tokens, grid.unsqueeze(1), hx=hx)
    action = logits[:, -1].argmax(1).item()
    # pass hx to the next call
"""

import torch
import torch.nn as nn

from models.agent_v3 import AgentV3

NUM_ACTIONS = 6


class AgentV3LSTM(nn.Module):
    """
    Args:
        base_model  : AgentV3 instance (without its final fc layer)
        fused_dim   : hidden_dim of the base model (= cross-attn output dim)
        lstm_hidden : hidden size of the temporal LSTM
        num_layers  : LSTM depth
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, cnn_channels: int = 64,
                 num_heads: int = 4, dropout: float = 0.2,
                 lstm_hidden: int = 128, num_layers: int = 1):
        super().__init__()

        # Build base (but override its fc head)
        self.base = AgentV3(
            vocab_size=vocab_size, embed_dim=embed_dim,
            hidden_dim=hidden_dim, cnn_channels=cnn_channels,
            num_heads=num_heads, dropout=dropout
        )
        # Remove base head — we replace it
        in_dim = self.base.fc.in_features
        self.base.fc = nn.Identity()

        # Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=lstm_hidden,
            num_layers=num_layers, batch_first=True, dropout=0.0
        )
        self.fc   = nn.Linear(lstm_hidden, NUM_ACTIONS)

    # ─────────────────────────────────────────────────────────────────

    def forward(self, tokens: torch.Tensor,
                grid: torch.Tensor,
                hx=None):
        """
        tokens : (B, T_lang)        — instruction token ids
        grid   : (B, T_seq, 5, 8, 8)  — sequence of grids (T_seq=1 for BC)
        hx     : optional LSTM state tuple (h, c), shape each (layers, B, lstm_hidden)
        returns: logits (B, T_seq, 6), hx_out
        """
        B, T_seq, C, H, W = grid.shape
        fused_seq = []

        for t in range(T_seq):
            # (B, fused_dim) — base.forward uses Identity fc now
            fused_t = self.base(tokens, grid[:, t])
            fused_seq.append(fused_t)

        fused_seq = torch.stack(fused_seq, dim=1)     # (B, T_seq, fused_dim)
        lstm_out, hx_out = self.lstm(fused_seq, hx)   # (B, T_seq, lstm_hidden)
        logits   = self.fc(lstm_out)                   # (B, T_seq, 6)
        return logits, hx_out

    def forward_step(self, tokens: torch.Tensor,
                     grid: torch.Tensor, hx=None):
        """
        Convenience wrapper for single-step evaluation.
        grid: (B, 5, 8, 8)  — no time dimension
        returns: logits (B, 6), hx
        """
        logits, hx = self.forward(tokens, grid.unsqueeze(1), hx)
        return logits[:, 0], hx
