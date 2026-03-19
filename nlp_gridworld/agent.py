"""
agent.py  —  BASELINE MODEL  (do not modify)
---------------------------------------------
Original architecture used in train.py and evaluate.py.

Architecture:
  - Instruction encoder : Embedding → GRU  →  hidden state (128-d)
  - Grid encoder        : 2-layer CNN      →  flat features
  - Fusion              : Concatenation
  - Head                : Linear → 6 actions

NOTE: The baseline dataset passes the instruction ONLY on step 0.
      To reproduce baseline behaviour use DemoDataset(propagate=False).
"""

import torch
import torch.nn as nn

GRID_SIZE     = 8
GRID_CHANNELS = 5
NUM_ACTIONS   = 6


class Agent(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128):
        super().__init__()

        # --- Language encoder ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru       = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # --- Grid (visual) encoder ---
        self.cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out_dim = 64 * GRID_SIZE * GRID_SIZE   # 4096

        # --- Policy head ---
        self.fc = nn.Linear(hidden_dim + cnn_out_dim, NUM_ACTIONS)

    def encode_instruction(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) → (B, hidden_dim)"""
        emb = self.embedding(tokens)
        _, h = self.gru(emb)
        return h.squeeze(0)

    def forward(self, tokens: torch.Tensor,
                grid: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, T)      — token ids
        grid   : (B, 5, 8, 8)
        returns: (B, 6)      — action logits
        """
        lang  = self.encode_instruction(tokens)
        vis   = self.cnn(grid)
        fused = torch.cat([lang, vis], dim=-1)
        return self.fc(fused)
