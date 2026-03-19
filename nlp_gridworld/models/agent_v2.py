"""
models/agent_v2.py  —  Stage 1
--------------------------------
Improvements over baseline:
  A. Instruction encoded at every timestep (not just step 0).
  B. LayerNorm + Dropout on the fused representation.
  D. 3-layer CNN grid encoder (was 2-layer).

Compatible with the same DataLoader / training loop as the baseline.
The only requirement is that DemoDataset(propagate=True) is used so
every sample carries the instruction string.
"""

import torch
import torch.nn as nn

GRID_SIZE     = 8
GRID_CHANNELS = 5
NUM_ACTIONS   = 6


class AgentV2(nn.Module):
    """
    Args:
        vocab_size   : size of the tokenizer vocabulary
        embed_dim    : word embedding dimension
        hidden_dim   : GRU hidden dimension
        dropout      : dropout probability on fused features
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()

        # ── A. Language encoder ───────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru       = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # ── D. 3-layer CNN grid encoder ───────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # ← new layer
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out_dim = 64 * GRID_SIZE * GRID_SIZE   # 4096

        # ── B. Fusion with LayerNorm + Dropout ────────────────────────
        fused_dim    = hidden_dim + cnn_out_dim
        self.norm    = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(fused_dim, NUM_ACTIONS)

    # ─────────────────────────────────────────────────────────────────

    def encode_instruction(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) → (B, hidden_dim)"""
        emb  = self.embedding(tokens)
        _, h = self.gru(emb)
        return h.squeeze(0)

    def forward(self, tokens: torch.Tensor,
                grid: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, T)       — token ids (instruction at EVERY step)
        grid   : (B, 5, 8, 8)
        returns: (B, 6)       — action logits
        """
        lang  = self.encode_instruction(tokens)        # (B, hidden_dim)
        vis   = self.cnn(grid)                         # (B, 4096)
        fused = torch.cat([lang, vis], dim=-1)         # (B, hidden_dim+4096)
        fused = self.dropout(self.norm(fused))
        return self.fc(fused)
