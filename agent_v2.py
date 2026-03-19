# models/agent_v2.py  (drop-in replacement for agent.py)
import torch, torch.nn as nn

class AgentV2(nn.Module):
    """
    Instruction passed at every timestep, not just step 0.
    Backward-compatible: same input/output shapes as the baseline.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128,
                 grid_channels=5, grid_size=8, num_actions=6):
        super().__init__()
        # Language encoder (unchanged)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # Grid encoder (unchanged)
        self.cnn = nn.Sequential(
            nn.Conv2d(grid_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 64 * grid_size * grid_size

        # Policy head with layer norm + dropout (stage 1-B bundled in)
        self.norm = nn.LayerNorm(hidden_dim + cnn_out)
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Linear(hidden_dim + cnn_out, num_actions)

    def encode_instruction(self, tokens):
        emb = self.embedding(tokens)          # (B, T, embed_dim)
        _, h = self.gru(emb)                  # h: (1, B, hidden_dim)
        return h.squeeze(0)                   # (B, hidden_dim)

    def forward(self, tokens, grid):
        lang  = self.encode_instruction(tokens)   # (B, hidden_dim)
        vis   = self.cnn(grid)                    # (B, cnn_out)
        fused = torch.cat([lang, vis], dim=-1)    # (B, hidden_dim + cnn_out)
        fused = self.drop(self.norm(fused))
        return self.fc(fused)                     # (B, num_actions)