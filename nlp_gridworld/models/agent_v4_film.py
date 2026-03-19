"""
models/agent_v4_film.py  —  Stage 3: FiLM + auxiliary losses
--------------------------------------------------------------
Improvements over AgentV3:
  I. FiLM conditioning: language vector modulates CNN feature maps
     channel-wise at every convolutional layer.
  J. Auxiliary prediction heads for object and sub-goal grounding
     (trained jointly with lambda-weighted auxiliary losses).

Retains cross-attention fusion for the policy head output.
"""

import torch
import torch.nn as nn

from models.film import FiLMCNN
from models.cross_attention_fusion import CrossAttentionFusion
from models.auxiliary_head import AuxObjectHead, AuxSubgoalHead

GRID_SIZE     = 8
GRID_CHANNELS = 5
NUM_ACTIONS   = 6
CNN_CHANNELS  = 64


class AgentV4FiLM(nn.Module):
    """
    Args:
        vocab_size      : tokenizer vocabulary size
        embed_dim       : word embedding dimension
        hidden_dim      : GRU hidden dim (= lang_dim for FiLM and cross-attn)
        num_heads       : attention heads in cross-attention fusion
        dropout         : dropout rate
        use_aux_obj     : attach auxiliary object prediction head
        use_aux_subgoal : attach auxiliary subgoal prediction head
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, num_heads: int = 4,
                 dropout: float = 0.2,
                 use_aux_obj: bool = True,
                 use_aux_subgoal: bool = False):
        super().__init__()

        # ── Language encoder ──────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru       = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # ── I. FiLM-conditioned CNN ───────────────────────────────────
        self.film_cnn = FiLMCNN(
            grid_channels=GRID_CHANNELS,
            lang_dim=hidden_dim,
            out_channels=CNN_CHANNELS,
        )

        # ── Cross-attention fusion over FiLM feature maps ─────────────
        # After FiLM the map is (B, CNN_CHANNELS, 8, 8) — we pass this
        # to cross-attention for further instruction-driven focus.
        self.fusion = CrossAttentionFusion(
            lang_dim=hidden_dim, vis_dim=CNN_CHANNELS,
            num_heads=num_heads, dropout=dropout
        )
        # Keep a flat projection path for the aux heads
        flat_dim = CNN_CHANNELS * GRID_SIZE * GRID_SIZE   # 4096

        # ── Policy head ───────────────────────────────────────────────
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, NUM_ACTIONS)

        # ── J. Auxiliary heads ────────────────────────────────────────
        self.use_aux_obj     = use_aux_obj
        self.use_aux_subgoal = use_aux_subgoal
        if use_aux_obj:
            self.aux_obj = AuxObjectHead(in_dim=hidden_dim)
        if use_aux_subgoal:
            self.aux_sub = AuxSubgoalHead(in_dim=hidden_dim)

    # ─────────────────────────────────────────────────────────────────

    def encode_instruction(self, tokens: torch.Tensor) -> torch.Tensor:
        emb  = self.embedding(tokens)
        _, h = self.gru(emb)
        return h.squeeze(0)   # (B, hidden_dim)

    def forward(self, tokens: torch.Tensor,
                grid: torch.Tensor):
        """
        tokens : (B, T)
        grid   : (B, 5, 8, 8)
        returns:
            If use_aux_obj or use_aux_subgoal:
                dict with keys 'action', 'obj' (optional), 'subgoal' (optional)
            Else:
                action logits tensor (B, 6)  — same as other agents
        """
        lang    = self.encode_instruction(tokens)      # (B, hidden_dim)

        # FiLM-modulated visual features
        vis_flat = self.film_cnn(grid, lang)            # (B, 4096) — flat
        # Reshape back to map for cross-attention
        B        = grid.size(0)
        vis_map  = vis_flat.view(B, CNN_CHANNELS, GRID_SIZE, GRID_SIZE)

        # Cross-attention fusion
        fused   = self.fusion(lang, vis_map)            # (B, hidden_dim)
        fused   = self.dropout(self.norm(fused))

        action_logits = self.fc(fused)                  # (B, 6)

        if not self.use_aux_obj and not self.use_aux_subgoal:
            return action_logits

        out = {"action": action_logits}
        if self.use_aux_obj:
            out["obj"]     = self.aux_obj(fused)        # (B, 4)
        if self.use_aux_subgoal:
            out["subgoal"] = self.aux_sub(fused)        # (B, 4)
        return out

    def forward_action_only(self, tokens: torch.Tensor,
                            grid: torch.Tensor) -> torch.Tensor:
        """
        Returns only action logits regardless of aux head config.
        Use this in evaluate_exp.py for uniform interface.
        """
        result = self.forward(tokens, grid)
        if isinstance(result, dict):
            return result["action"]
        return result
