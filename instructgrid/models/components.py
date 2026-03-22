"""
models/components.py
--------------------
Shared building blocks used by all models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Attentive GRU Encoder
# ======================================================================

class AttentiveGRUEncoder(nn.Module):
    """
    Encodes an instruction token sequence with a GRU, then uses a
    learned attention vector to pool over ALL hidden states — not just
    the final one.

    Why this helps:
      In "place the RED box in the blue zone", the colour word 'red'
      appears early. The final GRU hidden state has processed 8+ tokens
      since then and may have diluted that signal. Attention over all
      timesteps lets the model focus on whichever token carries the
      most discriminative information for the current decision.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru        = nn.GRU(embed_dim, hidden_dim, batch_first=True,
                                 bidirectional=False)
        # Attention: project each hidden state to a scalar score
        self.attn_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v     = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout    = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, T)
        returns: (B, hidden_dim)  — attention-weighted sum of GRU states
        """
        # Create padding mask — positions where token == 0 (PAD)
        pad_mask = (tokens == 0)                          # (B, T)

        emb      = self.dropout(self.embedding(tokens))   # (B, T, embed_dim)
        hs, _    = self.gru(emb)                          # (B, T, hidden_dim)

        # Additive attention
        scores = self.attn_v(torch.tanh(self.attn_proj(hs)))  # (B, T, 1)
        scores = scores.squeeze(-1)                            # (B, T)

        # Mask padding positions with large negative value before softmax
        scores = scores.masked_fill(pad_mask, -1e9)
        weights = F.softmax(scores, dim=-1)                    # (B, T)

        # Weighted sum
        out = (hs * weights.unsqueeze(-1)).sum(dim=1)          # (B, hidden_dim)
        return out


# ======================================================================
# Auxiliary Colour Head
# ======================================================================

class AuxColourHead(nn.Module):
    """
    Auxiliary classification head that predicts which box colour is
    relevant to the instruction: 0=red, 1=blue.

    This forces the shared language encoder to explicitly represent
    colour information — the main loss alone doesn't guarantee this
    because the model can get decent BC accuracy without reliably
    learning colour-conditional routing.

    Labels are derived cheaply from the instruction string at zero
    annotation cost (see derive_colour_label()).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),   # 2 classes: red=0, blue=1
        )

    def forward(self, lang_vec: torch.Tensor) -> torch.Tensor:
        """lang_vec: (B, hidden_dim)  →  logits (B, 2)"""
        return self.fc(lang_vec)


def derive_colour_labels(instructions: list, tokenizer) -> torch.Tensor:
    """
    Derive the relevant box colour from instruction strings.
    Returns a LongTensor of shape (B,) with values 0=red, 1=blue.
    Uses -1 for unknown (ignored by CrossEntropyLoss ignore_index).
    """
    labels = []
    for instr in instructions:
        text = instr.lower() if instr else ""
        if "red box" in text or "red block" in text:
            labels.append(0)
        elif "blue box" in text or "blue block" in text:
            labels.append(1)
        else:
            labels.append(-1)
    return torch.tensor(labels, dtype=torch.long)


# ======================================================================
# Cross-Attention Fusion
# ======================================================================

class CrossAttentionFusion(nn.Module):
    """
    Language vector attends over CNN spatial feature map.

    Query  = language hidden state  (B, lang_dim)
    Keys   = CNN spatial features   (B, HW, vis_dim)
    Values = CNN spatial features   (B, HW, vis_dim)
    Output = attended summary       (B, lang_dim)   + residual
    """

    def __init__(self, lang_dim: int, vis_dim: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.k_proj   = nn.Linear(vis_dim, lang_dim)
        self.v_proj   = nn.Linear(vis_dim, lang_dim)
        self.attn     = nn.MultiheadAttention(lang_dim, num_heads,
                                              dropout=dropout,
                                              batch_first=True)
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
        vis_flat = vis_map.flatten(2).permute(0, 2, 1)   # (B, HW, C)

        q = lang.unsqueeze(1)                             # (B, 1, lang_dim)
        k = self.k_proj(vis_flat)
        v = self.v_proj(vis_flat)

        attn_out, _ = self.attn(q, k, v)
        attn_out    = attn_out.squeeze(1)

        fused = self.out_proj(torch.cat([lang, attn_out], dim=-1))
        return self.norm(fused + lang)


# ======================================================================
# FiLM Layer & FiLM-conditioned CNN
# ======================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    output = gamma(lang) * features + beta(lang)
    Initialised to identity for stable training start.
    """

    def __init__(self, lang_dim: int, num_channels: int):
        super().__init__()
        self.gamma_fc = nn.Linear(lang_dim, num_channels)
        self.beta_fc  = nn.Linear(lang_dim, num_channels)
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, features: torch.Tensor,
                lang: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(lang).unsqueeze(-1).unsqueeze(-1)
        beta  = self.beta_fc(lang).unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


class FiLMCNN(nn.Module):
    """3-layer CNN where each layer is conditioned by the language vector."""

    def __init__(self, in_channels: int, lang_dim: int,
                 out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.film1 = FiLMLayer(lang_dim, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.film2 = FiLMLayer(lang_dim, 64)
        self.conv3 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.film3 = FiLMLayer(lang_dim, out_channels)

    def forward(self, grid: torch.Tensor,
                lang: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.film1(self.conv1(grid), lang))
        x = F.relu(self.film2(self.conv2(x),    lang))
        x = F.relu(self.film3(self.conv3(x),    lang))
        return x

