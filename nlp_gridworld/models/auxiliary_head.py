"""
models/auxiliary_head.py
-------------------------
Auxiliary prediction heads that provide extra supervision signals at
no additional annotation cost (labels are derived from the instruction).

Heads:
  AuxObjectHead : predict which object class is relevant (red_box / blue_box /
                  red_target / blue_target)
  AuxSubgoalHead: predict the next sub-goal (navigate / pick / navigate / place)

Both are added to the main cross-entropy loss with a small weight (lambda_aux).

Usage in train_v4.py:
    obj_logits    = aux_obj(fused)
    subgoal_logits = aux_sub(fused)
    loss = (ce_loss(action_logits, actions)
            + cfg["lambda_obj"]     * ce_loss(obj_logits, obj_labels)
            + cfg["lambda_subgoal"] * ce_loss(subgoal_logits, subgoal_labels))
"""

import torch
import torch.nn as nn

# Object classes: 0=red_box, 1=blue_box, 2=red_target, 3=blue_target
NUM_OBJECTS = 4
# Subgoal classes: 0=navigate_to_box, 1=pick_box, 2=navigate_to_target, 3=place_box
NUM_SUBGOALS = 4


class AuxObjectHead(nn.Module):
    """
    Predicts which object/target is relevant to the instruction.

    Args:
        in_dim      : dimension of the fused feature vector
        hidden_dim  : intermediate projection size
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, NUM_OBJECTS),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """fused: (B, in_dim) → (B, NUM_OBJECTS)"""
        return self.net(fused)


class AuxSubgoalHead(nn.Module):
    """
    Predicts the current sub-goal phase.

    Args:
        in_dim      : dimension of the fused feature vector
        hidden_dim  : intermediate projection size
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, NUM_SUBGOALS),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """fused: (B, in_dim) → (B, NUM_SUBGOALS)"""
        return self.net(fused)


# ─────────────────────────────────────────────────────────────────────
# Label derivation helpers
# ─────────────────────────────────────────────────────────────────────

def derive_object_label(instruction: str) -> int:
    """
    Heuristic: map instruction keywords to the relevant object index.
    Returns -1 (ignore) if no clear match.

    Object index mapping:
      0 = red_box, 1 = blue_box, 2 = red_target, 3 = blue_target
    """
    instr = instruction.lower()
    if "red" in instr and ("box" in instr or "block" in instr):
        return 0
    if "blue" in instr and ("box" in instr or "block" in instr):
        return 1
    if "red" in instr and ("zone" in instr or "target" in instr or "area" in instr):
        return 2
    if "blue" in instr and ("zone" in instr or "target" in instr or "area" in instr):
        return 3
    return -1   # unknown → will be ignored by CrossEntropyLoss(ignore_index=-1)


def derive_object_labels_batch(instructions: list) -> torch.Tensor:
    """
    instructions : list of B strings
    returns      : LongTensor (B,)
    """
    labels = [derive_object_label(i) for i in instructions]
    return torch.tensor(labels, dtype=torch.long)
