"""
utils/augment.py
----------------
Spatial augmentation for 7-channel grid tensors.

Only horizontal and vertical flips are used because they preserve
task semantics (the instruction language doesn't change meaning).

90°/270° rotations are excluded: they swap the row/column axes which
would require the instruction language to also rotate — not valid.
"""

import random as _rnd
import torch

# Action remapping for flips
_FLIP_H = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5}   # left <-> right
_FLIP_V = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5}   # up   <-> down


def augment_grid(grid: torch.Tensor, action: int,
                 p_h: float = 0.4, p_v: float = 0.3) -> tuple:
    """
    Args:
        grid   : (7, H, W) float tensor
        action : int label
    Returns:
        (augmented_grid, remapped_action)
    """
    if _rnd.random() < p_h:
        grid   = torch.flip(grid, dims=[2])
        action = _FLIP_H[action]
    if _rnd.random() < p_v:
        grid   = torch.flip(grid, dims=[1])
        action = _FLIP_V[action]
    return grid, action
