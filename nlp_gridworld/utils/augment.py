"""
utils/augment.py
----------------
Spatial augmentation for 5×8×8 grid tensors.

Valid operations that preserve task semantics:
  - Horizontal flip  (left <-> right, remaps move-left/move-right actions)
  - Vertical flip    (up   <-> down,  remaps move-up/move-down actions)
  - 180° rotation    (= horizontal + vertical flip, remaps both pairs)

90° / 270° rotations are intentionally excluded because the non-square
instruction language ("place the red box in the blue zone") doesn't change
semantics with pure reflections but DOES change with rotations that swap
the row/column meaning.
"""

import torch
import random as _rnd

# Action index constants
UP, DOWN, LEFT, RIGHT, PICK, PLACE = 0, 1, 2, 3, 4, 5

# Maps for each augmentation
_FLIP_H_MAP = {UP: UP, DOWN: DOWN, LEFT: RIGHT, RIGHT: LEFT, PICK: PICK, PLACE: PLACE}
_FLIP_V_MAP = {UP: DOWN, DOWN: UP, LEFT: LEFT, RIGHT: RIGHT, PICK: PICK, PLACE: PLACE}
_ROT180_MAP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT, PICK: PICK, PLACE: PLACE}


def augment_grid(grid: torch.Tensor, action: int,
                 p_h: float = 0.5, p_v: float = 0.3
                 ) -> tuple:
    """
    Apply random reflections to a 5×8×8 grid tensor and remap the action.

    Args:
        grid    : (5, H, W) float tensor
        action  : integer action label
        p_h     : probability of horizontal flip
        p_v     : probability of vertical flip

    Returns:
        (augmented_grid, remapped_action)
    """
    if _rnd.random() < p_h:
        grid   = torch.flip(grid, dims=[2])   # flip W
        action = _FLIP_H_MAP[action]

    if _rnd.random() < p_v:
        grid   = torch.flip(grid, dims=[1])   # flip H
        action = _FLIP_V_MAP[action]

    return grid, action


def augment_batch(grids: torch.Tensor, actions: torch.Tensor,
                  p_h: float = 0.5, p_v: float = 0.3
                  ) -> tuple:
    """
    Vectorised augmentation for a whole batch.

    Args:
        grids   : (B, 5, H, W)
        actions : (B,) LongTensor

    Returns:
        (augmented_grids, remapped_actions)
    """
    B = grids.size(0)
    actions = actions.clone()

    # Horizontal flip mask
    mask_h = torch.rand(B) < p_h
    if mask_h.any():
        grids[mask_h] = torch.flip(grids[mask_h], dims=[3])
        for i in mask_h.nonzero(as_tuple=True)[0]:
            actions[i] = _FLIP_H_MAP[actions[i].item()]

    # Vertical flip mask
    mask_v = torch.rand(B) < p_v
    if mask_v.any():
        grids[mask_v] = torch.flip(grids[mask_v], dims=[2])
        for i in mask_v.nonzero(as_tuple=True)[0]:
            actions[i] = _FLIP_V_MAP[actions[i].item()]

    return grids, actions
