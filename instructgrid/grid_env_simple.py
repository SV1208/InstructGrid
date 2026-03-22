"""
grid_env_simple.py
------------------
Grid-world environment for the InstructGrid agent.

Grid: 8x8
Objects: red_box, blue_box
Targets: red_target, blue_target

State tensor: 7 channels (5 spatial + 2 held-object)
  CH0: agent position
  CH1: red box position  (moves with agent when held)
  CH2: blue box position (moves with agent when held)
  CH3: red target        (fixed)
  CH4: blue target       (fixed)
  CH5: held_red flag     (1 at agent pos when holding red box)
  CH6: held_blue flag    (1 at agent pos when holding blue box)

Actions: 0=up 1=down 2=left 3=right 4=pick 5=place
"""

import random
import torch

GRID_SIZE     = 8
GRID_CHANNELS = 7   # 5 spatial + 2 held-object flags
NUM_ACTIONS   = 6

CH_AGENT     = 0
CH_RED_BOX   = 1
CH_BLUE_BOX  = 2
CH_RED_TGT   = 3
CH_BLUE_TGT  = 4
CH_HELD_RED  = 5   # NEW
CH_HELD_BLUE = 6   # NEW

ACTION_NAMES = {0:"up", 1:"down", 2:"left", 3:"right", 4:"pick", 5:"place"}


class GridEnv:
    def __init__(self, seed=None):
        self.grid_size = GRID_SIZE
        self._seed     = seed
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None):
        if seed is not None:
            self._rng = random.Random(seed)
        elif not hasattr(self, "_rng"):
            self._rng = random.Random(self._seed)

        positions = self._sample_unique(5)
        self.agent_pos    = positions[0]
        self.red_box_pos  = positions[1]
        self.blue_box_pos = positions[2]
        self.red_tgt_pos  = positions[3]
        self.blue_tgt_pos = positions[4]

        self.held_object = None   # None | "red" | "blue"
        self.done        = False
        self.steps       = 0
        self.max_steps   = 100

        return self._state()

    def step(self, action: int):
        assert not self.done, "Episode finished — call reset()"
        self.steps += 1
        reward = -0.01

        if   action in (0, 1, 2, 3): self._move(action)
        elif action == 4:             reward += self._pick()
        elif action == 5:             reward += self._place()

        red_done  = (self.red_box_pos  == self.red_tgt_pos)
        blue_done = (self.blue_box_pos == self.blue_tgt_pos)

        if red_done and blue_done:
            reward += 10.0
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True

        info = {
            "red_done":  red_done,
            "blue_done": blue_done,
            "steps":     self.steps,
            "held":      self.held_object,
        }
        return self._state(), reward, self.done, info

    def get_tensor(self, state=None):
        return state_to_tensor(state if state else self._state())

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _state(self):
        return {
            "agent_pos":    list(self.agent_pos),
            "red_box_pos":  list(self.red_box_pos),
            "blue_box_pos": list(self.blue_box_pos),
            "red_tgt_pos":  list(self.red_tgt_pos),
            "blue_tgt_pos": list(self.blue_tgt_pos),
            "held_object":  self.held_object,
        }

    def _sample_unique(self, n):
        seen, out = set(), []
        while len(out) < n:
            pos = (self._rng.randint(0, GRID_SIZE-1),
                   self._rng.randint(0, GRID_SIZE-1))
            if pos not in seen:
                seen.add(pos)
                out.append(pos)
        return out

    def _move(self, action):
        r, c = self.agent_pos
        dr, dc = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}[action]
        nr, nc = r+dr, c+dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            self.agent_pos = (nr, nc)
            if self.held_object == "red":
                self.red_box_pos  = (nr, nc)
            elif self.held_object == "blue":
                self.blue_box_pos = (nr, nc)

    def _pick(self):
        if self.held_object is not None:
            return -0.05
        if tuple(self.agent_pos) == tuple(self.red_box_pos):
            self.held_object = "red"
            return 0.5
        if tuple(self.agent_pos) == tuple(self.blue_box_pos):
            self.held_object = "blue"
            return 0.5
        return -0.05

    def _place(self):
        if self.held_object is None:
            return -0.05
        colour = self.held_object
        self.held_object = None
        # Reward regardless of correctness — success condition is environment-level
        return 0.5


# ------------------------------------------------------------------ #
# Standalone helpers
# ------------------------------------------------------------------ #

def state_to_tensor(state: dict) -> torch.Tensor:
    """Convert a state dict to a (7, 8, 8) float32 tensor."""
    grid = torch.zeros(GRID_CHANNELS, GRID_SIZE, GRID_SIZE, dtype=torch.float32)

    def _set(ch, pos):
        if pos is not None:
            grid[ch, int(pos[0]), int(pos[1])] = 1.0

    _set(CH_AGENT,    state["agent_pos"])
    _set(CH_RED_BOX,  state["red_box_pos"])
    _set(CH_BLUE_BOX, state["blue_box_pos"])
    _set(CH_RED_TGT,  state["red_tgt_pos"])
    _set(CH_BLUE_TGT, state["blue_tgt_pos"])

    held = state.get("held_object")
    ap   = state["agent_pos"]
    if held == "red":
        grid[CH_HELD_RED,  int(ap[0]), int(ap[1])] = 1.0
    elif held == "blue":
        grid[CH_HELD_BLUE, int(ap[0]), int(ap[1])] = 1.0

    return grid
