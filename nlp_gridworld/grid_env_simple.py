"""
grid_env_simple.py
------------------
Grid-world environment for NLP instruction-following agent.

Grid: 8x8
Objects: red_box, blue_box
Targets: red_target, blue_target
Agent: single position

Actions:
  0 = move up
  1 = move down
  2 = move left
  3 = move right
  4 = pick   (picks up the box the agent is standing on)
  5 = place  (places held box on current cell)
"""

import numpy as np
import torch
import random

GRID_SIZE   = 8
NUM_ACTIONS = 6

# Channel indices in the 5×8×8 tensor
CH_AGENT       = 0
CH_RED_BOX     = 1
CH_BLUE_BOX    = 2
CH_RED_TARGET  = 3
CH_BLUE_TARGET = 4


class GridEnv:
    def __init__(self, seed=None):
        self.grid_size = GRID_SIZE
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.RandomState(seed)

        positions = self._sample_unique_positions(5)
        self.agent_pos    = positions[0]
        self.red_box_pos  = positions[1]
        self.blue_box_pos = positions[2]
        self.red_tgt_pos  = positions[3]
        self.blue_tgt_pos = positions[4]

        self.held_object  = None   # None | "red" | "blue"
        self.done         = False
        self.steps        = 0
        self.max_steps    = 100

        return self._get_state()

    def step(self, action: int):
        assert not self.done, "Episode is finished. Call reset()."
        self.steps += 1
        reward = -0.01   # small step penalty

        if action in (0, 1, 2, 3):
            self._move(action)
        elif action == 4:
            reward += self._pick()
        elif action == 5:
            reward += self._place()

        # Check success
        red_done  = (self.red_box_pos  == self.red_tgt_pos)
        blue_done = (self.blue_box_pos == self.blue_tgt_pos)

        if red_done and blue_done:
            reward += 10.0
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done, {
            "red_done": red_done,
            "blue_done": blue_done,
            "steps": self.steps,
            "held": self.held_object,
        }

    def get_tensor(self, state=None):
        """Convert current (or given) state dict to a 5×8×8 float tensor."""
        if state is None:
            state = self._get_state()
        return state_to_tensor(state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self):
        return {
            "agent_pos":    list(self.agent_pos),
            "red_box_pos":  list(self.red_box_pos),
            "blue_box_pos": list(self.blue_box_pos),
            "red_tgt_pos":  list(self.red_tgt_pos),
            "blue_tgt_pos": list(self.blue_tgt_pos),
            "held_object":  self.held_object,
        }

    def _sample_unique_positions(self, n):
        positions = set()
        result = []
        while len(result) < n:
            pos = (self.rng.randint(0, GRID_SIZE - 1),
                   self.rng.randint(0, GRID_SIZE - 1))
            if pos not in positions:
                positions.add(pos)
                result.append(pos)
        return result

    def _move(self, action):
        r, c = self.agent_pos
        dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            self.agent_pos = (nr, nc)
            # Carry held object with agent
            if self.held_object == "red":
                self.red_box_pos = (nr, nc)
            elif self.held_object == "blue":
                self.blue_box_pos = (nr, nc)

    def _pick(self):
        if self.held_object is not None:
            return -0.1   # already holding something
        if self.agent_pos == self.red_box_pos:
            self.held_object = "red"
            return 0.1
        elif self.agent_pos == self.blue_box_pos:
            self.held_object = "blue"
            return 0.1
        return -0.1   # nothing to pick

    def _place(self):
        if self.held_object is None:
            return -0.1   # not holding anything
        placed_on_target = False
        if self.held_object == "red" and self.agent_pos == self.red_tgt_pos:
            placed_on_target = True
        elif self.held_object == "blue" and self.agent_pos == self.blue_tgt_pos:
            placed_on_target = True
        self.held_object = None
        return 1.0 if placed_on_target else -0.1


# ------------------------------------------------------------------
# Standalone helpers (used by train, evaluate, dataset, etc.)
# ------------------------------------------------------------------

def state_to_tensor(state: dict) -> torch.Tensor:
    """Convert a state dict to a 5×8×8 float32 tensor."""
    grid = torch.zeros(5, GRID_SIZE, GRID_SIZE, dtype=torch.float32)

    def _set(ch, pos):
        if pos is not None:
            grid[ch, pos[0], pos[1]] = 1.0

    _set(CH_AGENT,       state["agent_pos"])
    _set(CH_RED_BOX,     state["red_box_pos"])
    _set(CH_BLUE_BOX,    state["blue_box_pos"])
    _set(CH_RED_TARGET,  state["red_tgt_pos"])
    _set(CH_BLUE_TARGET, state["blue_tgt_pos"])
    return grid


ACTION_NAMES = {
    0: "up", 1: "down", 2: "left", 3: "right", 4: "pick", 5: "place"
}
