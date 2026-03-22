"""
utils/dataset_utils.py
-----------------------
Dataset that:
  - Loads from multiple .jsonl files (human + oracle)
  - Propagates instruction to every step in a trajectory
  - Splits by TRAJECTORY (not by individual step) for honest validation
"""

import json, random
import torch
from torch.utils.data import Dataset

from grid_env_simple import state_to_tensor
from utils.augment import augment_grid


class DemoDataset(Dataset):
    """
    Args:
        paths      : list of .jsonl file paths (e.g. human + oracle)
        tokenizer  : Tokenizer instance
        max_len    : token sequence length
        augment    : apply spatial augmentation
    """

    def __init__(self, paths, tokenizer, max_len: int = 20,
                 augment: bool = False):
        self.tok     = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.samples = []
        self.trajectories = []          # list of (start_idx, end_idx)

        for path in (paths if isinstance(paths, list) else [paths]):
            self._load_file(path)

        print(f"[Dataset] {len(self.samples)} transitions  "
              f"| {len(self.trajectories)} trajectories  "
              f"| {len(paths) if isinstance(paths,list) else 1} file(s)")

    def _load_file(self, path: str):
        try:
            current_instr = ""
            traj_start    = len(self.samples)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    if e.get("instruction"):
                        # New trajectory starts here
                        if len(self.samples) > traj_start:
                            self.trajectories.append(
                                (traj_start, len(self.samples)))
                        traj_start    = len(self.samples)
                        current_instr = e["instruction"]
                    e["instruction"] = current_instr
                    self.samples.append(e)
            # Close last trajectory
            if len(self.samples) > traj_start:
                self.trajectories.append((traj_start, len(self.samples)))
        except FileNotFoundError:
            print(f"[Dataset] Warning: {path} not found — skipping")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        tokens = self.tok.encode(s["instruction"], self.max_len)
        grid   = state_to_tensor(s["state"])
        action = int(s["action"])

        if self.augment:
            grid, action = augment_grid(grid, action)

        return tokens, grid, torch.tensor(action, dtype=torch.long)

    # ---------------------------------------------------------------- #
    # Trajectory-aware split
    # ---------------------------------------------------------------- #

    @classmethod
    def trajectory_split(cls, paths, tokenizer, max_len=20,
                         val_frac=0.15, augment=False, seed=42):
        """
        Split by whole trajectories — no trajectory is split across
        train/val, giving an honest validation set.
        """
        full = cls(paths, tokenizer, max_len=max_len, augment=False)
        trajs = list(full.trajectories)
        random.Random(seed).shuffle(trajs)

        split      = max(1, int(len(trajs) * val_frac))
        val_trajs  = trajs[:split]
        train_trajs = trajs[split:]

        train_ids = _traj_indices(train_trajs)
        val_ids   = _traj_indices(val_trajs)

        train_ds = _Subset(full, train_ids, augment)
        val_ds   = _Subset(full, val_ids,   False)

        print(f"[Dataset] train: {len(train_ids)} steps "
              f"({len(train_trajs)} trajs)  |  "
              f"val: {len(val_ids)} steps ({len(val_trajs)} trajs)")
        return train_ds, val_ds


def _traj_indices(trajs):
    ids = []
    for start, end in trajs:
        ids.extend(range(start, end))
    return ids


class _Subset(Dataset):
    def __init__(self, base, indices, augment):
        self.base    = base
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        tokens, grid, action = self.base[self.indices[i]]
        if self.augment:
            grid, a = augment_grid(grid, action.item())
            action  = torch.tensor(a, dtype=torch.long)
        return tokens, grid, action
