"""
utils/dataset_utils.py
-----------------------
Dataset class used by all experimental training scripts.

Key improvements over baseline:
  1. Instruction propagated to EVERY step (not just step 0).
  2. Optional spatial augmentation (horizontal flip).
  3. Optional label smoothing helper.
"""

import json
import torch
from torch.utils.data import Dataset

from grid_env_simple import state_to_tensor
from utils.augment import augment_grid


class DemoDataset(Dataset):
    """
    Loads language_demo_data.jsonl and returns (tokens, grid, action) triples.

    Args:
        jsonl_path  : path to the .jsonl file
        tokenizer   : Tokenizer instance
        max_len     : token sequence length (pad / truncate)
        augment     : apply random horizontal flip augmentation
        propagate   : fill instruction for every step in a trajectory
                      (set False to reproduce the baseline behaviour)
    """

    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 20,
                 augment: bool = False, propagate: bool = True):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.augment   = augment
        self.samples   = self._load(jsonl_path, propagate)

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry  = self.samples[idx]
        tokens = self.tokenizer.encode(entry["instruction"], self.max_len)
        grid   = state_to_tensor(entry["state"])
        action = int(entry["action"])

        if self.augment:
            grid, action = augment_grid(grid, action)

        return tokens, grid, torch.tensor(action, dtype=torch.long)

    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: str, propagate: bool) -> list:
        samples      = []
        current_instr = ""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("instruction"):
                    current_instr = entry["instruction"]
                if propagate:
                    entry["instruction"] = current_instr
                elif not entry.get("instruction"):
                    entry["instruction"] = current_instr   # fallback
                samples.append(entry)
        print(f"[DemoDataset] Loaded {len(samples)} transitions from {path}")
        return samples

    # ------------------------------------------------------------------
    # Class method for quick split into train / val
    # ------------------------------------------------------------------

    @classmethod
    def train_val_split(cls, jsonl_path, tokenizer, max_len=20,
                        val_frac=0.1, augment=False, seed=42):
        import random
        full = cls(jsonl_path, tokenizer, max_len=max_len,
                   augment=False, propagate=True)
        n    = len(full)
        idxs = list(range(n))
        random.Random(seed).shuffle(idxs)
        split = int(n * (1 - val_frac))
        train_ds = _SubsetDataset(full, idxs[:split], augment=augment)
        val_ds   = _SubsetDataset(full, idxs[split:], augment=False)
        return train_ds, val_ds


class _SubsetDataset(Dataset):
    """Wraps DemoDataset with a subset of indices and optional augmentation."""
    def __init__(self, base: DemoDataset, indices: list, augment: bool):
        self.base    = base
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        tokens, grid, action = self.base[self.indices[idx]]
        if self.augment:
            from utils.augment import augment_grid
            grid, action_int = augment_grid(grid, action.item())
            action = torch.tensor(action_int, dtype=torch.long)
        return tokens, grid, action
