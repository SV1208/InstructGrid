"""
train.py  —  BASELINE TRAINING  (do not modify)
------------------------------------------------
Trains the original Agent using behavior cloning (CrossEntropy loss).

Usage:
    python train.py
    python train.py --epochs 30 --lr 3e-4 --batch-size 64
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from agent import Agent
from grid_env_simple import state_to_tensor

# ── default hyper-parameters ──────────────────────────────────────────
DEFAULTS = dict(
    data_path  = "language_demo_data.jsonl",
    vocab_path = "vocab.json",
    model_out  = "trained_agent.pth",
    epochs     = 30,
    lr         = 3e-4,
    batch_size = 64,
    embed_dim  = 64,
    hidden_dim = 128,
    max_len    = 20,
    seed       = 42,
)


# ─────────────────────────────────────────────────────────────────────
# Minimal tokenizer (self-contained in this file to keep baseline simple)
# ─────────────────────────────────────────────────────────────────────
import re

def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_vocab(path):
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("instruction"):
                for w in _words(e["instruction"]):
                    if w not in word2idx:
                        word2idx[w] = len(word2idx)
    return word2idx

def encode(text, word2idx, max_len):
    ids = [word2idx.get(w, 1) for w in _words(text)][:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────
# Dataset  (baseline: instruction only on step 0)
# ─────────────────────────────────────────────────────────────────────
class BaselineDataset(Dataset):
    def __init__(self, path, word2idx, max_len):
        self.samples  = []
        self.word2idx = word2idx
        self.max_len  = max_len
        current_instr = ""
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("instruction"):
                    current_instr = e["instruction"]
                e["instruction"] = current_instr
                self.samples.append(e)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        tokens = encode(s["instruction"], self.word2idx, self.max_len)
        grid   = state_to_tensor(s["state"])
        action = torch.tensor(s["action"], dtype=torch.long)
        return tokens, grid, action


# ─────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────
def train(cfg):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # Vocab
    if os.path.exists(cfg["vocab_path"]):
        with open(cfg["vocab_path"]) as f:
            word2idx = json.load(f)
        print(f"[train] Loaded vocab ({len(word2idx)} tokens)")
    else:
        word2idx = build_vocab(cfg["data_path"])
        with open(cfg["vocab_path"], "w") as f:
            json.dump(word2idx, f, indent=2)
        print(f"[train] Built vocab ({len(word2idx)} tokens)")

    # Dataset + loader
    dataset = BaselineDataset(cfg["data_path"], word2idx, cfg["max_len"])
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"],
                         shuffle=True, num_workers=0)

    # Model
    model = Agent(vocab_size=len(word2idx),
                  embed_dim=cfg["embed_dim"],
                  hidden_dim=cfg["hidden_dim"]).to(device)
    print(f"[train] Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for tokens, grid, actions in loader:
            tokens, grid, actions = (tokens.to(device), grid.to(device),
                                     actions.to(device))
            logits = model(tokens, grid)
            loss   = criterion(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs         = actions.size(0)
            total_loss += loss.item() * bs
            total_acc  += (logits.argmax(1) == actions).float().sum().item()
            n          += bs

        avg_loss = total_loss / n
        avg_acc  = total_acc  / n
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  "
              f"loss={avg_loss:.4f}  acc={avg_acc:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state": model.state_dict(),
                        "word2idx": word2idx,
                        "cfg": cfg},
                       cfg["model_out"])

    print(f"\nTraining done. Best loss={best_loss:.4f}. Saved → {cfg['model_out']}")


# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    return vars(p.parse_args())


if __name__ == "__main__":
    train(parse_args())
