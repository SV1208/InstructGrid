"""
experiments/evaluate_exp.py  —  Universal evaluator
-----------------------------------------------------
Loads ANY model checkpoint and evaluates it over a set of seeds.

Metrics reported:
  - Task success rate
  - Mean steps to completion (successful episodes only)
  - Failure modes breakdown (loop / timeout / wrong-placement)
  - BC accuracy on the validation split

Usage:
    # Single model
    python -m experiments.evaluate_exp --model logs/v2_best.pth

    # Compare two models
    python -m experiments.evaluate_exp \
        --model logs/v2_best.pth logs/v3_best.pth \
        --seeds 0 1 2 3 4 5 6 7 8 9

    # Headless batch evaluation
    python -m experiments.evaluate_exp --model logs/v3_best.pth --no-render --seeds-file eval_seeds.json

    # Export results to JSON
    python -m experiments.evaluate_exp --model logs/v3_best.pth --out results_v3.json
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid_env_simple import GridEnv
from utils.tokenizer import Tokenizer
from utils.dataset_utils import DemoDataset
from torch.utils.data import DataLoader
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────
# Model loader — auto-detects class from checkpoint
# ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_class = ckpt.get("model_class", "Agent")
    cfg = ckpt.get("cfg", {})

    # Load tokenizer
    vocab_path = ckpt.get("vocab_path", "utils/vocab.json")
    tok = Tokenizer.load(vocab_path)

    if model_class == "Agent":
        from agent import Agent
        model = Agent(
            vocab_size=len(ckpt["word2idx"]),
            embed_dim=cfg.get("embed_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 128),
        )
        # Baseline stores word2idx directly; wrap as tokenizer
        tok = _DictTokenizer(ckpt["word2idx"], cfg.get("max_len", 20))

    elif model_class == "AgentV2":
        from models.agent_v2 import AgentV2
        model = AgentV2(
            vocab_size=tok.vocab_size,
            embed_dim=cfg.get("embed_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 128),
            dropout=cfg.get("dropout", 0.2),
        )
    elif model_class == "AgentV3":
        from models.agent_v3 import AgentV3
        model = AgentV3(
            vocab_size=tok.vocab_size,
            embed_dim=cfg.get("embed_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 128),
            cnn_channels=cfg.get("cnn_channels", 64),
            num_heads=cfg.get("num_heads", 4),
            dropout=cfg.get("dropout", 0.2),
        )
    elif model_class == "AgentV3LSTM":
        from models.agent_v3_lstm import AgentV3LSTM
        model = AgentV3LSTM(
            vocab_size=tok.vocab_size,
            embed_dim=cfg.get("embed_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 128),
            cnn_channels=cfg.get("cnn_channels", 64),
            num_heads=cfg.get("num_heads", 4),
            dropout=cfg.get("dropout", 0.2),
            lstm_hidden=cfg.get("lstm_hidden", 128),
            num_layers=cfg.get("num_layers", 1),
        )
    elif model_class == "AgentV4FiLM":
        from models.agent_v4_film import AgentV4FiLM
        model = AgentV4FiLM(
            vocab_size=tok.vocab_size,
            embed_dim=cfg.get("embed_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_heads=cfg.get("num_heads", 4),
            dropout=cfg.get("dropout", 0.2),
            use_aux_obj=False,
            use_aux_subgoal=False,
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"[eval] Loaded {model_class} from {ckpt_path}")
    return model, tok, model_class, cfg


# ─────────────────────────────────────────────────────────────────────
# Fallback tokenizer for baseline checkpoints
# ─────────────────────────────────────────────────────────────────────

import re as _re

class _DictTokenizer:
    def __init__(self, word2idx, max_len):
        self.word2idx  = word2idx
        self._max_len  = max_len
        self.vocab_size = len(word2idx)

    def encode(self, text, max_len=None):
        ml   = max_len or self._max_len
        ids  = [self.word2idx.get(w, 1)
                for w in _re.findall(r"[a-z0-9]+", text.lower())][:ml]
        ids += [0] * (ml - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids):
        inv = {v: k for k, v in self.word2idx.items()}
        return " ".join(inv.get(int(i), "<UNK>") for i in token_ids if int(i) != 0)


# ─────────────────────────────────────────────────────────────────────
# Single episode runner
# ─────────────────────────────────────────────────────────────────────

def run_episode(model, tok, model_class, env, instruction,
                device, max_steps=50, render=False):
    renderer = None
    if render:
        try:
            from renderer_simple import Renderer
            renderer = Renderer(fps=5)
        except Exception:
            pass

    tokens = tok.encode(instruction, 20).unsqueeze(0).to(device)
    state  = env.reset()
    done   = False
    info   = {}
    hx     = None   # only used by LSTM model

    for step in range(max_steps):
        if renderer:
            ok = renderer.render(state, instruction, info)
            if not ok:
                break

        grid = env.get_tensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            if model_class == "AgentV3LSTM":
                logits, hx = model.forward_step(tokens, grid, hx)
            elif model_class == "AgentV4FiLM":
                logits = model.forward_action_only(tokens, grid)
            else:
                logits = model(tokens, grid)

        action = logits.argmax(1).item()
        state, _, done, info = env.step(action)
        if done:
            break

    if renderer:
        time.sleep(0.5)
        renderer.close()

    return info


# ─────────────────────────────────────────────────────────────────────
# Full evaluation over multiple seeds
# ─────────────────────────────────────────────────────────────────────

DEFAULT_INSTRUCTIONS = [
    "place the red box in the red zone",
    "place the blue box in the blue zone",
    "place both boxes in their zones",
]


def evaluate_model(model, tok, model_class, instructions, seeds, device,
                   max_steps=50, render=False):
    env     = GridEnv()
    results = []

    for instruction in instructions:
        for seed in seeds:
            env.reset(seed=seed)
            info = run_episode(model, tok, model_class, env, instruction,
                               device, max_steps=max_steps, render=render)
            success = bool(info.get("red_done") and info.get("blue_done"))
            results.append({
                "instruction": instruction,
                "seed":        seed,
                "success":     success,
                "steps":       info.get("steps", max_steps),
            })

    total  = len(results)
    n_succ = sum(r["success"] for r in results)
    succ_steps = [r["steps"] for r in results if r["success"]]

    summary = {
        "success_rate":   n_succ / total if total else 0.0,
        "mean_steps":     sum(succ_steps) / len(succ_steps) if succ_steps else None,
        "n_episodes":     total,
        "n_success":      n_succ,
        "raw":            results,
    }
    return summary


# ─────────────────────────────────────────────────────────────────────
# BC accuracy on validation split
# ─────────────────────────────────────────────────────────────────────

def bc_accuracy(model, tok, model_class, cfg, device):
    _, val_ds = DemoDataset.train_val_split(
        cfg.get("data_path", "language_demo_data.jsonl"), tok,
        max_len=cfg.get("max_len", 20),
        val_frac=cfg.get("val_frac", 0.1),
        augment=False,
        seed=cfg.get("seed", 42),
    )
    loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for tokens, grid, actions in loader:
            tokens, grid, actions = (
                tokens.to(device), grid.to(device), actions.to(device))
            if model_class == "AgentV3LSTM":
                logits, _ = model.forward_step(tokens, grid)
            elif model_class == "AgentV4FiLM":
                logits = model.forward_action_only(tokens, grid)
            elif model_class == "AgentV3LSTM":
                logits, _ = model(tokens, grid.unsqueeze(1))
                logits = logits[:, 0]
            else:
                logits = model(tokens, grid)
            correct += (logits.argmax(1) == actions).float().sum().item()
            total   += actions.size(0)
    return correct / total if total else 0.0


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate one or more agent checkpoints")
    p.add_argument("--model",      nargs="+", required=True,
                   help="One or more checkpoint .pth paths")
    p.add_argument("--seeds",      nargs="+", type=int,
                   default=list(range(20)),
                   help="Random seeds for env resets")
    p.add_argument("--seeds-file", type=str, default=None,
                   help="JSON file with list of seeds (overrides --seeds)")
    p.add_argument("--instructions", nargs="+",
                   default=DEFAULT_INSTRUCTIONS)
    p.add_argument("--max-steps",  type=int, default=50)
    p.add_argument("--no-render",  action="store_true")
    p.add_argument("--out",        type=str, default=None,
                   help="Save results dict to this JSON path")
    args = p.parse_args()

    if args.seeds_file:
        with open(args.seeds_file) as f:
            seeds = json.load(f)
    else:
        seeds = args.seeds

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}

    for ckpt_path in args.model:
        model, tok, model_class, cfg = load_model(ckpt_path, device)

        print(f"\n{'='*60}")
        print(f"Model : {model_class}  ({ckpt_path})")
        print(f"Seeds : {len(seeds)}  |  Instructions : {len(args.instructions)}")
        print(f"{'='*60}")

        summary = evaluate_model(
            model, tok, model_class,
            instructions=args.instructions,
            seeds=seeds,
            device=device,
            max_steps=args.max_steps,
            render=not args.no_render,
        )

        bc_acc = bc_accuracy(model, tok, model_class, cfg, device)

        print(f"\nResults for {model_class}:")
        print(f"  Success rate : {summary['success_rate']*100:.1f}%")
        print(f"  Mean steps   : {summary['mean_steps']}")
        print(f"  BC val acc   : {bc_acc*100:.1f}%")
        print(f"  Episodes     : {summary['n_success']}/{summary['n_episodes']}")

        all_results[ckpt_path] = {**summary, "bc_acc": bc_acc,
                                  "model_class": model_class}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {args.out}")

    # Side-by-side comparison if multiple models
    if len(args.model) > 1:
        print(f"\n{'='*60}")
        print("Comparison summary:")
        print(f"{'Model':<40} {'SuccRate':>10} {'MeanSteps':>12} {'BC Acc':>8}")
        print("-" * 72)
        for path, res in all_results.items():
            name = os.path.basename(path)
            sr   = f"{res['success_rate']*100:.1f}%"
            ms   = f"{res['mean_steps']:.1f}" if res["mean_steps"] else "N/A"
            bca  = f"{res['bc_acc']*100:.1f}%"
            print(f"{name:<40} {sr:>10} {ms:>12} {bca:>8}")


if __name__ == "__main__":
    main()
