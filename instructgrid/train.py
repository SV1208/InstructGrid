"""
train.py — Unified training script for InstructGrid.

Usage:
    python train.py --model cnn_gru
    python train.py --model attention
    python train.py --model lstm_attention
    python train.py --model film
    python train.py --model film --epochs 120 --lr 5e-5
"""

import argparse
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.tokenizer     import Tokenizer
from utils.dataset_utils import DemoDataset
from models              import build_model, ALL_MODEL_NAMES
from models.components   import derive_colour_labels
from generate_demo_data  import INSTRUCTION_CATALOGUE

from torch.utils.tensorboard import SummaryWriter


HUMAN_DATA  = "language_demo_data_human.jsonl"
ORACLE_DATA = "language_demo_data_oracle.jsonl"
VOCAB_PATH  = "logs/vocab.json"
LOG_DIR     = "logs"


# =====================================================================
# Loss
# =====================================================================

def build_criterion(cfg: dict, device):
    mw  = cfg.get("move_weight",  1.0)
    pkw = cfg.get("pick_weight",  6.0)
    plw = cfg.get("place_weight", 8.0)
    ls  = cfg.get("label_smoothing", 0.1)
    weights = torch.tensor([mw, mw, mw, mw, pkw, plw], dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=ls)


# =====================================================================
# One training epoch
# =====================================================================

def run_epoch(model, loader, criterion, aux_ce, optimizer,
              device, is_lstm, lambda_colour, train_mode: bool,
              tok):
    model.train() if train_mode else model.eval()
    total_loss = total_acc = n = 0

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for tokens, grid, actions in loader:
            tokens, grid, actions = (tokens.to(device), grid.to(device),
                                     actions.to(device))

            if is_lstm:
                out = model(tokens, grid, hx=None)
            else:
                out = model(tokens, grid)

            # out is always a dict now
            action_logits  = out["action"]
            colour_logits  = out.get("colour")

            # Main BC loss
            loss = criterion(action_logits, actions)

            # Auxiliary colour loss
            if colour_logits is not None and lambda_colour > 0:
                # Decode token ids back to strings for label derivation
                instrs      = [tok.decode(tokens[i].cpu())
                               for i in range(tokens.size(0))]
                clr_labels  = derive_colour_labels(instrs, tok).to(device)
                # ignore_index=-1 for unknown/ambiguous instructions
                aux_loss    = aux_ce(colour_logits, clr_labels)
                if not torch.isnan(aux_loss):
                    loss = loss + lambda_colour * aux_loss

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs          = actions.size(0)
            total_loss += loss.item() * bs
            total_acc  += (action_logits.argmax(1) == actions).float().sum().item()
            n          += bs

    return total_loss / n, total_acc / n


# =====================================================================
# Main
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True, choices=ALL_MODEL_NAMES)
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--batch-size",  type=int,   default=None)
    p.add_argument("--no-augment",  action="store_true")
    p.add_argument("--seed",        type=int,   default=None)
    args = p.parse_args()

    cfg_path = os.path.join("configs", f"{args.model}.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.epochs     is not None: cfg["epochs"]     = args.epochs
    if args.lr         is not None: cfg["lr"]         = args.lr
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.no_augment:             cfg["augment"]    = False
    if args.seed       is not None: cfg["seed"]       = args.seed

    torch.manual_seed(cfg["seed"])
    os.makedirs(LOG_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print(f"\n{'='*60}")
    print(f"Training: {args.model.upper()}  |  device={device}")
    print(f"{'='*60}")

    # Tokenizer
    if os.path.exists(VOCAB_PATH):
        tok = Tokenizer.load(VOCAB_PATH)
    else:
        tok = Tokenizer()
        tok.build_from_catalogue(INSTRUCTION_CATALOGUE)
        tok.save(VOCAB_PATH)

    # Dataset
    data_paths = [p for p in [HUMAN_DATA, ORACLE_DATA] if os.path.exists(p)]
    if not data_paths:
        print("ERROR: No data files found. Run generate_demo_data.py first.")
        sys.exit(1)

    train_ds, val_ds = DemoDataset.trajectory_split(
        data_paths, tok,
        max_len  = cfg["max_len"],
        val_frac = cfg["val_frac"],
        augment  = cfg["augment"],
        seed     = cfg["seed"],
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0)
    print(f"Data: {len(data_paths)} file(s)  |  "
          f"train={len(train_ds)}  val={len(val_ds)}")

    # Model
    model = build_model(args.model, tok.vocab_size, cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    is_lstm       = (args.model == "lstm_attention")
    lambda_colour = cfg.get("lambda_colour", 0.3)
    optimizer     = torch.optim.AdamW(model.parameters(),
                                      lr=cfg["lr"], weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"])
    criterion     = build_criterion(cfg, device)
    # ignore_index=-1 so unknown instructions don't contribute to aux loss
    aux_ce        = nn.CrossEntropyLoss(ignore_index=-1)

    run_name = f"{args.model}_{time.strftime('%m%d_%H%M')}"
    writer   = None
    
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))

    best_val_loss = float("inf")
    out_path      = os.path.join(LOG_DIR, f"{args.model}_best.pth")

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, aux_ce,
                                    optimizer, device, is_lstm,
                                    lambda_colour, True, tok)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, aux_ce,
                                    None, device, is_lstm,
                                    lambda_colour, False, tok)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{cfg['epochs']}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val loss={vl_loss:.4f} acc={vl_acc:.3f}")

        if writer:
            writer.add_scalars("loss", {"train": tr_loss, "val": vl_loss}, epoch)
            writer.add_scalars("acc",  {"train": tr_acc,  "val": vl_acc},  epoch)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({
                "model_name":  args.model,
                "model_state": model.state_dict(),
                "vocab_path":  VOCAB_PATH,
                "cfg":         cfg,
                "epoch":       epoch,
                "val_loss":    vl_loss,
                "val_acc":     vl_acc,
            }, out_path)
            print(f"  ✓ Saved best (val_loss={vl_loss:.4f})")

    if writer:
        writer.close()
    print(f"\nDone. Best val_loss={best_val_loss:.4f}  →  {out_path}")


if __name__ == "__main__":
    main()
