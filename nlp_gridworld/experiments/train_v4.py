"""
experiments/train_v4.py  —  Stage 3 training (FiLM + auxiliary losses)
-----------------------------------------------------------------------
Trains AgentV4FiLM with:
  I. FiLM-conditioned CNN
  J. Optional auxiliary object-prediction head (lambda_obj weight)

Usage:
    python -m experiments.train_v4
    python -m experiments.train_v4 --config experiments/configs/v4_film.yaml
    python -m experiments.train_v4 --lambda-obj 0.3 --use-aux-obj
"""

import argparse
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agent_v4_film import AgentV4FiLM
from models.auxiliary_head import derive_object_labels_batch
from utils.tokenizer import Tokenizer
from utils.dataset_utils import DemoDataset

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

DEFAULTS = {
    "data_path":       "language_demo_data.jsonl",
    "vocab_path":      "utils/vocab.json",
    "model_out":       "logs/v4_best.pth",
    "epochs":          70,
    "lr":              1e-4,
    "batch_size":      64,
    "embed_dim":       64,
    "hidden_dim":      128,
    "num_heads":       4,
    "dropout":         0.2,
    "max_len":         20,
    "augment":         True,
    "val_frac":        0.1,
    "seed":            42,
    "log_dir":         "logs",
    "use_aux_obj":     True,
    "use_aux_subgoal": False,
    "lambda_obj":      0.2,
    "lambda_subgoal":  0.1,
}


def train(cfg: dict):
    torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["log_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_v4] device={device}")

    # Tokenizer
    if os.path.exists(cfg["vocab_path"]):
        tok = Tokenizer.load(cfg["vocab_path"])
    else:
        tok = Tokenizer()
        tok.build_from_file(cfg["data_path"])
        os.makedirs(os.path.dirname(cfg["vocab_path"]), exist_ok=True)
        tok.save(cfg["vocab_path"])

    # Dataset — we also need instruction strings for aux label derivation
    train_ds, val_ds = DemoDataset.train_val_split(
        cfg["data_path"], tok,
        max_len=cfg["max_len"],
        val_frac=cfg["val_frac"],
        augment=cfg["augment"],
        seed=cfg["seed"],
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0)
    print(f"[train_v4] Train={len(train_ds)}  Val={len(val_ds)}")

    # Model
    model = AgentV4FiLM(
        vocab_size=tok.vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        use_aux_obj=cfg["use_aux_obj"],
        use_aux_subgoal=cfg["use_aux_subgoal"],
    ).to(device)
    print(f"[train_v4] AgentV4FiLM params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                   weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"])
    # ignore_index=-1 lets unknown instructions be ignored in aux loss
    ce_main    = nn.CrossEntropyLoss()
    ce_aux_obj = nn.CrossEntropyLoss(ignore_index=-1)

    run_name = f"v4_{time.strftime('%m%d_%H%M')}"
    writer   = None
    if TB_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(cfg["log_dir"], run_name))
        writer.add_text("config", yaml.dump(cfg))

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tr_loss = tr_acc = n = 0

        for tokens, grid, actions in train_loader:
            tokens, grid, actions = (
                tokens.to(device), grid.to(device), actions.to(device))

            output = model(tokens, grid)

            if isinstance(output, dict):
                action_logits = output["action"]
            else:
                action_logits = output

            # Main BC loss
            loss = ce_main(action_logits, actions)

            # Auxiliary object loss
            if cfg["use_aux_obj"] and isinstance(output, dict) and "obj" in output:
                # Decode instruction strings from token ids for label derivation
                instrs = [tok.decode(tokens[i].cpu()) for i in range(tokens.size(0))]
                obj_labels = derive_object_labels_batch(instrs).to(device)
                loss = loss + cfg["lambda_obj"] * ce_aux_obj(output["obj"], obj_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs       = actions.size(0)
            tr_loss += loss.item() * bs
            tr_acc  += (action_logits.argmax(1) == actions).float().sum().item()
            n       += bs
            global_step += 1
            if writer and global_step % 20 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)

        scheduler.step()

        # Validation (action loss only)
        model.eval()
        vl_loss = vl_acc = vn = 0
        with torch.no_grad():
            for tokens, grid, actions in val_loader:
                tokens, grid, actions = (
                    tokens.to(device), grid.to(device), actions.to(device))
                output = model(tokens, grid)
                logits = output["action"] if isinstance(output, dict) else output
                loss_v = ce_main(logits, actions)
                bs     = actions.size(0)
                vl_loss += loss_v.item() * bs
                vl_acc  += (logits.argmax(1) == actions).float().sum().item()
                vn      += bs

        avg_tl = tr_loss / n;   avg_ta = tr_acc / n
        avg_vl = vl_loss / vn;  avg_va = vl_acc / vn
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  "
              f"train loss={avg_tl:.4f} acc={avg_ta:.4f}  "
              f"val loss={avg_vl:.4f} acc={avg_va:.4f}")

        if writer:
            writer.add_scalars("loss", {"train": avg_tl, "val": avg_vl}, epoch)
            writer.add_scalars("acc",  {"train": avg_ta, "val": avg_va}, epoch)

        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save({
                "model_state": model.state_dict(),
                "vocab_path":  cfg["vocab_path"],
                "cfg":         cfg,
                "epoch":       epoch,
                "val_loss":    avg_vl,
                "model_class": "AgentV4FiLM",
                "run_name":    run_name,
            }, cfg["model_out"])
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

    if writer:
        writer.close()
    print(f"\n[train_v4] Done. → {cfg['model_out']}")


def parse_args():
    p = argparse.ArgumentParser(description="Train AgentV4FiLM (Stage 3)")
    p.add_argument("--config", type=str, default=None)
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_','-')}", action="store_true", default=v)
        else:
            p.add_argument(f"--{k.replace('_','-')}", type=t, default=v)
    args = vars(p.parse_args())
    if args["config"]:
        with open(args["config"]) as f:
            yaml_cfg = yaml.safe_load(f)
        cfg = {**DEFAULTS, **yaml_cfg, **{k: args[k] for k in DEFAULTS if args[k] != DEFAULTS[k]}}
    else:
        cfg = {k: args[k] for k in DEFAULTS}
    return cfg


if __name__ == "__main__":
    train(parse_args())
