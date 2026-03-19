# NLP Grid-World Agent — Experimental Repo

A grid-world NLP instruction-following agent trained with behaviour cloning,
with a full experimental pipeline spanning three stages of architectural
improvement.

---

## Repository Structure

```
nlp_gridworld/
├── grid_env_simple.py          ← environment  (never modified)
├── renderer_simple.py          ← pygame renderer
├── language_logger.py          ← human demo recorder
├── generate_demo_data.py       ← oracle demo generator (no pygame needed)
├── agent.py                    ← baseline model  (never modified)
├── train.py                    ← baseline training  (never modified)
├── evaluate.py                 ← baseline evaluation
├── language_demo_data.jsonl    ← demonstration dataset
├── eval_seeds.json             ← fixed seeds for fair comparison
├── requirements.txt
│
├── utils/
│   ├── tokenizer.py            ← shared word tokenizer
│   ├── dataset_utils.py        ← dataset with instruction propagation
│   └── augment.py              ← spatial augmentation
│
├── models/
│   ├── agent_v2.py             ← Stage 1: LayerNorm + 3-layer CNN
│   ├── agent_v3.py             ← Stage 2: cross-attention fusion
│   ├── agent_v3_lstm.py        ← Stage 2: LSTM temporal head
│   ├── agent_v4_film.py        ← Stage 3: FiLM + auxiliary losses
│   ├── cross_attention_fusion.py
│   ├── film.py
│   └── auxiliary_head.py
│
├── experiments/
│   ├── train_v2.py
│   ├── train_v3.py
│   ├── train_v3_lstm.py
│   ├── train_v4.py
│   ├── evaluate_exp.py         ← universal evaluator (all models)
│   └── configs/
│       ├── v2_stage1.yaml
│       ├── v3_crossattn.yaml
│       ├── v3_lstm.yaml
│       └── v4_film.yaml
│
└── logs/                       ← auto-created; stores checkpoints + TB runs
```

---

## Step-by-Step Running Process

### Step 0 — Install dependencies

```bash
cd nlp_gridworld
pip install -r requirements.txt
```

Verify PyTorch is working:

```bash
python -c "import torch; print(torch.__version__)"
```

---

### Step 1 — Get demonstration data

**Option A — Record your own (requires pygame / display)**

```bash
python language_logger.py
```

The game window opens. Use:
- `W/A/S/D` or arrow keys to move
- `P` to pick up a box
- `L` to place a box
- `R` to reset (failed episode — not saved)
- `Q` or `ESC` to quit and save

You will be prompted for an instruction before each episode.
Only complete (successful) episodes are saved to `language_demo_data.jsonl`.
Aim for at least 100–200 successful episodes (500+ recommended).

**Option B — Generate oracle data (headless, fastest)**

```bash
python generate_demo_data.py --episodes 500 --out language_demo_data.jsonl
```

This uses a scripted rule-based oracle policy. Takes ~10 seconds.
Good for testing the full pipeline quickly.

---

### Step 2 — Build the vocabulary

The tokenizer is built automatically during training, but you can build it
explicitly and inspect it:

```bash
python -c "
from utils.tokenizer import Tokenizer
import os; os.makedirs('utils', exist_ok=True)
tok = Tokenizer()
tok.build_from_file('language_demo_data.jsonl')
tok.save('utils/vocab.json')
print('Vocab size:', tok.vocab_size)
"
```

---

### Step 3 — Train the baseline (optional reference point)

```bash
python train.py --epochs 30
```

Outputs: `trained_agent.pth`, `vocab.json`

Evaluate the baseline:

```bash
python evaluate.py --no-render
```

Or with the pygame window:

```bash
python evaluate.py --instruction "place the red box in the red zone"
```

---

### Step 4 — Stage 1: Train AgentV2

Improvements: instruction at every step, LayerNorm, Dropout, 3-layer CNN.

```bash
# Using defaults
python -m experiments.train_v2

# Using config file
python -m experiments.train_v2 --config experiments/configs/v2_stage1.yaml

# With augmentation (recommended if you have 200+ demos)
python -m experiments.train_v2 --augment --epochs 50
```

Output: `logs/v2_best.pth`

Expected improvement over baseline: +5–15% BC accuracy, more stable training.

---

### Step 5 — Stage 2A: Train AgentV3 (cross-attention)

Improvement: language vector attends over CNN spatial feature maps.

```bash
python -m experiments.train_v3 --config experiments/configs/v3_crossattn.yaml

# Or with CLI overrides
python -m experiments.train_v3 --epochs 60 --lr 2e-4 --num-heads 4 --augment
```

Output: `logs/v3_best.pth`

Expected improvement: agent focuses on the instruction-relevant spatial region
rather than blindly using all of the grid.

---

### Step 6 — Stage 2B: Train AgentV3LSTM (optional temporal memory)

Improvement: LSTM across timesteps gives the agent memory of what it has done.

```bash
python -m experiments.train_v3_lstm --config experiments/configs/v3_lstm.yaml
```

Output: `logs/v3_lstm_best.pth`

Note: This is most useful when the agent tends to loop. If BC accuracy is
already high from v3, this adds only marginal benefit on the BC metric
but helps at eval-time with compounding errors.

---

### Step 7 — Stage 3: Train AgentV4FiLM (FiLM + auxiliary loss)

Improvements: FiLM-conditioned CNN + auxiliary object-prediction loss.

```bash
python -m experiments.train_v4 --config experiments/configs/v4_film.yaml

# Adjust auxiliary loss weight
python -m experiments.train_v4 --lambda-obj 0.3

# Disable auxiliary loss (pure FiLM only)
python -m experiments.train_v4 --no-use-aux-obj
```

Output: `logs/v4_best.pth`

---

### Step 8 — Compare all models

```bash
# Evaluate a single model
python -m experiments.evaluate_exp --model logs/v2_best.pth --no-render

# Compare all models side by side
python -m experiments.evaluate_exp \
    --model logs/v2_best.pth logs/v3_best.pth logs/v4_best.pth \
    --seeds-file eval_seeds.json \
    --no-render \
    --out results_comparison.json

# With visual rendering (slower)
python -m experiments.evaluate_exp --model logs/v3_best.pth --seeds 0 1 2 3 4
```

The comparison table printed at the end looks like:

```
============================================================
Model                                   SuccRate    MeanSteps   BC Acc
------------------------------------------------------------------------
v2_best.pth                              72.0%         24.3    83.2%
v3_best.pth                              84.0%         21.7    87.5%
v4_best.pth                              89.0%         19.1    90.3%
```

---

### Step 9 — Monitor training with TensorBoard

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 in your browser.

You will see:
- `loss/train` and `loss/val` — should decrease together
- `acc/train` and `acc/val`  — should increase together
- If train loss drops but val loss rises: overfitting → increase dropout or
  reduce epochs.

---

## Git Branch Strategy

Keep the baseline untouched on `main`. Each experiment lives on its own branch:

```bash
# Create and switch to Stage 1 branch
git checkout -b exp/stage1-v2
# Train, evaluate, commit results
git add logs/v2_best.pth results_v2.json
git commit -m "Stage 1: AgentV2 — 84% val acc"

# Stage 2 branch
git checkout main
git checkout -b exp/stage2-crossattn
# ...
```

---

## Quick Smoke Test (no pygame, no GPU, ~2 minutes)

Run this to verify the entire pipeline is wired correctly before a full
training run:

```bash
# 1. Generate tiny dataset
python generate_demo_data.py --episodes 20 --out language_demo_data.jsonl

# 2. Build vocab
python -c "
from utils.tokenizer import Tokenizer; import os
os.makedirs('utils', exist_ok=True)
t = Tokenizer(); t.build_from_file('language_demo_data.jsonl'); t.save('utils/vocab.json')
"

# 3. Quick Stage 1 train (5 epochs)
python -m experiments.train_v2 --epochs 5 --batch-size 16

# 4. Quick eval
python -m experiments.evaluate_exp --model logs/v2_best.pth --no-render --seeds 0 1 2
```

You should see training output, a saved checkpoint, and evaluation results —
all in under 2 minutes on CPU.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: utils` | Run all commands from the `nlp_gridworld/` root directory |
| `FileNotFoundError: language_demo_data.jsonl` | Run `generate_demo_data.py` first |
| `FileNotFoundError: utils/vocab.json` | It is built automatically on first train — or run Step 2 manually |
| `pygame.error: No video mode` | Add `--no-render` flag, or set `DISPLAY=:0` |
| CUDA out of memory | Reduce `--batch-size` to 32 or 16 |
| Val loss not improving | Try more demos, increase `--epochs`, or add `--augment` |
| Training very slow | Normal on CPU for small datasets — 50 epochs takes ~2–5 minutes |
