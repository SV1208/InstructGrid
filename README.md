# InstructGrid

**Language-conditioned imitation learning in a spatial grid environment.**
An agent learns to follow natural-language instructions such as
"place the red box in the blue zone" purely by imitating demonstrations,
with no reinforcement learning and no large language model.

> **Paper:** *InstructGrid: Language-Conditioned Imitation Learning in Spatial Environments*
> Sujal Verma\*, Shivanshu Vishwakarma\*, Santosh Kumar. IIIT Naya Raipur, India (\*equal contribution).
> Accepted at **NLPIR 2026**, Nara, Japan (paper NJ017).

---

## Key result

Four ways of combining the instruction with the grid were compared, using the
same language encoder and the same data. The three that merge language *late*
(after the CNN) stall below 30% task success. **FiLM**, which feeds the
instruction into every convolutional layer, reaches **95.8%**.

| Model | Val acc | Success | Red→Red | Blue→Blue | Red→Blue | Blue→Red |
|---|---|---|---|---|---|---|
| CNN+GRU | 75.1% | 23.6% | 32% | 10% | 34% | 24% |
| Cross-Attention | 75.7% | 26.4% | 36% | 20% | 32% | 20% |
| LSTM+Attention | 75.3% | 27.2% | 30% | 24% | 36% | 22% |
| **FiLM** | **90.3%** | **95.8%** | **96%** | **96%** | **96%** | **94%** |

Success is measured on 500 episodes per model: 50 fixed layouts × 10 instructions,
none of them seen in training. Trained on 3,200 oracle demonstrations (800 per instruction type).

---

## How it works

- **Environment:** an 8×8 grid with an agent, a red box, a blue box, a red zone and a blue zone.
  6 actions: up, down, left, right, pick, place.
- **State:** a 7-channel 8×8 tensor. Channels 0–4 mark the agent, the boxes and the zones;
  channels 5–6 mark whether the agent is holding the red or the blue box.
- **Data:** a scripted oracle solves each task, taking a random action 10% of the time so
  the demonstrations include recoveries from mistakes.
- **Language:** a word-level tokenizer and an attentive GRU turn the instruction into a
  256-dimensional vector, shared by all four models.
- **Models:** see [The 4 models](#the-4-models) below.

---

## Repository layout

All code lives at the repository root; run every command from there.

```
InstructGrid/
├── grid_env_simple.py            ← Environment (7-channel state tensor)
├── renderer_simple.py            ← Pygame visualiser
├── generate_demo_data.py         ← Oracle demo generator (4 instruction types)
├── language_logger.py            ← Record human demos with the keyboard
├── inspect_data.py               ← Data quality checks and episode replay
├── train.py                      ← Train one model from configs/<model>.yaml
├── evaluate.py                   ← Visual + headless evaluation of trained models
├── manual_evaluate.py            ← Type an instruction, watch the agent do it
├── log_evaluator.py              ← Plot train/val loss from a TensorBoard run
├── InstructGrid_walkthrough.ipynb← Short walkthrough of the whole pipeline
├── eval_seeds.json               ← Fixed evaluation seeds (fair comparison)
├── language_demo_data_oracle.jsonl  ← Oracle dataset used in the paper (3,200 episodes)
├── notebook_walkthrough_*        ← Small dataset + vocab used by the notebook
├── requirements.txt
│
├── models/
│   ├── __init__.py               ← build_model() registry
│   ├── components.py             ← Attentive GRU, cross-attention, FiLM, aux colour head
│   ├── model_cnn_gru.py
│   ├── model_attention.py
│   ├── model_lstm_attention.py
│   └── model_film.py
│
├── utils/
│   ├── tokenizer.py              ← Word-level tokenizer
│   ├── dataset_utils.py          ← Dataset with trajectory-level train/val split
│   └── augment.py                ← Flip augmentation (actions remapped)
│
├── configs/                      ← One YAML of hyperparameters per model
│
└── logs/                         ← Created by train.py (not in the repo)
    ├── vocab.json
    ├── <model>_best.pth          ← Best checkpoint per model
    └── <model>_<MMDD_HHMM>/      ← TensorBoard runs
```

> **Note:** `logs/` is not committed, so trained checkpoints are not included.
> Train the models first (see below). Training runs fine on a CPU.

---

## Setup

Tested with Python 3.11.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
pip install matplotlib         # only needed for log_evaluator.py
```

---

## Quick start

The oracle dataset used in the paper is already included, so you can train and evaluate straight away:

```bash
python train.py --model film
python evaluate.py --models film --no-visual
```

---

## Data

### Generate oracle demonstrations
```bash
# Default: 150 episodes per type = 600 total
python generate_demo_data.py --per-type 150

# The paper used 800 per type = 3,200 total
python generate_demo_data.py --per-type 800

# Only print statistics of the existing file
python generate_demo_data.py --stats-only
```
Generating writes to `language_demo_data_oracle.jsonl` by default, **replacing the included
dataset**. Use `--out <file>` to write elsewhere, or `--append` to add to it.

### Record human demonstrations (optional)
```bash
python language_logger.py
```
Saved to `language_demo_data_human.jsonl`; `train.py` uses it automatically if it exists.

**Controls in the game window:**

| Key | Action |
|-----|--------|
| W / ↑ | Move up |
| S / ↓ | Move down |
| A / ← | Move left |
| D / → | Move right |
| P | Pick up box |
| L | Place box |
| R | Reset episode (discard) |
| Q / ESC | Quit and save |

**In the instruction prompt:**
- Keys **1–4** fill a preset instruction instantly
- **Ctrl+V** to paste from clipboard
- **↑ / ↓** arrows to cycle through instruction history
- **Enter** to confirm (empty → uses preset 1)

### Inspect data
```bash
python inspect_data.py                                  # stats for both files
python inspect_data.py --replay                         # replay a random episode
python inspect_data.py --replay --type red_to_blue      # replay one instruction type
python inspect_data.py --file language_demo_data_human.jsonl
```

---

## Training

```bash
python train.py --model cnn_gru
python train.py --model attention
python train.py --model lstm_attention
python train.py --model film
```

Override hyperparameters from the command line:
```bash
python train.py --model cnn_gru --epochs 50 --lr 1e-4
python train.py --model film --no-augment
```

Each run saves the epoch with the lowest validation loss to `logs/<model>_best.pth`
and writes a TensorBoard run to `logs/<model>_<MMDD_HHMM>/`:
```bash
tensorboard --logdir logs/
# open http://localhost:6006
```

---

## Evaluation

### All trained models (visual demos + headless statistics)
```bash
python evaluate.py                                # all models
python evaluate.py --models cnn_gru attention     # selected models
python evaluate.py --no-visual                    # headless only (no pygame window)
python evaluate.py --visual-episodes 8            # more visual episodes per model
```
The headless run uses 50 seeds from `eval_seeds.json` × 10 instructions = 500 episodes,
the protocol behind the results table.

### Interactive: type your own instruction
```bash
python manual_evaluate.py                  # FiLM model (default)
python manual_evaluate.py --model cnn_gru
```

### Loss curve
`log_evaluator.py` plots the training and validation loss of one TensorBoard run.
Set `base_path` at the top of the file to your run (for example `logs/film_<MMDD_HHMM>`), then:
```bash
python log_evaluator.py
```

### Notebook
`InstructGrid_walkthrough.ipynb` walks through the environment, data, models and a short
training run, using the small `notebook_walkthrough_*` files.

---

## The 4 instruction types

| Type | Example instruction | Goal |
|------|---------------------|------|
| `red_to_red` | "place the red box in the red zone" | Red box → red zone |
| `blue_to_blue` | "place the blue box in the blue zone" | Blue box → blue zone |
| `red_to_blue` | "place the red box in the blue zone" | Red box → blue zone |
| `blue_to_red` | "place the blue box in the red zone" | Blue box → red zone |

Each type has 7–8 phrasings (place / put / move / take / bring / drop / carry,
box / block, zone / target / area).

---

## The 4 models

| Model | File | Where the instruction meets the grid |
|-------|------|--------------------------------------|
| `cnn_gru` | `model_cnn_gru.py` | After the CNN, by concatenation (baseline) |
| `attention` | `model_attention.py` | After the CNN, by cross-attention over grid cells |
| `lstm_attention` | `model_lstm_attention.py` | Cross-attention followed by an LSTM |
| `film` | `model_film.py` | Inside every conv layer (FiLM: per-channel scale γ and shift β), then cross-attention |

All models share:
- 7-channel state tensor, including the two "holding" channels
- Attentive-GRU instruction encoder, with the instruction given at every step
- Auxiliary colour loss (predicts which box colour the instruction names, weight 0.3)
- Class-weighted loss (pick ×6, place ×8 versus move ×1) with label smoothing 0.1
- Trajectory-level train/validation split and flip augmentation

---

## Key design decisions

**Why a 7-channel state (not 5)?**
With 5 channels, standing on a box and carrying it look identical, so the agent picked up
the box and then froze. `CH_HELD_RED` and `CH_HELD_BLUE` tell the model what it is holding.

**Why a weighted loss?**
Moves are about 82% of all actions. Without weighting, the model learns to predict "move"
almost always, never picks or places, and gets 0% task success.

**Why trajectory-level splits?**
A random split by step would put steps 1–4 of an episode in training and step 5 in validation,
so the model would already have seen that exact layout. Splitting by whole episodes keeps
every validation layout unseen.

**Why 10% oracle noise?**
The oracle always takes the shortest path. Without noise, the model never sees how to recover
from a wrong step, which is exactly what happens at test time as small errors accumulate.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run all commands from the repository root |
| `No data files found` | Run `python generate_demo_data.py` first |
| No trained models found | Train first: `python train.py --model film` |
| `pygame.error: No video mode` | Add `--no-visual` to the evaluate command |
| `vocab.json not found` | Built automatically on the first `train.py` run |
| `UnicodeEncodeError` when redirecting output (Windows) | Set `PYTHONIOENCODING=utf-8` |
| CUDA out of memory | Reduce the batch size, e.g. `--batch-size 32` |
| 0% success rate | Check `inspect_data.py`: pick + place should be ≥ 12% of actions |

---

## Citation

The citation will be added here once the NLPIR 2026 proceedings are published.
