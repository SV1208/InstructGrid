# InstructGrid

Language-conditioned imitation learning in a spatial grid environment.
An agent learns to follow natural language instructions like
"place the red box in the blue zone" by imitating human demonstrations.

---

## Directory Structure

```
instructgrid/
├── grid_env_simple.py          ← Environment (7-channel state tensor)
├── renderer_simple.py          ← Pygame visualiser
├── language_logger.py          ← Human demo recorder
├── generate_demo_data.py       ← Oracle demo generator (4 types)
├── train.py                    ← Unified training script
├── evaluate.py                 ← Unified evaluation script
├── inspect_data.py             ← Data quality inspector
├── eval_seeds.json             ← Fixed seeds for fair comparison
├── requirements.txt
│
├── utils/
│   ├── tokenizer.py            ← Word-level tokeniser
│   ├── dataset_utils.py        ← Dataset with trajectory-based split
│   └── augment.py              ← Spatial augmentation
│
├── models/
│   ├── components.py           ← CrossAttention + FiLM building blocks
│   ├── model_cnn_gru.py        ← Model 1: CNN + GRU concat
│   ├── model_attention.py      ← Model 2: CNN + GRU + cross-attention
│   ├── model_lstm_attention.py ← Model 3: LSTM + cross-attention (memory)
│   └── model_film.py           ← Model 4: FiLM-CNN + cross-attention
│
├── configs/
│   ├── cnn_gru.yaml
│   ├── attention.yaml
│   ├── lstm_attention.yaml
│   └── film.yaml
│
├── logs/                       ← Checkpoints + TensorBoard runs
│   ├── vocab.json              ← Built automatically on first train
│   ├── cnn_gru_best.pth
│   ├── attention_best.pth
│   ├── lstm_attention_best.pth
│   └── film_best.pth
│
├── language_demo_data_human.jsonl   ← Human-recorded demos
└── language_demo_data_oracle.jsonl  ← Oracle-generated demos
```

---

## Step-by-Step Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate oracle training data
```bash
# 150 episodes per type = 600 total (takes ~10 seconds)
python generate_demo_data.py --per-type 150

# Check what was generated
python inspect_data.py
```

### 3. (Optional) Record human demonstrations
```bash
python language_logger.py
```

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

---

## Training

Train one model:
```bash
python train.py --model cnn_gru
python train.py --model attention
python train.py --model lstm_attention
python train.py --model film
```

Train all 4 (run in separate terminals or sequentially):
```bash
for model in cnn_gru attention lstm_attention film; do
    python train.py --model $model
done
```

Windows CMD equivalent:
```cmd
python train.py --model cnn_gru
python train.py --model attention
python train.py --model lstm_attention
python train.py --model film
```

Override hyperparameters:
```bash
python train.py --model cnn_gru --epochs 50 --lr 1e-4
python train.py --model film --no-augment
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

---

## Evaluation

Evaluate all trained models (visual demos + headless stats):
```bash
python evaluate.py
```

Evaluate specific models:
```bash
python evaluate.py --models cnn_gru attention
```

Headless only (no pygame window):
```bash
python evaluate.py --no-visual
```

More visual episodes per model:
```bash
python evaluate.py --visual-episodes 8
```

---

## Inspecting Data

```bash
# Stats for both files
python inspect_data.py

# Replay a random episode visually
python inspect_data.py --replay

# Replay a specific instruction type
python inspect_data.py --replay --type red_to_blue

# Only inspect human data
python inspect_data.py --file language_demo_data_human.jsonl
```

---

## The 4 Instruction Types

| Type | Example instruction | Goal |
|------|---------------------|------|
| `red_to_red` | "place the red box in the red zone" | Red box → red target |
| `blue_to_blue` | "place the blue box in the blue zone" | Blue box → blue target |
| `red_to_blue` | "place the red box in the blue zone" | Red box → blue target |
| `blue_to_red` | "place the blue box in the red zone" | Blue box → red target |

---

## The 4 Models

| Model | File | Key feature |
|-------|------|-------------|
| `cnn_gru` | `model_cnn_gru.py` | Simple concat fusion, solid baseline |
| `attention` | `model_attention.py` | Language attends over grid locations |
| `lstm_attention` | `model_lstm_attention.py` | Temporal memory, fixes looping |
| `film` | `model_film.py` | Language conditions every CNN layer |

All models include:
- 7-channel state tensor (agent + 2 boxes + 2 targets + **2 held-object channels**)
- Instruction fed at every timestep (not just step 0)
- Class-weighted loss (pick/place weighted 6× vs move)
- Label smoothing (0.1)
- Trajectory-based train/val split

---

## Key Design Decisions

**Why 7-channel state (not 5)?**
The original 5-channel tensor had no way to tell the model it was
holding a box. Adding `CH_HELD_RED` and `CH_HELD_BLUE` gives the model
the information it needs to act differently after picking up an object.

**Why weighted loss?**
In a typical episode, move actions occur ~10× more than pick/place.
Without weighting, the model learns to predict "move" almost always
and rarely picks or places — giving 0% task success.

**Why trajectory-based splits?**
Splitting individual steps randomly means steps 1–4 of an episode
are in training and step 5 in validation. The model has seen the
exact context. Trajectory-based split ensures the validation set
contains entire episodes the model has never seen any part of.

**Why oracle noise (10%)?**
The oracle always takes the optimal path. Without noise, the model
never sees how to recover from suboptimal positions — which is exactly
what happens at test time when small errors accumulate.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run all commands from the `instructgrid/` directory |
| `No data files found` | Run `python generate_demo_data.py` first |
| `pygame.error: No video mode` | Add `--no-visual` to evaluate command |
| `vocab.json not found` | Built automatically on first `train.py` run |
| CUDA out of memory | Reduce `--batch-size 32` in the train command |
| 0% success rate | Check `inspect_data.py` — pick+place should be ≥12% of actions |
