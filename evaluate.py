"""
evaluate.py — Evaluate trained InstructGrid models.

For each model:
  1. Shows NUM_VISUAL episodes with pygame rendering
     (one per instruction type, cycling through types)
  2. Runs NUM_HEADLESS episodes headlessly for statistics
  3. Prints per-model results

Finally prints a side-by-side comparison table.

Usage:
    python evaluate.py                          # all trained models
    python evaluate.py --models cnn_gru film    # specific models
    python evaluate.py --no-visual              # headless only
    python evaluate.py --visual-episodes 4      # more visual demos
    python evaluate.py --headless-seeds 100     # larger eval set
"""

import argparse
import json
import os
import time
import torch

from grid_env_simple  import GridEnv
from utils.tokenizer  import Tokenizer
from models           import build_model, ALL_MODEL_NAMES

LOG_DIR   = "logs"
VOCAB_PATH = "logs/vocab.json"
SEEDS_FILE = "eval_seeds.json"

# 4 instruction types — one sample phrasing each (for visual demos)
DEMO_INSTRUCTIONS = [
    ("red_to_red",   "place the red box in the red zone"),
    ("blue_to_blue", "place the blue box in the blue zone"),
    ("red_to_blue",  "place the red box in the blue zone"),
    ("blue_to_red",  "place the blue box in the red zone"),
]

# All evaluation instructions (varied phrasings)
EVAL_INSTRUCTIONS = [
    "place the red box in the red zone",
    "put the red block on the red target",
    "move the red box to the red area",
    "place the blue box in the blue zone",
    "put the blue block on the blue target",
    "move the blue box to the blue area",
    "place the red box in the blue zone",
    "put the red block on the blue target",
    "place the blue box in the red zone",
    "put the blue block on the red target",
]


# =====================================================================
# Goal-checking (same logic as language_logger)
# =====================================================================

def _goal_met(instruction: str, state: dict) -> bool:
    """
    True only when the correct box is physically placed on the correct
    target (not still being held by the agent).
    """
    instr = instruction.lower()
    rp = list(state["red_box_pos"])
    bp = list(state["blue_box_pos"])
    rt = list(state["red_tgt_pos"])
    bt = list(state["blue_tgt_pos"])
    held = state.get("held_object")

    red_box   = "red box"   in instr or "red block"   in instr
    blue_box  = "blue box"  in instr or "blue block"  in instr
    red_zone  = "red zone"  in instr or "red target"  in instr or "red area"  in instr
    blue_zone = "blue zone" in instr or "blue target" in instr or "blue area" in instr

    if red_box and red_zone and not blue_zone:
        return held != "red" and rp == rt
    elif red_box and blue_zone and not red_zone:
        return held != "red" and rp == bt
    elif blue_box and blue_zone and not red_zone:
        return held != "blue" and bp == bt
    elif blue_box and red_zone and not blue_zone:
        return held != "blue" and bp == rt
    return rp == rt and bp == bt


# =====================================================================
# Model loading
# =====================================================================

def load_checkpoint(model_name: str, device):
    path = os.path.join(LOG_DIR, f"{model_name}_best.pth")
    if not os.path.exists(path):
        return None, None, None

    ckpt = torch.load(path, map_location=device)
    tok  = Tokenizer.load(ckpt.get("vocab_path", VOCAB_PATH))
    cfg  = ckpt["cfg"]
    model = build_model(model_name, tok.vocab_size, cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded {model_name}  (epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss',0):.4f}  "
          f"val_acc={ckpt.get('val_acc',0):.3f})")
    return model, tok, cfg


# =====================================================================
# Single episode runner
# =====================================================================

def run_episode(model, tok, env, instruction, device,
                max_steps=100, renderer=None,
                model_name="", show_result=False):
    tokens   = tok.encode(instruction).unsqueeze(0).to(device)
    state    = env.reset()
    done     = False
    hx       = None
    info     = {}
    is_lstm  = model.MODEL_NAME == "lstm_attention"

    while not done:
        # Render
        if renderer:
            for event in __import__("pygame").event.get():
                t = event.type
                import pygame
                if t == pygame.QUIT or (
                        t == pygame.KEYDOWN and event.key == pygame.K_q):
                    renderer.close()
                    return None, None
            renderer.draw(state, instruction, info,
                          model_name=model_name)

        # Predict
        grid = env.get_tensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, hx = model.predict(tokens, grid, hx)

        state, _, done, info = env.step(action)

        # Check instruction-specific goal
        if _goal_met(instruction, state):
            done = True

    success = _goal_met(instruction, state)

    # Final render frame with result
    if renderer:
        result_str = f"{'SUCCESS' if success else 'FAIL'}  ({info.get('steps',0)} steps)"
        for _ in range(12):   # hold for ~1.5 s at 8fps
            renderer.draw(state, instruction, info,
                          model_name=model_name,
                          episode_result=result_str)
            for event in __import__("pygame").event.get():
                pass

    return success, info.get("steps", max_steps)


# =====================================================================
# Main
# =====================================================================

def main():
    p = argparse.ArgumentParser(description="Evaluate InstructGrid models")
    p.add_argument("--models",           nargs="+", default=None,
                   help="Models to evaluate (default: all trained)")
    p.add_argument("--no-visual",        action="store_true")
    p.add_argument("--visual-episodes",  type=int, default=4,
                   help="Visual episodes per model (cycles through 4 types)")
    p.add_argument("--headless-seeds",   type=int, default=50)
    p.add_argument("--max-steps",        type=int, default=100)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Which models to evaluate
    if args.models:
        model_names = args.models
    else:
        model_names = [n for n in ALL_MODEL_NAMES
                       if os.path.exists(os.path.join(LOG_DIR, f"{n}_best.pth"))]
    if not model_names:
        print("No trained models found. Run: python train.py --model <name>")
        return

    # Headless seeds
    if os.path.exists(SEEDS_FILE):
        with open(SEEDS_FILE) as f:
            all_seeds = json.load(f)
        seeds = all_seeds[:args.headless_seeds]
    else:
        seeds = list(range(args.headless_seeds))

    # Renderer (created once if visual mode)
    renderer  = None
    use_visual = not args.no_visual
    if use_visual:
        try:
            from renderer_simple import Renderer
            renderer = Renderer(fps=8)
        except Exception as e:
            print(f"[Warning] Could not open renderer ({e}) — headless only.")
            use_visual = False

    env     = GridEnv()
    results = {}

    print(f"\n{'='*65}")
    print(f"Evaluating {len(model_names)} model(s)  |  "
          f"{args.headless_seeds} headless seeds")
    print(f"{'='*65}\n")

    for model_name in model_names:
        print(f"── {model_name.upper()} {'─'*(50-len(model_name))}")
        model, tok, cfg = load_checkpoint(model_name, device)
        if model is None:
            print(f"  Skipping (no checkpoint found)")
            continue

        # ---- Visual episodes ----
        if use_visual and renderer:
            print(f"  Visual demos ({args.visual_episodes} episodes):")
            for vi in range(args.visual_episodes):
                instr_type, instr = DEMO_INSTRUCTIONS[vi % 4]
                ep_seed = seeds[vi] if vi < len(seeds) else vi * 7 + 3
                env.reset(seed=ep_seed)
                success, steps = run_episode(
                    model, tok, env, instr, device,
                    max_steps  = args.max_steps,
                    renderer   = renderer,
                    model_name = model_name,
                )
                if success is None:   # user closed window
                    renderer = None
                    use_visual = False
                    break
                tag = "✓" if success else "✗"
                print(f"    {tag} [{instr_type}]  steps={steps}")

            if renderer and use_visual:
                # Small pause between models
                print(f"\n  [Press Enter in terminal to start headless eval]", end="", flush=True)
                try:
                    input()
                except EOFError:
                    pass

        # ---- Headless bulk evaluation ----
        print(f"  Headless evaluation ({len(seeds)} seeds × {len(EVAL_INSTRUCTIONS)} instructions)…")

        successes, step_counts = 0, []
        per_type = {"red_to_red":0,"blue_to_blue":0,"red_to_blue":0,"blue_to_red":0}
        per_type_total = {k:0 for k in per_type}

        total = 0
        for seed in seeds:
            for instr in EVAL_INSTRUCTIONS:
                env.reset(seed=seed)
                success, steps = run_episode(
                    model, tok, env, instr, device,
                    max_steps=args.max_steps,
                )
                if success is None:
                    success = False
                successes   += int(success)
                step_counts += [steps] if success else []
                total       += 1

                # Per-type tracking
                i = instr.lower()
                if   "red box" in i and "red zone"  in i: k="red_to_red"
                elif "blue box" in i and "blue zone" in i: k="blue_to_blue"
                elif "red box" in i and "blue zone"  in i: k="red_to_blue"
                elif "blue box" in i and "red zone"  in i: k="blue_to_red"
                else: k=None
                if k:
                    per_type_total[k] += 1
                    if success:
                        per_type[k] += 1

        sr        = successes / total * 100
        mean_steps = sum(step_counts)/len(step_counts) if step_counts else float("nan")

        print(f"\n  Results:")
        print(f"    Success rate : {sr:.1f}%  ({successes}/{total})")
        print(f"    Mean steps   : {mean_steps:.1f}")
        print(f"    Per-type success:")
        for k, v in per_type.items():
            t = per_type_total[k]
            print(f"      {k:<16} {v:3d}/{t:3d}  ({v/t*100:.0f}%)")
        print()

        results[model_name] = {
            "success_rate": sr,
            "mean_steps":   mean_steps,
            "per_type":     {k: per_type[k]/per_type_total[k]*100
                             for k in per_type},
        }

    # ---- Comparison table ----
    if len(results) > 1:
        print(f"\n{'='*65}")
        print(f"{'Model':<20} {'SuccRate':>10} {'MeanSteps':>12}"
              f"  RR   BB   RB   BR")
        print(f"{'─'*65}")
        for name, r in results.items():
            pt = r["per_type"]
            print(f"{name:<20} {r['success_rate']:>9.1f}%  "
                  f"{r['mean_steps']:>10.1f}  "
                  f"{pt.get('red_to_red',0):>4.0f}%"
                  f"{pt.get('blue_to_blue',0):>5.0f}%"
                  f"{pt.get('red_to_blue',0):>5.0f}%"
                  f"{pt.get('blue_to_red',0):>5.0f}%")
        print(f"{'='*65}")
        print("  RR=red→red  BB=blue→blue  RB=red→blue  BR=blue→red")

    if renderer:
        renderer.close()


if __name__ == "__main__":
    main()
