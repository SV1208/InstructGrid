"""
evaluate.py — Interactive InstructGrid evaluator.

The environment is rendered first; the user then types an instruction
and watches the FiLM agent (or any chosen model) execute it.
After each episode a new environment is generated automatically.

Usage:
    python evaluate.py                        # FiLM model (default)
    python evaluate.py --model cnn_gru        # different model
    python evaluate.py --max-steps 150
"""

import argparse
import os
import time
import torch
import pygame

from grid_env_simple import GridEnv
from utils.tokenizer import Tokenizer
from models          import build_model, ALL_MODEL_NAMES
from renderer_simple import Renderer

LOG_DIR    = "logs"
VOCAB_PATH = "logs/vocab.json"


# ── goal checker (unchanged from original) ───────────────────────────

def _goal_met(instruction: str, state: dict) -> bool:
    instr     = instruction.lower()
    rp        = list(state["red_box_pos"])
    bp        = list(state["blue_box_pos"])
    rt        = list(state["red_tgt_pos"])
    bt        = list(state["blue_tgt_pos"])
    held      = state.get("held_object")

    red_box   = "red box"   in instr or "red block"   in instr
    blue_box  = "blue box"  in instr or "blue block"  in instr
    red_zone  = "red zone"  in instr or "red target"  in instr or "red area"  in instr
    blue_zone = "blue zone" in instr or "blue target" in instr or "blue area" in instr

    if   red_box  and red_zone  and not blue_zone: return held != "red"  and rp == rt
    elif red_box  and blue_zone and not red_zone:  return held != "red"  and rp == bt
    elif blue_box and blue_zone and not red_zone:  return held != "blue" and bp == bt
    elif blue_box and red_zone  and not blue_zone: return held != "blue" and bp == rt
    return rp == rt and bp == bt


# ── checkpoint loader ─────────────────────────────────────────────────

def load_checkpoint(model_name: str, device):
    path = os.path.join(LOG_DIR, f"{model_name}_best.pth")
    if not os.path.exists(path):
        print(f"  [Error] No checkpoint found at {path}")
        return None, None, None

    ckpt  = torch.load(path, map_location=device)
    tok   = Tokenizer.load(ckpt.get("vocab_path", VOCAB_PATH))
    cfg   = ckpt["cfg"]
    model = build_model(model_name, tok.vocab_size, cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded '{model_name}'  "
          f"epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss',0):.4f}  "
          f"val_acc={ckpt.get('val_acc',0):.3f}")
    return model, tok, cfg


# ── single episode ────────────────────────────────────────────────────

def run_episode(model, tok, env, instruction, device,
                max_steps, renderer, model_name):
    """
    Run one episode.  Returns (success, steps) or (None, None) if the
    user closed the pygame window.
    """
    tokens = tok.encode(instruction).unsqueeze(0).to(device)
    state  = env._state()          # environment already reset by caller
    done   = False
    hx     = None
    info   = {}

    step = 0
    while not done and step < max_steps:
        # ── pump events / allow quit ──────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return None, None

        renderer.draw(state, instruction, info, model_name=model_name)

        # ── predict ───────────────────────────────────────────────────
        grid = env.get_tensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, hx = model.predict(tokens, grid, hx)

        state, _, done, info = env.step(action)
        step += 1

        if _goal_met(instruction, state):
            done = True

    success = _goal_met(instruction, state)

    # ── hold result on screen for ~1.5 s ─────────────────────────────
    result_str = f"{'SUCCESS ✓' if success else 'FAIL ✗'}  ({info.get('steps', step)} steps)"
    for _ in range(12):
        renderer.draw(state, instruction, info,
                      model_name=model_name,
                      episode_result=result_str)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None

    return success, info.get("steps", step)


# ── main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Interactive InstructGrid evaluator")
    p.add_argument("--model",      default="film",
                   help="Model to use (default: film)")
    p.add_argument("--max-steps",  type=int, default=100)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model, tok, cfg = load_checkpoint(args.model, device)
    if model is None:
        print("Aborting — train the model first:  python train.py --model film")
        return

    # ── set up renderer & env ─────────────────────────────────────────
    renderer = Renderer(fps=8)
    env      = GridEnv()

    episode  = 0
    seed     = 0

    print("\n" + "="*55)
    print("  InstructGrid — Interactive Mode")
    print("  Commands:  'r' = new env   'q' / 'quit' = exit")
    print("="*55)

    state = env.reset(seed=seed)
    renderer.draw(state, "(waiting for instruction…)", {}, model_name=args.model)

    while True:
        # ── flush pygame events so the window stays responsive ────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                renderer.close()
                return

        # ── get instruction from terminal ─────────────────────────────
        print(f"\nEpisode {episode + 1}  |  seed={seed}")
        print("Environment is visible in the pygame window.")
        instruction = input("Instruction (or 'r' to reshuffle / 'q' to quit): ").strip()

        if instruction.lower() in ("q", "quit", "exit"):
            break

        if instruction.lower() == "r":
            seed  = seed + 1
            state = env.reset(seed=seed)
            renderer.draw(state, "(waiting for instruction…)", {}, model_name=args.model)
            continue

        if not instruction:
            continue

        # ── run episode ───────────────────────────────────────────────
        episode += 1
        print(f"\n  Running: \"{instruction}\"")
        success, steps = run_episode(
            model, tok, env, instruction, device,
            max_steps  = args.max_steps,
            renderer   = renderer,
            model_name = args.model,
        )

        if success is None:       # window was closed mid-episode
            break

        tag = "✓ SUCCESS" if success else "✗ FAIL"
        print(f"  {tag}  —  steps={steps}")

        # ── auto-advance to a fresh environment ───────────────────────
        seed  += 1
        state  = env.reset(seed=seed)
        renderer.draw(state, "(waiting for instruction…)", {}, model_name=args.model)

    renderer.close()
    print("\nEvaluation session ended.")


if __name__ == "__main__":
    main()