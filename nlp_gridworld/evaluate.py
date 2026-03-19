"""
evaluate.py  —  BASELINE EVALUATION  (do not modify)
-----------------------------------------------------
Loads trained_agent.pth and runs the agent interactively in the environment.

Usage:
    python evaluate.py
    python evaluate.py --model trained_agent.pth --instruction "place the red box in the red zone"
    python evaluate.py --no-render   # headless mode
"""

import argparse
import json
import re
import torch

from agent import Agent
from grid_env_simple import GridEnv


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def encode(text, word2idx, max_len=20):
    ids = [word2idx.get(w, 1) for w in _words(text)][:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)   # (1, max_len)


def run_episode(model, word2idx, env, instruction, device,
                max_steps=50, render=True):
    renderer = None
    if render:
        try:
            from renderer_simple import Renderer
            renderer = Renderer()
        except Exception:
            render = False

    tokens = encode(instruction, word2idx).to(device)
    state  = env.reset()
    done   = False
    info   = {}

    while not done:
        if render and renderer:
            ok = renderer.render(state, instruction, info)
            if not ok:
                break

        grid    = env.get_tensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tokens, grid)
        action  = logits.argmax(1).item()
        state, _, done, info = env.step(action)

    if renderer:
        import time; time.sleep(1.0)
        renderer.close()

    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="trained_agent.pth")
    p.add_argument("--instruction", default="")
    p.add_argument("--no-render",   action="store_true")
    p.add_argument("--seed",        type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.model, map_location=device)

    word2idx = ckpt["word2idx"]
    cfg      = ckpt.get("cfg", {})

    model = Agent(vocab_size=len(word2idx),
                  embed_dim=cfg.get("embed_dim", 64),
                  hidden_dim=cfg.get("hidden_dim", 128)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[evaluate] Loaded model from {args.model}")

    instruction = args.instruction or input("Enter instruction: ").strip()
    env  = GridEnv()
    info = run_episode(model, word2idx, env, instruction, device,
                       render=not args.no_render)
    success = info.get("red_done") and info.get("blue_done")
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}  "
          f"steps={info.get('steps', '?')}")


if __name__ == "__main__":
    main()
