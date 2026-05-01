"""
inspect_data.py
----------------
Inspect and visualise the demonstration dataset.

Shows:
  - Episode counts per instruction type and per file
  - Action distribution (moves vs pick vs place)
  - Average episode length
  - Replay a random episode in pygame (optional)

Usage:
    python inspect_data.py
    python inspect_data.py --replay           # watch a random episode
    python inspect_data.py --replay --type red_to_blue
    python inspect_data.py --file language_demo_data_human.jsonl
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

HUMAN_FILE  = "language_demo_data_human.jsonl"
ORACLE_FILE = "language_demo_data_oracle.jsonl"

TYPE_KEYS = {
    "red_to_red":   lambda i: ("red box" in i or "red block" in i)
                              and ("red zone" in i or "red target" in i or "red area" in i)
                              and "blue zone" not in i and "blue target" not in i,
    "blue_to_blue": lambda i: ("blue box" in i or "blue block" in i)
                              and ("blue zone" in i or "blue target" in i or "blue area" in i)
                              and "red zone" not in i and "red target" not in i,
    "red_to_blue":  lambda i: ("red box" in i or "red block" in i)
                              and ("blue zone" in i or "blue target" in i or "blue area" in i),
    "blue_to_red":  lambda i: ("blue box" in i or "blue block" in i)
                              and ("red zone" in i or "red target" in i or "red area" in i),
}


def classify(instr: str) -> str:
    il = instr.lower()
    for key, fn in TYPE_KEYS.items():
        if fn(il):
            return key
    return "unknown"


def load_file(path: str):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_trajectories(lines: list) -> list:
    """Group lines into trajectories (split at non-empty instruction)."""
    trajs, buf = [], []
    for line in lines:
        if line.get("instruction") and buf:
            trajs.append(buf)
            buf = []
        buf.append(line)
    if buf:
        trajs.append(buf)
    return trajs


def print_file_stats(path: str, label: str):
    lines = load_file(path)
    if not lines:
        print(f"\n[{label}] — file not found or empty")
        return

    trajs = load_trajectories(lines)
    lengths = [len(t) for t in trajs]

    # Per-type counts
    type_counts = Counter()
    for t in trajs:
        instr = t[0].get("instruction", "")
        type_counts[classify(instr)] += 1

    # Action distribution
    action_counts = Counter(l["action"] for l in lines)
    names = {0:"up",1:"down",2:"left",3:"right",4:"pick",5:"place"}

    print(f"\n{'━'*55}")
    print(f"  {label}  ({path})")
    print(f"{'━'*55}")
    print(f"  Transitions : {len(lines):,}")
    print(f"  Trajectories: {len(trajs):,}")
    if lengths:
        print(f"  Ep length   : avg={sum(lengths)/len(lengths):.1f}  "
              f"min={min(lengths)}  max={max(lengths)}")

    print(f"\n  By instruction type:")
    for key in ["red_to_red","blue_to_blue","red_to_blue","blue_to_red","unknown"]:
        n   = type_counts.get(key, 0)
        bar = "█" * (n // 3) if n else ""
        print(f"    {key:<16} {n:4d}  {bar}")

    print(f"\n  Action distribution:")
    total = len(lines)
    for a in range(6):
        n   = action_counts.get(a, 0)
        pct = n / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {names[a]:<6} {n:6d}  {pct:5.1f}%  {bar}")


def replay_episode(path: str, type_filter: str | None):
    """Load a random episode and replay it in pygame."""
    lines = load_file(path)
    trajs = load_trajectories(lines)

    if type_filter:
        trajs = [t for t in trajs
                 if classify(t[0].get("instruction","")) == type_filter]
    if not trajs:
        print("No matching episodes found.")
        return

    traj  = random.choice(trajs)
    instr = traj[0]["instruction"]
    print(f"\nReplaying: '{instr}'  ({len(traj)} steps)")

    try:
        import pygame
        from renderer_simple import Renderer
        from grid_env_simple import state_to_tensor

        pygame.init()
        r = Renderer(fps=4)

        for i, step in enumerate(traj):
            state  = step["state"]
            action = step["action"]
            names  = {0:"up",1:"down",2:"left",3:"right",4:"pick",5:"place"}

            info = {"steps": i+1, "held": state.get("held_object") or "none",
                    "red_done": False, "blue_done": False}
            for event in pygame.event.get():
                pass
            r.draw(state, instr, info,
                   model_name=f"[Replay] action={names.get(action,action)}")

        import time; time.sleep(1.5)
        r.close()
    except Exception as e:
        print(f"Replay error: {e}")


def main():
    p = argparse.ArgumentParser(description="Inspect InstructGrid dataset")
    p.add_argument("--file",   type=str, default=None,
                   help="Specific file to inspect (default: both)")
    p.add_argument("--replay", action="store_true",
                   help="Replay a random episode in pygame")
    p.add_argument("--type",   type=str, default=None,
                   choices=["red_to_red","blue_to_blue","red_to_blue","blue_to_red"],
                   help="Filter replay by instruction type")
    args = p.parse_args()

    if args.file:
        print_file_stats(args.file, os.path.basename(args.file))
    else:
        print_file_stats(HUMAN_FILE,  "Human data")
        print_file_stats(ORACLE_FILE, "Oracle data")

        # Combined totals
        human  = load_file(HUMAN_FILE)
        oracle = load_file(ORACLE_FILE)
        all_lines = human + oracle
        if all_lines:
            all_trajs = load_trajectories(human) + load_trajectories(oracle)
            print(f"\n{'━'*55}")
            print(f"  COMBINED")
            print(f"{'━'*55}")
            print(f"  Transitions : {len(all_lines):,}")
            print(f"  Trajectories: {len(all_trajs):,}")
            action_counts = Counter(l["action"] for l in all_lines)
            names = {0:"up",1:"down",2:"left",3:"right",4:"pick",5:"place"}
            pick_pct  = action_counts.get(4,0) / len(all_lines) * 100
            place_pct = action_counts.get(5,0) / len(all_lines) * 100
            print(f"  Pick actions : {pick_pct:.1f}%  "
                  f"Place actions: {place_pct:.1f}%")
            print(f"  (Ideally pick+place together ≥ 12% for good training)")

    if args.replay:
        src = args.file or (HUMAN_FILE if os.path.exists(HUMAN_FILE)
                            else ORACLE_FILE)
        replay_episode(src, args.type)


if __name__ == "__main__":
    main()
