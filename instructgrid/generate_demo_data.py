"""
generate_demo_data.py
----------------------
Oracle demonstration generator for 4 instruction types:
  red_to_red    — place red box on red target
  blue_to_blue  — place blue box on blue target
  red_to_blue   — place red box on blue target
  blue_to_red   — place blue box on red target

Features:
  - Balanced: equal episodes per type
  - Multiple phrasings per type
  - 10% oracle noise (random action) so the model learns recovery,
    not just the single shortest path
  - Saves to language_demo_data_oracle.jsonl (separate from human data)

Usage:
    python generate_demo_data.py
    python generate_demo_data.py --per-type 200
    python generate_demo_data.py --stats-only
    python generate_demo_data.py --per-type 100 --noise 0.05
"""

import argparse
import json
import os
import random
from collections import Counter

from grid_env_simple import GridEnv

ORACLE_FILE = "language_demo_data_oracle.jsonl"

# =====================================================================
# Instruction catalogue — all valid phrasings per type
# =====================================================================

INSTRUCTION_CATALOGUE = {
    "red_to_red": [
        "place the red box in the red zone",
        "put the red box on the red target",
        "put the red block on the red target",
        "move the red box to the red area",
        "take the red box to the red zone",
        "bring the red box to the red target",
        "drop the red box in the red zone",
        "carry the red box to the red target",
    ],
    "blue_to_blue": [
        "place the blue box in the blue zone",
        "put the blue box on the blue target",
        "put the blue block on the blue target",
        "move the blue box to the blue area",
        "take the blue box to the blue zone",
        "bring the blue box to the blue target",
        "drop the blue box in the blue zone",
        "carry the blue box to the blue target",
    ],
    "red_to_blue": [
        "place the red box in the blue zone",
        "put the red box on the blue target",
        "put the red block on the blue target",
        "move the red box to the blue area",
        "take the red box to the blue zone",
        "bring the red box to the blue target",
        "drop the red box in the blue zone",
    ],
    "blue_to_red": [
        "place the blue box in the red zone",
        "put the blue box on the red target",
        "put the blue block on the red target",
        "move the blue box to the red area",
        "take the blue box to the red zone",
        "bring the blue box to the red target",
        "drop the blue box in the red zone",
    ],
}

# (box_colour, target_colour) for each type
TYPE_GOALS = {
    "red_to_red":   ("red",  "red"),
    "blue_to_blue": ("blue", "blue"),
    "red_to_blue":  ("red",  "blue"),
    "blue_to_red":  ("blue", "red"),
}


# =====================================================================
# Oracle policy
# =====================================================================

def _dir(dr: int, dc: int) -> int:
    """Largest delta → action (Manhattan-distance step)."""
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 1
    return 2 if dc < 0 else 3


def _box_pos(state, colour):
    return list(state["red_box_pos"] if colour == "red" else state["blue_box_pos"])


def _tgt_pos(state, colour):
    return list(state["red_tgt_pos"] if colour == "red" else state["blue_tgt_pos"])


def _on_target(state, box_colour, tgt_colour):
    """
    True only when the box is physically placed on the target — NOT
    when the agent is merely carrying it while standing on the target.
    Without the held check, the episode ends the moment the agent steps
    onto the target holding the box, before the place action fires.
    """
    if state.get("held_object") == box_colour:
        return False   # still carrying — doesn't count as placed
    return _box_pos(state, box_colour) == _tgt_pos(state, tgt_colour)


def oracle_action(state: dict, box_colour: str, tgt_colour: str) -> int:
    """Optimal action toward placing box_colour on tgt_colour."""
    agent = list(state["agent_pos"])
    held  = state["held_object"]

    if held is None:
        # Navigate to the box, then pick
        box = _box_pos(state, box_colour)
        if agent == box:
            return 4   # pick
        dr = box[0] - agent[0]
        dc = box[1] - agent[1]
        return _dir(dr, dc)

    elif held == box_colour:
        # Navigate to the target, then place
        tgt = _tgt_pos(state, tgt_colour)
        if agent == tgt:
            return 5   # place
        dr = tgt[0] - agent[0]
        dc = tgt[1] - agent[1]
        return _dir(dr, dc)

    else:
        # Holding the wrong box — put it down and go get the right one
        return 5   # place (wherever we are)


# =====================================================================
# Episode runner with noise
# =====================================================================

def _ser(state: dict) -> dict:
    return {k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in state.items()}


def run_oracle_episode(env: GridEnv, type_key: str, instruction: str,
                       seed: int, noise: float = 0.10,
                       max_steps: int = 150) -> list:
    """
    Run one oracle episode.  noise = probability of random action.
    Returns list of transitions on success, [] on failure.
    """
    box_colour, tgt_colour = TYPE_GOALS[type_key]
    rng   = random.Random(seed * 1337 + 7)
    state = env.reset(seed=seed)
    buf   = []

    for step in range(max_steps):
        # 10% noise: take a random action instead of optimal
        if rng.random() < noise:
            action = rng.randint(0, 5)
        else:
            action = oracle_action(state, box_colour, tgt_colour)

        next_state, _, env_done, _ = env.step(action)

        buf.append({
            "instruction": instruction if step == 0 else "",
            "state":       _ser(state),
            "action":      action,
            "next_state":  _ser(next_state),
        })
        state = next_state

        if _on_target(state, box_colour, tgt_colour):
            return buf   # success

        if env_done:
            return []    # timed out

    return []


# =====================================================================
# Stats
# =====================================================================

def print_stats(path: str):
    if not os.path.exists(path):
        print(f"[stats] {path} does not exist yet.")
        return
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    episodes = [l for l in lines if l.get("instruction")]
    counts   = Counter(l["instruction"] for l in episodes)

    print(f"\n{'─'*55}")
    print(f"File : {path}")
    print(f"  Transitions : {len(lines):,}")
    print(f"  Episodes    : {len(episodes):,}")

    # Group by type
    type_counts = Counter()
    for instr in (l["instruction"] for l in episodes):
        for key, phrasings in INSTRUCTION_CATALOGUE.items():
            if instr in phrasings:
                type_counts[key] += 1
                break

    for key in TYPE_GOALS:
        n   = type_counts.get(key, 0)
        bar = "█" * (n // 2)
        print(f"  {key:<16} {n:4d}  {bar}")

    # Action distribution
    action_counts = Counter(l["action"] for l in lines)
    names = {0:"up",1:"down",2:"left",3:"right",4:"pick",5:"place"}
    print(f"\n  Action distribution:")
    for a in range(6):
        n = action_counts.get(a, 0)
        pct = n / len(lines) * 100 if lines else 0
        print(f"    {names[a]:<6} {n:6d}  ({pct:5.1f}%)")
    print(f"{'─'*55}")


# =====================================================================
# Main
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-type",   type=int,   default=150,
                   help="Successful episodes per instruction type")
    p.add_argument("--out",        type=str,   default=ORACLE_FILE)
    p.add_argument("--noise",      type=float, default=0.10,
                   help="Probability of random action (oracle noise)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--append",     action="store_true",
                   help="Append to existing file instead of overwriting")
    args = p.parse_args()

    if args.stats_only:
        print_stats(args.out)
        return

    print_stats(args.out)

    env  = GridEnv()
    rng  = random.Random(args.seed)
    all_demos = []
    total_target = args.per_type * len(TYPE_GOALS)

    print(f"\nGenerating {args.per_type} episodes × {len(TYPE_GOALS)} types"
          f" = {total_target} total  (noise={args.noise:.0%})\n")

    for type_key in TYPE_GOALS:
        phrasings  = INSTRUCTION_CATALOGUE[type_key]
        count      = 0
        tried      = 0
        type_demos = []

        while count < args.per_type:
            ep_seed = rng.randint(0, 999_999)
            instr   = rng.choice(phrasings)
            ep      = run_oracle_episode(env, type_key, instr,
                                         seed=ep_seed, noise=args.noise)
            tried  += 1
            if ep:
                type_demos.extend(ep)
                count += 1

        avg = len(type_demos) / count if count else 0
        sr  = count / tried * 100
        all_demos.extend(type_demos)
        print(f"  {type_key:<16}  {count} eps  "
              f"{len(type_demos):5d} transitions  "
              f"avg {avg:.1f} steps  "
              f"({sr:.0f}% success)")

    mode = "a" if args.append else "w"
    with open(args.out, mode) as f:
        for d in all_demos:
            f.write(json.dumps(d) + "\n")

    n_ep = sum(1 for d in all_demos if d.get("instruction"))
    print(f"\nWrote {len(all_demos):,} transitions "
          f"({n_ep} episodes) → {args.out}")

    print_stats(args.out)


if __name__ == "__main__":
    main()
