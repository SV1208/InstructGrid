"""
generate_demo_data.py
----------------------
Generates synthetic oracle demonstrations for ALL instruction types,
including cross-placement cases where boxes go to the opposite-colour zone.

Instruction types handled:
  1. red_to_red    — "place the red box in the red zone"
  2. blue_to_blue  — "place the blue box in the blue zone"
  3. red_to_blue   — "place the red box in the blue zone"
  4. blue_to_red   — "place the blue box in the red zone"
  5. both_correct  — "place both boxes in their zones"
  6. both_swapped  — "place red box in blue zone and blue box in red zone"

Each type has multiple natural language phrasings so the model learns
to handle varied wording, not just one fixed template.

Usage:
    # Balanced: 100 episodes per type (600 total)
    python generate_demo_data.py

    # Custom count per type
    python generate_demo_data.py --per-type 200

    # Append to existing human-recorded data
    python generate_demo_data.py --mode append --out language_demo_data.jsonl

    # Check what you already have before generating more
    python generate_demo_data.py --stats-only
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

from grid_env_simple import GridEnv


# ─────────────────────────────────────────────────────────────────────
# Instruction catalogue
# Each entry: (type_key, [list of natural language phrasings])
# The oracle uses type_key to determine which box goes where.
# Multiple phrasings per type teach the model language variation.
# ─────────────────────────────────────────────────────────────────────

INSTRUCTION_CATALOGUE = {

    "red_to_red": [
        "place the red box in the red zone",
        "put the red box on the red target",
        "put the red block on the red target",
        "move the red box to the red area",
        "take the red box to the red zone",
        "bring the red box to the red target",
        "drop the red box in the red zone",
    ],

    "blue_to_blue": [
        "place the blue box in the blue zone",
        "put the blue box on the blue target",
        "put the blue block on the blue target",
        "move the blue box to the blue area",
        "take the blue box to the blue zone",
        "bring the blue box to the blue target",
        "drop the blue box in the blue zone",
    ],

    "red_to_blue": [
        "place the red box in the blue zone",
        "put the red box on the blue target",
        "put the red block on the blue target",
        "move the red box to the blue area",
        "take the red box to the blue zone",
        "bring the red box to the blue target",
    ],

    "blue_to_red": [
        "place the blue box in the red zone",
        "put the blue box on the red target",
        "put the blue block on the red target",
        "move the blue box to the red area",
        "take the blue box to the red zone",
        "bring the blue box to the red target",
    ],

    "both_correct": [
        "place both boxes in their zones",
        "put both boxes on their matching targets",
        "move both boxes to their correct zones",
        "place the red box in the red zone and the blue box in the blue zone",
        "put each box on its matching target",
        "sort both boxes into their correct areas",
        "place every box in its correct zone",
    ],

    "both_swapped": [
        "place the red box in the blue zone and the blue box in the red zone",
        "put the red box on the blue target and the blue box on the red target",
        "swap the boxes: red to blue zone and blue to red zone",
        "move the red box to the blue area and the blue box to the red area",
        "place both boxes in the opposite zones",
        "switch the boxes into each other's zones",
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Goal specification per type
#
# A "goal" is a list of (box_colour, target_colour) pairs the oracle
# must complete. The oracle works through them in order.
# ─────────────────────────────────────────────────────────────────────

TYPE_GOALS = {
    "red_to_red":   [("red",  "red")],
    "blue_to_blue": [("blue", "blue")],
    "red_to_blue":  [("red",  "blue")],
    "blue_to_red":  [("blue", "red")],
    "both_correct": [("red",  "red"),  ("blue", "blue")],
    "both_swapped": [("red",  "blue"), ("blue", "red")],
}


# ─────────────────────────────────────────────────────────────────────
# Oracle policy — goal-aware
# ─────────────────────────────────────────────────────────────────────

def _dir_to_action(dr: int, dc: int) -> int:
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 1
    return 2 if dc < 0 else 3


def _box_pos(state: dict, colour: str) -> tuple:
    return tuple(state["red_box_pos"] if colour == "red" else state["blue_box_pos"])


def _tgt_pos(state: dict, colour: str) -> tuple:
    return tuple(state["red_tgt_pos"] if colour == "red" else state["blue_tgt_pos"])


def _goal_satisfied(state: dict, box_colour: str, tgt_colour: str) -> bool:
    """True when the named box is sitting on the named target."""
    return _box_pos(state, box_colour) == _tgt_pos(state, tgt_colour)


def oracle_action_for_goals(state: dict, goals: list) -> int:
    """
    goals: list of (box_colour, tgt_colour) pairs still to complete.
    Finds the first unsatisfied goal and works toward it.
    Returns action int 0-5.
    """
    agent = tuple(state["agent_pos"])
    held  = state["held_object"]

    # Find the first goal not yet satisfied
    pending = [(b, t) for b, t in goals if not _goal_satisfied(state, b, t)]

    if not pending:
        return 5  # all done — idle place (won't matter, episode ends)

    box_colour, tgt_colour = pending[0]

    if held is None:
        # Walk to the next box and pick it up
        box = _box_pos(state, box_colour)
        if agent == box:
            return 4  # pick
        dr = box[0] - agent[0]
        dc = box[1] - agent[1]
        return _dir_to_action(dr, dc)

    elif held == box_colour:
        # Carrying the right box — walk to the target and place
        tgt = _tgt_pos(state, tgt_colour)
        if agent == tgt:
            return 5  # place
        dr = tgt[0] - agent[0]
        dc = tgt[1] - agent[1]
        return _dir_to_action(dr, dc)

    else:
        # Carrying the WRONG box — put it down immediately, then continue
        return 5  # place wherever we are, then re-pick correct box


# ─────────────────────────────────────────────────────────────────────
# Custom success check per type (does not rely on info["red_done"])
# ─────────────────────────────────────────────────────────────────────

def episode_succeeded(state: dict, type_key: str) -> bool:
    goals = TYPE_GOALS[type_key]
    return all(_goal_satisfied(state, b, t) for b, t in goals)


# ─────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────

def _ser(state: dict) -> dict:
    return {k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in state.items()}


def run_oracle_episode(env: GridEnv, type_key: str, instruction: str,
                       seed: int, max_steps: int = 120) -> list:
    """
    Run one episode with the goal-aware oracle.
    Returns list of transition dicts, or [] on failure.
    """
    goals = TYPE_GOALS[type_key]
    state = env.reset(seed=seed)
    buf   = []

    for step in range(max_steps):
        action = oracle_action_for_goals(state, goals)
        next_state, _, env_done, _ = env.step(action)

        buf.append({
            "instruction": instruction if step == 0 else "",
            "state":       _ser(state),
            "action":      action,
            "next_state":  _ser(next_state),
        })
        state = next_state

        if episode_succeeded(state, type_key):
            return buf   # success
        if env_done:
            break        # timed out

    return []   # failure — discard


# ─────────────────────────────────────────────────────────────────────
# Stats helper
# ─────────────────────────────────────────────────────────────────────

def print_stats(jsonl_path: str):
    if not os.path.exists(jsonl_path):
        print(f"[stats] {jsonl_path} does not exist yet.")
        return
    with open(jsonl_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    episodes = [l for l in lines if l.get("instruction")]
    counts   = Counter(l["instruction"] for l in episodes)
    print(f"\n[stats] {jsonl_path}")
    print(f"  Total transitions : {len(lines)}")
    print(f"  Total episodes    : {len(episodes)}")
    print(f"  Unique instructions: {len(counts)}")
    print()
    # Group by type_key
    type_counts = defaultdict(int)
    for instr, n in counts.items():
        matched = False
        for tk, phrasings in INSTRUCTION_CATALOGUE.items():
            if instr in phrasings:
                type_counts[tk] += n
                matched = True
                break
        if not matched:
            type_counts["(human/other)"] += n
    for tk in list(TYPE_GOALS.keys()) + ["(human/other)"]:
        n = type_counts.get(tk, 0)
        bar = "█" * (n // 2)
        print(f"  {tk:<18}  {n:4d}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate oracle demonstration data for all instruction types")
    p.add_argument("--per-type",    type=int, default=100,
                   help="Successful episodes to generate per instruction type")
    p.add_argument("--out",         type=str, default="language_demo_data.jsonl",
                   help="Output .jsonl path")
    p.add_argument("--mode",        choices=["overwrite", "append"], default="overwrite",
                   help="overwrite: replace existing file  |  append: add to it")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--max-steps",   type=int, default=120,
                   help="Max steps per episode before discarding")
    p.add_argument("--stats-only",  action="store_true",
                   help="Just print dataset stats and exit")
    p.add_argument("--types",       nargs="+",
                   choices=list(TYPE_GOALS.keys()),
                   default=list(TYPE_GOALS.keys()),
                   help="Which instruction types to generate (default: all)")
    args = p.parse_args()

    if args.stats_only:
        print_stats(args.out)
        return

    print_stats(args.out)

    env = GridEnv()
    rng = random.Random(args.seed)
    all_demos = []

    total_types = len(args.types)
    total_target = args.per_type * total_types

    print(f"Generating {args.per_type} episodes × {total_types} types "
          f"= {total_target} episodes target\n")

    for type_key in args.types:
        phrasings   = INSTRUCTION_CATALOGUE[type_key]
        done_count  = 0
        tried       = 0
        type_demos  = []

        while done_count < args.per_type:
            seed_ep = rng.randint(0, 999999)
            instr   = rng.choice(phrasings)
            ep      = run_oracle_episode(env, type_key, instr,
                                         seed=seed_ep, max_steps=args.max_steps)
            tried  += 1
            if ep:
                type_demos.extend(ep)
                done_count += 1

        all_demos.extend(type_demos)
        n_trans = len(type_demos)
        avg     = n_trans / done_count if done_count else 0
        sr      = done_count / tried * 100
        print(f"  {type_key:<18}  {done_count:3d} episodes  "
              f"{n_trans:5d} transitions  avg {avg:.1f} steps  "
              f"success rate {sr:.0f}%")

    # Write file
    mode = "a" if args.mode == "append" else "w"
    with open(args.out, mode) as f:
        for d in all_demos:
            f.write(json.dumps(d) + "\n")

    n_ep = sum(1 for d in all_demos if d.get("instruction"))
    print(f"\nWrote {len(all_demos)} transitions ({n_ep} episodes) "
          f"→ {args.out}  [mode={args.mode}]")

    print_stats(args.out)


if __name__ == "__main__":
    main()
