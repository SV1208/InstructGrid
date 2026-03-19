"""
generate_demo_data.py
----------------------
Generates synthetic demonstration data using a scripted (near-optimal) oracle
policy. Use this if you do not yet have human demonstrations, or to quickly
test the full pipeline end-to-end without running language_logger.py.

The oracle policy uses simple rule-based planning:
  1. If not holding a box, navigate to the closest unplaced box and pick it.
  2. If holding a box, navigate to its target and place it.
  3. Repeat until both boxes are placed.

The script generates N episodes and saves them to language_demo_data.jsonl.

Usage:
    python generate_demo_data.py
    python generate_demo_data.py --episodes 200 --out language_demo_data.jsonl
"""

import argparse
import json
import random

from grid_env_simple import GridEnv, ACTION_NAMES

INSTRUCTIONS = [
    "place the red box in the red zone",
    "place the blue box in the blue zone",
    "place both boxes in their zones",
    "put the red block on the red target",
    "put the blue block on the blue target",
    "move the red box to the red area",
    "move the blue box to the blue area",
]


# ─────────────────────────────────────────────────────────────────────
# Oracle policy
# ─────────────────────────────────────────────────────────────────────

def _dir_to_action(dr, dc):
    """Convert a (row, col) delta to the first move action."""
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 1
    return 2 if dc < 0 else 3


def oracle_action(state: dict) -> int:
    """Scripted policy: move toward the next objective, pick/place when adjacent."""
    agent = tuple(state["agent_pos"])
    held  = state["held_object"]

    if held is None:
        # Decide which box to go for
        red_placed  = state["red_box_pos"]  == state["red_tgt_pos"]
        blue_placed = state["blue_box_pos"] == state["blue_tgt_pos"]

        if not red_placed:
            target = tuple(state["red_box_pos"])
        elif not blue_placed:
            target = tuple(state["blue_box_pos"])
        else:
            return 5  # both placed, just place to end

        if agent == target:
            return 4  # pick
        dr = target[0] - agent[0]
        dc = target[1] - agent[1]
        return _dir_to_action(dr, dc)

    else:
        # Navigate to the correct target zone
        tgt = tuple(state["red_tgt_pos"] if held == "red" else state["blue_tgt_pos"])
        if agent == tgt:
            return 5  # place
        dr = tgt[0] - agent[0]
        dc = tgt[1] - agent[1]
        return _dir_to_action(dr, dc)


# ─────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────

def run_oracle_episode(env: GridEnv, instruction: str, seed: int,
                       max_steps: int = 100) -> list:
    """
    Returns a list of transition dicts (may be empty if episode failed).
    """
    state = env.reset(seed=seed)
    done  = False
    buf   = []

    for step in range(max_steps):
        action     = oracle_action(state)
        next_state, _, done, info = env.step(action)

        # Instruction only in first step (matches original format)
        buf.append({
            "instruction": instruction if step == 0 else "",
            "state":       {k: (list(v) if isinstance(v, (list, tuple)) else v)
                            for k, v in state.items()},
            "action":      action,
            "next_state":  {k: (list(v) if isinstance(v, (list, tuple)) else v)
                            for k, v in next_state.items()},
        })
        state = next_state
        if done:
            break

    # Only keep successful episodes
    if info.get("red_done") and info.get("blue_done"):
        return buf
    return []


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500,
                   help="Number of successful episodes to generate")
    p.add_argument("--out",      type=str, default="language_demo_data.jsonl")
    p.add_argument("--seed",     type=int, default=0)
    args = p.parse_args()

    env    = GridEnv()
    rng    = random.Random(args.seed)
    demos  = []
    tried  = 0

    print(f"Generating {args.episodes} successful episodes ...")
    while len(demos) < args.episodes:
        seed_ep  = rng.randint(0, 99999)
        instr    = rng.choice(INSTRUCTIONS)
        episode  = run_oracle_episode(env, instr, seed=seed_ep)
        tried   += 1
        if episode:
            demos.extend(episode)
            if len(demos) % 500 == 0 or len(demos) // len(episode) % 50 == 0:
                n_ep = sum(1 for d in demos if d["instruction"])
                print(f"  {n_ep} episodes / {len(demos)} transitions")

    with open(args.out, "w") as f:
        for d in demos:
            f.write(json.dumps(d) + "\n")

    n_ep = sum(1 for d in demos if d["instruction"])
    print(f"\nSaved {len(demos)} transitions ({n_ep} episodes) → {args.out}")
    print(f"Oracle success rate: {n_ep/tried*100:.1f}% ({n_ep}/{tried} attempts)")


if __name__ == "__main__":
    main()
