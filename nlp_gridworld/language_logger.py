"""
language_logger.py
------------------
Human-controlled agent that records demonstrations to language_demo_data.jsonl.

FIX (v2): Single master event loop owns ALL pygame.event.get() calls.
The renderer's draw() is a pure paint call — it never touches events.
This eliminates the double-drain freeze that caused the black screen.

Controls:
  W / UP    = move up     (action 0)
  S / DOWN  = move down   (action 1)
  A / LEFT  = move left   (action 2)
  D / RIGHT = move right  (action 3)
  P         = pick        (action 4)
  L         = place       (action 5)
  R         = reset episode (discards current, not saved)
  Q / ESC   = quit and save all completed episodes

Usage:
    python language_logger.py
"""

import json
import os
import sys

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame is required. Install with: pip install pygame")
    sys.exit(1)

from grid_env_simple import GridEnv
from renderer_simple import Renderer, W_WIN, H_WIN

DATA_PATH = "language_demo_data.jsonl"

KEY_ACTION_MAP = {
    pygame.K_w:     0,
    pygame.K_UP:    0,
    pygame.K_s:     1,
    pygame.K_DOWN:  1,
    pygame.K_a:     2,
    pygame.K_LEFT:  2,
    pygame.K_d:     3,
    pygame.K_RIGHT: 3,
    pygame.K_p:     4,
    pygame.K_l:     5,
}

SAMPLE_INSTRUCTIONS = [
    "place the red box in the red zone",
    "place the blue box in the blue zone",
    "place both boxes in their zones",
    "put the red block on the red target",
    "put the blue block on the blue target",
]


def prompt_instruction_in_window(screen, clock, font, small):
    """
    Show an overlay input box inside the pygame window.
    User types and presses Enter. Returns string or None on quit.
    """
    text      = ""
    BOX_CLR   = (255, 255, 255)
    TXT_CLR   = (20,  20,  20)
    HINT_CLR  = (160, 160, 160)
    LABEL_CLR = (240, 240, 240)

    overlay = pygame.Surface((W_WIN, H_WIN), pygame.SRCALPHA)
    overlay.fill((20, 20, 40, 210))

    while True:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_RETURN:
                    s = text.strip()
                    return s if s else SAMPLE_INSTRUCTIONS[0]
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                else:
                    if event.unicode and event.unicode.isprintable():
                        text += event.unicode

        screen.blit(overlay, (0, 0))

        lbl = font.render("Enter instruction (press Enter to confirm):", True, LABEL_CLR)
        screen.blit(lbl, (40, H_WIN // 2 - 88))

        box_rect = pygame.Rect(40, H_WIN // 2 - 60, W_WIN - 80, 44)
        pygame.draw.rect(screen, BOX_CLR, box_rect, border_radius=6)
        pygame.draw.rect(screen, (100, 100, 220), box_rect, 2, border_radius=6)

        display = text if text else SAMPLE_INSTRUCTIONS[0]
        clr     = TXT_CLR if text else HINT_CLR
        ts      = font.render(display, True, clr)
        screen.blit(ts, (box_rect.x + 8, box_rect.y + 10))

        if (pygame.time.get_ticks() // 500) % 2 == 0:
            cx = box_rect.x + 8 + ts.get_width() + 2
            pygame.draw.line(screen, TXT_CLR, (cx, box_rect.y + 8), (cx, box_rect.y + 36), 2)

        hint_y = H_WIN // 2 + 6
        screen.blit(small.render("Examples:", True, LABEL_CLR), (40, hint_y))
        for i, s in enumerate(SAMPLE_INSTRUCTIONS[:4]):
            screen.blit(small.render(f"  {i+1}. {s}", True, HINT_CLR), (40, hint_y + 18 + i * 17))

        ctrl = small.render("R=reset   Q=quit & save   P=pick   L=place   WASD/arrows=move", True, HINT_CLR)
        screen.blit(ctrl, (40, hint_y + 18 + 4 * 17 + 8))

        pygame.display.flip()


# ─────────────────────────────────────────────────────────────────────
# Instruction-aware success detection
# ─────────────────────────────────────────────────────────────────────

def _box_on_target(state: dict, box_colour: str, tgt_colour: str) -> bool:
    """Check if a specific box is sitting on a specific target zone."""
    box_key = "red_box_pos"  if box_colour == "red"  else "blue_box_pos"
    tgt_key = "red_tgt_pos"  if tgt_colour == "red"  else "blue_tgt_pos"
    return list(state[box_key]) == list(state[tgt_key])


def instruction_goal_met(instruction: str, info: dict,
                         state: dict = None) -> bool:
    """
    Returns True when the instruction's specific goal has been satisfied.

    Handles all 6 instruction types including cross-placement cases:
      red_to_red    "place the red box in the red zone"
      blue_to_blue  "place the blue box in the blue zone"
      red_to_blue   "place the red box in the blue zone"
      blue_to_red   "place the blue box in the red zone"
      both_correct  "place both boxes in their zones"
      both_swapped  "place red box in blue zone and blue box in red zone"

    For cross-placement cases (red→blue, blue→red) info["red_done"] is
    NOT enough — we need to check actual positions from the state dict.
    Falls back gracefully when state is not provided.
    """
    instr = instruction.lower()

    # ── Detect "both swapped" first (most specific pattern) ──────────
    # Indicator: instruction mentions both colours AND ("swap", "opposite",
    # "switch", or "blue zone" appears before/after "red zone" suggesting
    # cross-placement, or both red-to-blue and blue-to-red phrases present)
    both_swapped = (
        "swap" in instr
        or "opposite" in instr
        or "switch" in instr
        or ("red" in instr and "blue zone" in instr
            and "blue" in instr and "red zone" in instr)
    )
    if both_swapped:
        if state:
            return (_box_on_target(state, "red",  "blue")
                    and _box_on_target(state, "blue", "red"))
        # Fallback without state: both boxes must have moved (no env signal)
        return bool(info.get("red_done")) and bool(info.get("blue_done"))

    # ── Detect "both correct" ─────────────────────────────────────────
    has_both = "both" in instr or "every" in instr or "each" in instr
    if has_both:
        return bool(info.get("red_done")) and bool(info.get("blue_done"))

    # ── Single-box cross-placement: "red box in the blue zone" ────────
    # Key signal: the box colour and zone colour are DIFFERENT
    red_box_mentioned  = "red box"   in instr or "red block" in instr
    blue_box_mentioned = "blue box"  in instr or "blue block" in instr
    red_zone_mentioned  = "red zone"  in instr or "red target" in instr or "red area" in instr
    blue_zone_mentioned = "blue zone" in instr or "blue target" in instr or "blue area" in instr

    if red_box_mentioned and blue_zone_mentioned and not red_zone_mentioned:
        # red box → blue zone
        if state:
            return _box_on_target(state, "red", "blue")
        return bool(info.get("blue_done"))   # imperfect but only fallback

    if blue_box_mentioned and red_zone_mentioned and not blue_zone_mentioned:
        # blue box → red zone
        if state:
            return _box_on_target(state, "blue", "red")
        return bool(info.get("red_done"))

    # ── Standard single-box cases ─────────────────────────────────────
    has_red  = "red"  in instr
    has_blue = "blue" in instr

    if has_red and not has_blue:
        return bool(info.get("red_done"))
    if has_blue and not has_red:
        return bool(info.get("blue_done"))

    # Both colours mentioned without clear cross-placement → require both
    return bool(info.get("red_done")) and bool(info.get("blue_done"))


def run_demo_session():
    pygame.init()
    screen = pygame.display.set_mode((W_WIN, H_WIN))
    pygame.display.set_caption("NLP GridWorld — Demo Recorder")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 15)
    small  = pygame.font.SysFont("monospace", 12)

    # Build renderer manually so it shares our existing window
    renderer         = Renderer.__new__(Renderer)
    renderer.screen  = screen
    renderer.clock   = clock
    renderer.fps     = 10
    renderer.font    = font
    renderer.small   = small

    env   = GridEnv()
    demos = []

    print("[logger] Session started — close window or press Q to quit & save.")

    while True:
        instruction = prompt_instruction_in_window(screen, clock, font, small)
        if instruction is None:
            break

        print(f"[logger] Instruction: '{instruction}'")
        state  = env.reset()
        done   = False
        info   = {}
        ep_buf = []

        while not done:
            action     = None
            quit_req   = False
            reset_req  = False

            # ── ONE event poll for the whole frame ────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_req = True
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        quit_req = True
                        break
                    if event.key == pygame.K_r:
                        reset_req = True
                        break
                    if event.key in KEY_ACTION_MAP:
                        action = KEY_ACTION_MAP[event.key]

            if quit_req:
                ep_buf      = []
                instruction = None
                done        = True
                break

            if reset_req:
                print("[logger] Episode reset — discarded.")
                ep_buf = []
                state  = env.reset()
                info   = {}
                renderer.draw(state, instruction, info)
                continue

            # ── Pure draw — no event polling inside ───────────────────
            renderer.draw(state, instruction, info)

            if action is not None:
                next_state, _reward, env_done, info = env.step(action)
                ep_buf.append({
                    "instruction": instruction if len(ep_buf) == 0 else "",
                    "state":       _ser(state),
                    "action":      action,
                    "next_state":  _ser(next_state),
                })
                state = next_state

                # Check instruction-specific success BEFORE env_done.
                # Pass next_state so cross-placement cases can check
                # actual box positions (info alone is not enough for those).
                goal_met = instruction_goal_met(instruction, info, next_state)

                if goal_met:
                    n = sum(1 for d in demos if d["instruction"]) + 1
                    print(f"[logger] SUCCESS — episode {n} saved "
                          f"({len(ep_buf)} steps)  "
                          f"[red={info.get('red_done')} blue={info.get('blue_done')}]")
                    demos.extend(ep_buf)
                    ep_buf = []
                    done   = True   # exit the episode loop

                elif env_done:
                    # Env timed out without reaching the goal
                    print("[logger] Time limit reached — episode discarded.")
                    ep_buf = []
                    done   = True

            clock.tick(30)

        if instruction is None:
            break

    pygame.quit()
    _save(demos)


def _ser(state):
    return {k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in state.items()}


def _save(demos):
    if not demos:
        print("[logger] No completed episodes to save.")
        return
    mode = "a" if os.path.exists(DATA_PATH) else "w"
    with open(DATA_PATH, mode) as f:
        for d in demos:
            f.write(json.dumps(d) + "\n")
    n_ep = sum(1 for d in demos if d["instruction"])
    print(f"[logger] Saved {len(demos)} transitions ({n_ep} episodes) → {DATA_PATH}")


if __name__ == "__main__":
    run_demo_session()
