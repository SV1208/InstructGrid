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
                next_state, _reward, done, info = env.step(action)
                ep_buf.append({
                    "instruction": instruction if len(ep_buf) == 0 else "",
                    "state":       _ser(state),
                    "action":      action,
                    "next_state":  _ser(next_state),
                })
                state = next_state

                if done:
                    if info.get("red_done") and info.get("blue_done"):
                        n = sum(1 for d in demos if d["instruction"]) + 1
                        print(f"[logger] SUCCESS — episode {n} saved "
                              f"({len(ep_buf)} steps)")
                        demos.extend(ep_buf)
                    else:
                        print("[logger] Time limit reached — episode discarded.")
                    ep_buf = []

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
