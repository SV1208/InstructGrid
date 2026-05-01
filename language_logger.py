"""
language_logger.py
------------------
Human demonstration recorder for InstructGrid.

Improvements:
  - Clipboard paste (Ctrl+V)
  - Instruction history (up/down arrows in prompt to cycle)
  - Number keys 1-4 to fill a preset instruction instantly
  - 4 instruction types only (no "both" variants)
  - Saves to language_demo_data_human.jsonl (separate from oracle)
  - Instruction-aware success detection for all 4 types
  - Better renderer with held-object colour on agent circle

Controls (in-game):
  W/A/S/D or arrow keys  — move
  P                      — pick
  L                      — place
  R                      — reset episode (discard, not saved)
  Q / ESC                — quit and save all completed episodes
"""

import json
import os
import sys

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False
    print("pygame required: pip install pygame")
    sys.exit(1)

from grid_env_simple import GridEnv
from renderer_simple import Renderer, W_WIN, H_WIN

HUMAN_FILE = "language_demo_data_human.jsonl"

KEY_ACTION = {
    pygame.K_w: 0, pygame.K_UP:    0,
    pygame.K_s: 1, pygame.K_DOWN:  1,
    pygame.K_a: 2, pygame.K_LEFT:  2,
    pygame.K_d: 3, pygame.K_RIGHT: 3,
    pygame.K_p: 4,
    pygame.K_l: 5,
}

PRESET_INSTRUCTIONS = [
    "place the red box in the red zone",    # key 1
    "place the blue box in the blue zone",  # key 2
    "place the red box in the blue zone",   # key 3
    "place the blue box in the red zone",   # key 4
]

EXAMPLE_PHRASINGS = [
    "place the red box in the red zone",
    "put the red block on the red target",
    "place the blue box in the blue zone",
    "put the blue block on the blue target",
    "place the red box in the blue zone",
    "put the red block on the blue target",
    "place the blue box in the red zone",
    "put the blue block on the red target",
]

# Colours for the prompt overlay
DARK   = ( 12,  18,  34)
CARD   = ( 20,  30,  54)
BORDER = ( 60,  90, 160)
WHITE  = (230, 235, 245)
MUTED  = (100, 115, 150)
CYAN   = ( 80, 200, 255)
AMBER  = (255, 200,  60)
GREEN  = ( 80, 220, 140)
RED_C  = (220,  80,  80)


# =====================================================================
# Goal checking
# =====================================================================

def _goal_met(instruction: str, state: dict) -> bool:
    """
    True only when the correct box is physically placed on the correct
    target (held_object must be None for that box colour).
    Without this check, the episode ends the moment the agent steps onto
    the target while carrying the box — before pressing L to place.
    """
    instr = instruction.lower()
    rp = list(state["red_box_pos"])
    bp = list(state["blue_box_pos"])
    rt = list(state["red_tgt_pos"])
    bt = list(state["blue_tgt_pos"])
    held = state.get("held_object")

    red_box  = "red box"  in instr or "red block"  in instr
    blue_box = "blue box" in instr or "blue block" in instr
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
# Instruction prompt overlay
# =====================================================================

def prompt_instruction(screen, clock, font, bold, small,
                        history: list) -> str | None:
    """
    Returns instruction string, or None if user wants to quit.
    Supports:
      - Typing characters
      - Backspace / Ctrl+Backspace (delete word)
      - Ctrl+V paste
      - Up/Down arrows to cycle history
      - Number keys 1–4 for presets
      - Enter to confirm  (empty → use first preset)
      - ESC to quit session
    """
    # Init clipboard
    try:
        if not pygame.scrap.get_init():
            pygame.scrap.init()
        clipboard_ok = True
    except Exception:
        clipboard_ok = False

    text      = ""
    hist_idx  = -1            # -1 = not browsing history
    overlay   = pygame.Surface((W_WIN, H_WIN), pygame.SRCALPHA)
    overlay.fill((10, 15, 30, 215))

    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()

                if event.key == pygame.K_ESCAPE:
                    return None

                elif event.key == pygame.K_RETURN:
                    return text.strip() if text.strip() else PRESET_INSTRUCTIONS[0]

                elif event.key == pygame.K_BACKSPACE:
                    if mods & pygame.KMOD_CTRL:
                        # Delete last word
                        stripped = text.rstrip()
                        last_sp  = stripped.rfind(" ")
                        text = stripped[:last_sp+1] if last_sp >= 0 else ""
                    else:
                        text = text[:-1]
                    hist_idx = -1

                elif event.key == pygame.K_v and (mods & pygame.KMOD_CTRL):
                    if clipboard_ok:
                        try:
                            raw = pygame.scrap.get(pygame.SCRAP_TEXT)
                            if raw:
                                # Windows returns null-terminated bytes
                                clip = raw.rstrip(b"\x00").decode("utf-8", errors="ignore")
                                text += clip.replace("\n", " ").replace("\r", "").strip()
                        except Exception:
                            pass

                elif event.key == pygame.K_UP:
                    if history:
                        hist_idx = min(hist_idx + 1, len(history) - 1)
                        text     = history[-(hist_idx + 1)]

                elif event.key == pygame.K_DOWN:
                    if hist_idx > 0:
                        hist_idx -= 1
                        text      = history[-(hist_idx + 1)]
                    else:
                        hist_idx = -1
                        text     = ""

                elif event.key in (pygame.K_1, pygame.K_2,
                                   pygame.K_3, pygame.K_4):
                    idx  = event.key - pygame.K_1
                    text = PRESET_INSTRUCTIONS[idx]
                    hist_idx = -1

                else:
                    if event.unicode and event.unicode.isprintable():
                        text    += event.unicode
                        hist_idx = -1

        # Draw overlay
        screen.blit(overlay, (0, 0))

        # Card background
        card_rect = pygame.Rect(30, H_WIN//2 - 160, W_WIN - 60, 310)
        pygame.draw.rect(screen, CARD, card_rect, border_radius=10)
        pygame.draw.rect(screen, BORDER, card_rect, 2, border_radius=10)

        # Title
        title = bold.render("Enter instruction", True, WHITE)
        screen.blit(title, (card_rect.x + 18, card_rect.y + 14))

        # Preset pills row
        pill_y = card_rect.y + 44
        for i, (key, instr) in enumerate(zip("1234", PRESET_INSTRUCTIONS)):
            short = instr.replace("place the ", "").replace(" box in the ", "→").replace(" zone","")
            px    = card_rect.x + 18 + i * ((card_rect.w - 36) // 4)
            pw    = (card_rect.w - 36) // 4 - 6
            pygame.draw.rect(screen, BORDER,
                             pygame.Rect(px, pill_y, pw, 26), border_radius=5)
            ks = small.render(f"[{key}] {short}", True, CYAN)
            screen.blit(ks, (px + 6, pill_y + 5))

        # Input box
        box_y = pill_y + 38
        box_rect = pygame.Rect(card_rect.x + 18, box_y,
                               card_rect.w - 36, 38)
        pygame.draw.rect(screen, DARK, box_rect, border_radius=6)
        pygame.draw.rect(screen, CYAN, box_rect, 1, border_radius=6)

        display = text if text else PRESET_INSTRUCTIONS[0]
        clr     = WHITE if text else MUTED
        ts      = font.render(display[:70], True, clr)
        screen.blit(ts, (box_rect.x + 8, box_rect.y + 9))

        # Blinking cursor
        if (pygame.time.get_ticks() // 500) % 2 == 0 and text:
            cx = box_rect.x + 8 + ts.get_width() + 2
            pygame.draw.line(screen, WHITE,
                             (cx, box_rect.y + 6), (cx, box_rect.y + 30), 2)

        # Tips row
        tips_y = box_y + 50
        tips = [
            ("Enter", "confirm"),
            ("Ctrl+V", "paste"),
            ("↑↓", "history"),
            ("ESC", "quit"),
        ]
        tx = card_rect.x + 18
        for key_txt, desc_txt in tips:
            ks = small.render(f"[{key_txt}]", True, AMBER)
            ds = small.render(f" {desc_txt}  ", True, MUTED)
            screen.blit(ks, (tx, tips_y))
            screen.blit(ds, (tx + ks.get_width(), tips_y))
            tx += ks.get_width() + ds.get_width() + 4

        # Paste note if clipboard unavailable
        if not clipboard_ok:
            ns = small.render("(clipboard unavailable on this system)", True, MUTED)
            screen.blit(ns, (card_rect.x + 18, tips_y + 18))

        # History list
        if history:
            hy = tips_y + 40
            screen.blit(small.render("Recent:", True, MUTED),
                        (card_rect.x + 18, hy))
            for j, h in enumerate(reversed(history[-4:])):
                clr = CYAN if j == hist_idx else MUTED
                hs  = small.render(f"  {h[:60]}", True, clr)
                screen.blit(hs, (card_rect.x + 18, hy + 14 + j * 15))

        pygame.display.flip()


# =====================================================================
# Main recording session
# =====================================================================

def _ser(state: dict) -> dict:
    return {k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in state.items()}


def _save(demos: list):
    if not demos:
        print("[logger] No completed episodes — nothing saved.")
        return
    mode = "a" if os.path.exists(HUMAN_FILE) else "w"
    with open(HUMAN_FILE, mode) as f:
        for d in demos:
            f.write(json.dumps(d) + "\n")
    n_ep = sum(1 for d in demos if d.get("instruction"))
    print(f"[logger] Saved {len(demos)} transitions "
          f"({n_ep} episodes) → {HUMAN_FILE}")


def run_session():
    pygame.init()
    screen = pygame.display.set_mode((W_WIN, H_WIN))
    pygame.display.set_caption("InstructGrid — Demo Recorder")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("consolas", 14)
    bold  = pygame.font.SysFont("consolas", 14, bold=True)
    small = pygame.font.SysFont("consolas", 11)

    # Attach renderer to existing window (no second pygame.init)
    renderer        = Renderer.__new__(Renderer)
    renderer.screen = screen
    renderer.clock  = clock
    renderer.fps    = 8
    renderer.font   = font
    renderer.bold   = bold
    renderer.small  = small

    env     = GridEnv()
    demos   = []
    history = []

    print("[logger] Session started. Q to quit and save.")

    while True:
        instruction = prompt_instruction(screen, clock, font, bold, small, history)
        if instruction is None:
            break

        # Track history (most recent unique at end)
        if instruction in history:
            history.remove(instruction)
        history.append(instruction)

        print(f"[logger] Instruction: '{instruction}'")
        state    = env.reset()
        done     = False
        info     = {}
        ep_buf   = []
        ep_count = sum(1 for d in demos if d.get("instruction"))

        while not done:
            action    = None
            quit_req  = False
            reset_req = False

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
                    if event.key in KEY_ACTION:
                        action = KEY_ACTION[event.key]

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

            renderer.draw(state, instruction, info,
                          model_name=f"Episode {ep_count+1}")

            if action is not None:
                next_state, _, env_done, info = env.step(action)

                ep_buf.append({
                    "instruction": instruction if len(ep_buf) == 0 else "",
                    "state":       _ser(state),
                    "action":      action,
                    "next_state":  _ser(next_state),
                })
                state = next_state

                if _goal_met(instruction, state):
                    n = ep_count + 1
                    print(f"[logger] SUCCESS — episode {n} "
                          f"({len(ep_buf)} steps)")
                    demos.extend(ep_buf)
                    ep_buf = []
                    done   = True
                elif env_done:
                    print("[logger] Time limit — discarded.")
                    ep_buf = []
                    done   = True

            clock.tick(30)

        if instruction is None:
            break

    pygame.quit()
    _save(demos)


if __name__ == "__main__":
    run_session()
