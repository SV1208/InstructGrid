"""
renderer_simple.py
------------------
Pygame-based renderer for the GridEnv.

IMPORTANT: This class is a PURE DRAWING utility.
It does NOT poll pygame events — the caller owns the event loop.
This prevents double-draining the event queue (which causes freezes).

Usage:
    renderer = Renderer()
    renderer.draw(state, instruction="place the red box in the blue zone")
    # You handle pygame.event.get() yourself in your game loop.

The old renderer.render() is kept as an alias for backward compatibility
with evaluate.py, but it only handles events when called standalone
(i.e. when no external event loop is running).
"""

import sys

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[renderer] pygame not installed — visual rendering disabled.")

CELL   = 70          # pixels per cell
MARGIN = 10          # border margin
PANEL  = 110         # bottom panel height for text
W_WIN  = 8 * CELL + 2 * MARGIN
H_WIN  = 8 * CELL + 2 * MARGIN + PANEL

# Colours
BG      = (245, 245, 240)
GRID_C  = (200, 200, 190)
AGENT_C = (50,  50,  200)
RED_BOX = (220, 60,  60)
BLU_BOX = (60,  100, 220)
RED_TGT = (255, 160, 160)
BLU_TGT = (160, 190, 255)
TEXT_C  = (30,  30,  30)
HELD_C  = (255, 200, 0)
PANEL_C = (230, 230, 225)


class Renderer:
    def __init__(self, title="NLP GridWorld", fps=10):
        if not PYGAME_AVAILABLE:
            return
        pygame.init()
        self.screen = pygame.display.set_mode((W_WIN, H_WIN))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.fps   = fps
        self.font  = pygame.font.SysFont("monospace", 15)
        self.small = pygame.font.SysFont("monospace", 12)

    def draw(self, state: dict, instruction: str = "", info: dict = None):
        """
        Pure draw call — NO event polling.
        Call pygame.event.get() in your own loop before calling this.
        """
        if not PYGAME_AVAILABLE:
            return
        self.screen.fill(BG)
        self._draw_grid()
        self._draw_targets(state)
        self._draw_objects(state)
        self._draw_agent(state)
        self._draw_panel(state, instruction, info)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def render(self, state: dict, instruction: str = "", info: dict = None):
        """
        Backward-compatible wrapper used by evaluate.py.
        Handles its own events — only use this when YOU are not
        running an external event loop (i.e. in evaluate.py only).
        Returns False when the user requests quit.
        """
        if not PYGAME_AVAILABLE:
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                return False
        self.draw(state, instruction, info)
        return True

    def close(self):
        if PYGAME_AVAILABLE:
            pygame.quit()

    # ----------------------------------------------------------------
    # Drawing helpers
    # ----------------------------------------------------------------

    def _cell_rect(self, r, c):
        x = MARGIN + c * CELL
        y = MARGIN + r * CELL
        return pygame.Rect(x, y, CELL, CELL)

    def _draw_grid(self):
        for r in range(8):
            for c in range(8):
                rect = self._cell_rect(r, c)
                pygame.draw.rect(self.screen, (255, 255, 255), rect)
                pygame.draw.rect(self.screen, GRID_C, rect, 1)

    def _draw_targets(self, state):
        for pos, colour in [
            (state["red_tgt_pos"],  RED_TGT),
            (state["blue_tgt_pos"], BLU_TGT),
        ]:
            rect = self._cell_rect(*pos).inflate(-4, -4)
            pygame.draw.rect(self.screen, colour, rect, border_radius=6)

    def _draw_objects(self, state):
        for pos, colour, label in [
            (state["red_box_pos"],  RED_BOX, "R"),
            (state["blue_box_pos"], BLU_BOX, "B"),
        ]:
            rect = self._cell_rect(*pos).inflate(-14, -14)
            pygame.draw.rect(self.screen, colour, rect, border_radius=4)
            txt = self.font.render(label, True, (255, 255, 255))
            self.screen.blit(txt, txt.get_rect(center=rect.center))

    def _draw_agent(self, state):
        pos   = state["agent_pos"]
        held  = state.get("held_object")
        rect  = self._cell_rect(*pos)
        cx, cy = rect.centerx, rect.centery
        # Agent body
        pygame.draw.circle(self.screen, AGENT_C, (cx, cy), CELL // 3)
        # Held object indicator
        if held:
            clr = RED_BOX if held == "red" else BLU_BOX
            pygame.draw.circle(self.screen, HELD_C, (cx, cy - CELL // 3 - 4), 7)
            pygame.draw.circle(self.screen, clr,    (cx, cy - CELL // 3 - 4), 5)

    def _draw_panel(self, state, instruction, info):
        panel_y = 8 * CELL + 2 * MARGIN
        pygame.draw.rect(self.screen, PANEL_C,
                         pygame.Rect(0, panel_y, W_WIN, PANEL))
        lines = [f"Instruction: {instruction}"]
        if info:
            lines.append(
                f"Steps: {info.get('steps', 0)}  "
                f"Held: {info.get('held', 'none')}  "
                f"Red done: {info.get('red_done', False)}  "
                f"Blue done: {info.get('blue_done', False)}"
            )
        for i, line in enumerate(lines):
            surf = self.small.render(line, True, TEXT_C)
            self.screen.blit(surf, (MARGIN, panel_y + 10 + i * 18))
