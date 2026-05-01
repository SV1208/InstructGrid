"""
renderer_simple.py
------------------
Pygame renderer for InstructGrid.

draw()   — pure paint call, no event handling (use in game loops)
render() — draws + handles quit events (use in evaluate.py standalone)
"""

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

CELL   = 68
MARGIN = 10
PANEL  = 130
W_WIN  = CELL * 8 + MARGIN * 2
H_WIN  = CELL * 8 + MARGIN * 2 + PANEL

# --- colour palette ---
BG        = (18,  24,  40)
GRID_LINE = (40,  52,  78)
CELL_BG   = (24,  34,  58)

AGENT_C   = (80, 160, 255)
RED_BOX   = (220, 70,  70)
BLUE_BOX  = (70, 130, 220)
RED_TGT   = (120, 40,  40)
BLUE_TGT  = (40,  70, 130)
RED_RING  = (255, 100, 100)
BLUE_RING = (100, 160, 255)

HELD_RING = (255, 220,  60)

PANEL_BG  = (12,  18,  34)
WHITE     = (230, 235, 245)
MUTED     = (100, 115, 150)
SUCCESS   = (80,  220, 140)
FAIL      = (220,  80,  80)


class Renderer:
    def __init__(self, title="InstructGrid", fps=8):
        if not PYGAME_OK:
            return
        pygame.init()
        self.screen = pygame.display.set_mode((W_WIN, H_WIN))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.fps   = fps
        self.font  = pygame.font.SysFont("consolas", 14)
        self.bold  = pygame.font.SysFont("consolas", 14, bold=True)
        self.small = pygame.font.SysFont("consolas", 11)

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def draw(self, state: dict, instruction: str = "", info: dict = None,
             model_name: str = "", episode_result: str = ""):
        """Pure draw — no event handling."""
        if not PYGAME_OK:
            return
        self.screen.fill(BG)
        self._draw_grid()
        self._draw_targets(state)
        self._draw_objects(state)
        self._draw_agent(state)
        self._draw_panel(state, instruction, info, model_name, episode_result)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def render(self, state: dict, instruction: str = "", info: dict = None,
               model_name: str = "", episode_result: str = ""):
        """Draw + handle quit events. Returns False when user quits."""
        if not PYGAME_OK:
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                return False
        self.draw(state, instruction, info, model_name, episode_result)
        return True

    def close(self):
        if PYGAME_OK:
            pygame.quit()

    # ---------------------------------------------------------------- #
    # Drawing helpers
    # ---------------------------------------------------------------- #

    def _cell_rect(self, r, c):
        return pygame.Rect(MARGIN + c * CELL, MARGIN + r * CELL, CELL, CELL)

    def _draw_grid(self):
        for r in range(8):
            for c in range(8):
                rect = self._cell_rect(r, c)
                pygame.draw.rect(self.screen, CELL_BG, rect)
                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

    def _draw_targets(self, state):
        for pos, fill, ring in [
            (state["red_tgt_pos"],  RED_TGT,  RED_RING),
            (state["blue_tgt_pos"], BLUE_TGT, BLUE_RING),
        ]:
            r, c = int(pos[0]), int(pos[1])
            rect = self._cell_rect(r, c).inflate(-6, -6)
            pygame.draw.rect(self.screen, fill, rect, border_radius=5)
            pygame.draw.rect(self.screen, ring, rect, 2, border_radius=5)
            # "X" target marker
            mid = rect.center
            s   = rect.width // 4
            pygame.draw.line(self.screen, ring,
                             (mid[0]-s, mid[1]-s), (mid[0]+s, mid[1]+s), 2)
            pygame.draw.line(self.screen, ring,
                             (mid[0]+s, mid[1]-s), (mid[0]-s, mid[1]+s), 2)

    def _draw_objects(self, state):
        held = state.get("held_object")
        for pos, fill, colour_name in [
            (state["red_box_pos"],  RED_BOX,  "red"),
            (state["blue_box_pos"], BLUE_BOX, "blue"),
        ]:
            r, c   = int(pos[0]), int(pos[1])
            rect   = self._cell_rect(r, c).inflate(-10, -10)
            pygame.draw.rect(self.screen, fill, rect, border_radius=4)
            # Bold outline if this box is being held
            if held == colour_name:
                pygame.draw.rect(self.screen, HELD_RING, rect, 3, border_radius=4)
            label = self.bold.render("R" if colour_name == "red" else "B",
                                     True, WHITE)
            self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_agent(self, state):
        held     = state.get("held_object")
        r, c     = int(state["agent_pos"][0]), int(state["agent_pos"][1])
        rect     = self._cell_rect(r, c)
        cx, cy   = rect.centerx, rect.centery
        radius   = CELL // 3

        # Shadow circle
        pygame.draw.circle(self.screen, (0, 0, 0), (cx+2, cy+2), radius)
        # Main agent circle — tinted if holding something
        color = (RED_BOX if held == "red" else
                 BLUE_BOX if held == "blue" else AGENT_C)
        pygame.draw.circle(self.screen, color, (cx, cy), radius)
        # White ring outline
        pygame.draw.circle(self.screen, WHITE, (cx, cy), radius, 2)
        # Small dot in centre
        pygame.draw.circle(self.screen, WHITE, (cx, cy), 4)

    def _draw_panel(self, state, instruction, info, model_name, episode_result):
        py = 8 * CELL + 2 * MARGIN
        pygame.draw.rect(self.screen, PANEL_BG,
                         pygame.Rect(0, py, W_WIN, PANEL))
        pygame.draw.line(self.screen, GRID_LINE, (0, py), (W_WIN, py), 1)

        steps    = info.get("steps", 0)   if info else 0
        max_s    = 100
        held     = state.get("held_object") or "none"
        red_done = info.get("red_done", False) if info else False
        blu_done = info.get("blue_done", False) if info else False

        # Row 1: instruction
        instr_surf = self.font.render(
            f"Instruction: {instruction[:60]}", True, WHITE)
        self.screen.blit(instr_surf, (MARGIN, py + 8))

        # Row 2: model name + episode result
        if model_name or episode_result:
            clr = (SUCCESS if "SUCCESS" in episode_result else
                   FAIL    if "FAIL"    in episode_result else MUTED)
            mn_surf = self.small.render(
                f"Model: {model_name}   {episode_result}", True, clr)
            self.screen.blit(mn_surf, (MARGIN, py + 28))

        # Row 3: stats
        held_clr = (RED_RING  if held == "red"  else
                    BLUE_RING if held == "blue" else MUTED)
        held_s   = self.font.render(f"Held: {held.upper()}", True, held_clr)
        self.screen.blit(held_s, (MARGIN, py + 48))

        rd_clr = SUCCESS if red_done  else MUTED
        bd_clr = SUCCESS if blu_done  else MUTED
        self.screen.blit(self.font.render("Red ✓",  True, rd_clr), (180, py + 48))
        self.screen.blit(self.font.render("Blue ✓", True, bd_clr), (270, py + 48))

        # Row 4: step progress bar
        bar_x, bar_y, bar_w, bar_h = MARGIN, py + 72, W_WIN - MARGIN*2, 10
        pygame.draw.rect(self.screen, GRID_LINE,
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=5)
        fill_w = int(bar_w * min(steps / max_s, 1.0))
        bar_clr = (SUCCESS if steps < 60 else
                   (255, 180, 60) if steps < 85 else FAIL)
        if fill_w > 0:
            pygame.draw.rect(self.screen, bar_clr,
                             pygame.Rect(bar_x, bar_y, fill_w, bar_h),
                             border_radius=5)
        step_s = self.small.render(f"Steps: {steps}/{max_s}", True, MUTED)
        self.screen.blit(step_s, (MARGIN, py + 86))

        # Row 5: legend
        legend = [("● Agent", AGENT_C), ("■ Red box", RED_BOX),
                  ("■ Blue box", BLUE_BOX), ("✕ Red tgt", RED_RING),
                  ("✕ Blue tgt", BLUE_RING)]
        lx = MARGIN
        for txt, clr in legend:
            s = self.small.render(txt, True, clr)
            self.screen.blit(s, (lx, py + 108))
            lx += s.get_width() + 14
