import pygame

class SimpleRenderer:
    # Added a 'title' parameter with a default value
    def __init__(self, width=8, height=8, cell_size=60, title="Pick and Place"):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption(title) # Use the dynamic title
        
        self.COLORS = {
            "bg": (240, 240, 240),
            "grid": (200, 200, 200),
            "agent": (50, 150, 250),   
            "red_box": (220, 50, 50),  
            "blue_box": (50, 50, 220), 
            "red_target": (250, 150, 150),
            "blue_target": (150, 150, 250)
        }

    def render(self, state):
        self.screen.fill(self.COLORS["bg"])

        for x in range(0, self.width * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.COLORS["grid"], (x, 0), (x, self.height * self.cell_size))
        for y in range(0, self.height * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.COLORS["grid"], (0, y), (self.width * self.cell_size, y))

        def draw_rect(pos, color, scale=1.0, width=0):
            if pos[0] == -1: return 
            margin = (self.cell_size * (1 - scale)) / 2
            rect = pygame.Rect(
                pos[0] * self.cell_size + margin, 
                pos[1] * self.cell_size + margin, 
                self.cell_size * scale, 
                self.cell_size * scale
            )
            pygame.draw.rect(self.screen, color, rect, width)

        # Draw Targets (Outlines)
        draw_rect(state["red_target"], self.COLORS["red_target"], scale=0.9, width=4)
        draw_rect(state["blue_target"], self.COLORS["blue_target"], scale=0.9, width=4)

        # Draw Boxes
        draw_rect(state["red_box"], self.COLORS["red_box"], scale=0.6)
        draw_rect(state["blue_box"], self.COLORS["blue_box"], scale=0.6)

        # Draw Agent as a Circle
        ax, ay = state["agent"]
        agent_center = (int(ax * self.cell_size + self.cell_size/2), 
                        int(ay * self.cell_size + self.cell_size/2))
        agent_radius = int((self.cell_size / 2) * 0.8) 
        pygame.draw.circle(self.screen, self.COLORS["agent"], agent_center, agent_radius)

        # Draw Inventory Indicator
        if state["inventory"]:
            inv_color = self.COLORS.get(state["inventory"])
            pygame.draw.circle(self.screen, inv_color, agent_center, self.cell_size // 6)

        pygame.display.flip()

    def close(self):
        pygame.quit()


# import pygame

# class SimpleRenderer:
#     def __init__(self, width=8, height=8, cell_size=60):
#         self.width = width
#         self.height = height
#         self.cell_size = cell_size
        
#         pygame.init()
#         self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
#         pygame.display.set_caption("Pick and Place Data Collection")
        
#         self.COLORS = {
#             "bg": (240, 240, 240),
#             "grid": (200, 200, 200),
#             "agent": (0, 200, 0),   
#             "red_box": (220, 50, 50),  
#             "blue_box": (50, 50, 220), 
#             "red_target": (250, 150, 150),
#             "blue_target": (150, 150, 250)
#         }

#     def render(self, state):
#         self.screen.fill(self.COLORS["bg"])

#         for x in range(0, self.width * self.cell_size, self.cell_size):
#             pygame.draw.line(self.screen, self.COLORS["grid"], (x, 0), (x, self.height * self.cell_size))
#         for y in range(0, self.height * self.cell_size, self.cell_size):
#             pygame.draw.line(self.screen, self.COLORS["grid"], (0, y), (self.width * self.cell_size, y))

#         def draw_rect(pos, color, scale=1.0, width=0):
#             if pos[0] == -1: return 
#             margin = (self.cell_size * (1 - scale)) / 2
#             rect = pygame.Rect(
#                 pos[0] * self.cell_size + margin, 
#                 pos[1] * self.cell_size + margin, 
#                 self.cell_size * scale, 
#                 self.cell_size * scale
#             )
#             pygame.draw.rect(self.screen, color, rect, width)

#         # Draw Targets (Outlines)
#         draw_rect(state["red_target"], self.COLORS["red_target"], scale=0.9, width=4)
#         draw_rect(state["blue_target"], self.COLORS["blue_target"], scale=0.9, width=4)

#         # Draw Boxes
#         draw_rect(state["red_box"], self.COLORS["red_box"], scale=0.6)
#         draw_rect(state["blue_box"], self.COLORS["blue_box"], scale=0.6)

#         # Draw Agent as a Circle
#         ax, ay = state["agent"]
#         agent_center = (int(ax * self.cell_size + self.cell_size/2), 
#                         int(ay * self.cell_size + self.cell_size/2))
#         # Radius set slightly smaller than half-cell size to fit nicely
#         agent_radius = int((self.cell_size / 2) * 0.8) 
#         pygame.draw.circle(self.screen, self.COLORS["agent"], agent_center, agent_radius)

#         # Draw Inventory Indicator (inner circle on top of agent)
#         if state["inventory"]:
#             inv_color = self.COLORS.get(state["inventory"])
#             # Inventory dot remains centered on the agent, smaller radius
#             pygame.draw.circle(self.screen, inv_color, agent_center, self.cell_size // 6)

#         pygame.display.flip()

#     def close(self):
#         pygame.quit()