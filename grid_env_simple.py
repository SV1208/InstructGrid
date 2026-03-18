import random
import copy

class SimplePickPlaceEnv:
    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height
        self.action_space = [0, 1, 2, 3, 4, 5]
        self.state = None

    def reset(self):
        """Resets the environment. Instruction will be injected later."""
        def rand_pos():
            return [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]

        self.state = {
            "agent": rand_pos(),
            "red_box": rand_pos(),
            "blue_box": rand_pos(),
            "red_target": rand_pos(),
            "blue_target": rand_pos(),
            "inventory": None, # Can hold 'red_box' or 'blue_box'
            "instruction": ""  # Left blank initially
        }
        return copy.deepcopy(self.state)

    def step(self, action):
        if self.state is None:
            raise ValueError("Call reset() before step()")

        info = {"feedback": "Moved"}
        ax, ay = self.state["agent"]
        dx, dy = 0, 0

        if action == 0: dy = -1
        elif action == 1: dy = 1
        elif action == 2: dx = -1
        elif action == 3: dx = 1

        # Handle Movement
        if action in [0, 1, 2, 3]:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                self.state["agent"] = [nx, ny]
            else:
                info["feedback"] = "Collision: Wall"

        # Handle Pick
        elif action == 4:
            if self.state["inventory"] is not None:
                info["feedback"] = "Inventory full"
            else:
                agent_pos = self.state["agent"]
                if agent_pos == self.state["red_box"]:
                    self.state["inventory"] = "red_box"
                    self.state["red_box"] = [-1, -1]
                    info["feedback"] = "Picked up red box"
                elif agent_pos == self.state["blue_box"]:
                    self.state["inventory"] = "blue_box"
                    self.state["blue_box"] = [-1, -1]
                    info["feedback"] = "Picked up blue box"
                else:
                    info["feedback"] = "Nothing to pick up"

        # Handle Place
        elif action == 5:
            if self.state["inventory"] is not None:
                item = self.state["inventory"]
                self.state[item] = copy.deepcopy(self.state["agent"])
                self.state["inventory"] = None
                info["feedback"] = f"Placed {item}"
            else:
                info["feedback"] = "Nothing to place"

        # Reward and Done are set to 0/False. 
        # For Behavioral Cloning, we rely on the human to declare when the task is done.
        return copy.deepcopy(self.state), 0.0, False, info