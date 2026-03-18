import json
import time
import pygame
from grid_env_simple import SimplePickPlaceEnv
from renderer_simple import SimpleRenderer

def main():
    env = SimplePickPlaceEnv(width=8, height=8)
    renderer = SimpleRenderer(width=8, height=8)
    log_file = open('language_demo_data.jsonl', 'a')
    
    print("--- Language-Conditioned Teleoperation ---")
    print("Use CTRL+C in the terminal to exit the program completely.")
    
    try:
        while True:
            # 1. Reset and Render
            obs = env.reset()
            renderer.render(obs)
            
            # 2. Ask for Instruction
            print("\n" + "="*40)
            print("Look at the Pygame window.")
            instruction = input("Enter instruction (or type 'quit' to exit): ")
            
            if instruction.lower() in ['quit', 'q', 'exit']:
                break
                
            obs["instruction"] = instruction
            episode_buffer = [] # Store steps here temporarily
            
            print("\nPygame active! Control the agent.")
            print("Keys: Arrow Keys/WASD (Move), P (Pick), Space (Place)")
            print("ENTER to SAVE task. ESC to DISCARD and try a new layout.")
            
            recording = True
            while recording:
                action = None
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        recording = False
                        raise KeyboardInterrupt # Force exit
                        
                    elif event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_UP, pygame.K_w]: action = 0
                        elif event.key in [pygame.K_DOWN, pygame.K_s]: action = 1
                        elif event.key in [pygame.K_LEFT, pygame.K_a]: action = 2
                        elif event.key in [pygame.K_RIGHT, pygame.K_d]: action = 3
                        elif event.key == pygame.K_p: action = 4
                        elif event.key == pygame.K_SPACE: action = 5
                        elif event.key == pygame.K_ESCAPE: 
                            print("--> Demonstration DISCARDED. Resetting...")
                            recording = False # Break inner loop, don't save
                        elif event.key == pygame.K_RETURN:  
                            print("--> Task SUCCESSFUL. Saving to dataset...")
                            # Flush buffer to file
                            for transition in episode_buffer:
                                log_file.write(json.dumps(transition) + "\n")
                            recording = False # Break inner loop

                if action is not None and recording:
                    next_obs, reward, done, info = env.step(action)
                    
                    # Add to temporary buffer instead of writing directly
                    episode_buffer.append({
                        "instruction": obs["instruction"],
                        "state": obs,
                        "action": action,
                        "next_state": next_obs
                    })
                    
                    obs = next_obs
                    renderer.render(obs)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        log_file.close()
        renderer.close()
        print("Data saved safely. Goodbye!")

if __name__ == "__main__":
    main()


# import json
# import time
# import pygame
# from grid_env_simple import SimplePickPlaceEnv
# from renderer_simple import SimpleRenderer

# def main():
#     env = SimplePickPlaceEnv(width=8, height=8)
    
#     # 1. Reset and initialize the environment FIRST
#     obs = env.reset()
    
#     # 2. Render it so the human can see the layout
#     renderer = SimpleRenderer(width=8, height=8)
#     renderer.render(obs)
    
#     # 3. Ask for the instruction based on what the human sees
#     print("--- Language-Conditioned Teleoperation ---")
#     print("Look at the Pygame window to see the randomized environment.")
#     instruction = input("Enter the task instruction based on the layout: ")
    
#     # Inject the instruction into the observation state
#     obs["instruction"] = instruction
    
#     log_file = open('language_demo_data.jsonl', 'a')
    
#     print("\nPygame window is now active! Please click on the window to control the agent.")
#     print("Controls: Arrow Keys or W/A/S/D (Move), P (Pick), Space (Place)")
#     print("Press ENTER (Return) when you have finished the task.")
#     print("Press ESC to quit without saving the final step.")
    
#     running = True
#     while running:
#         action = None
        
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type == pygame.KEYDOWN:
#                 if event.key in [pygame.K_UP, pygame.K_w]: action = 0
#                 elif event.key in [pygame.K_DOWN, pygame.K_s]: action = 1
#                 elif event.key in [pygame.K_LEFT, pygame.K_a]: action = 2
#                 elif event.key in [pygame.K_RIGHT, pygame.K_d]: action = 3
#                 elif event.key == pygame.K_p: action = 4
#                 elif event.key == pygame.K_SPACE: action = 5
#                 elif event.key == pygame.K_ESCAPE: 
#                     running = False
#                 elif event.key == pygame.K_RETURN:  # User declares task is done
#                     print("\nTask manually marked as Completed!")
#                     running = False

#         if action is not None and running:
#             next_obs, reward, done, info = env.step(action)
            
#             # Log the transition
#             transition = {
#                 "instruction": obs["instruction"],
#                 "state": obs,
#                 "action": action,
#                 "next_state": next_obs,
#                 "feedback": info["feedback"]
#             }
#             log_file.write(json.dumps(transition) + "\n")
            
#             print(f"Action: {action} | Feedback: {info['feedback']}")
            
#             obs = next_obs
#             renderer.render(obs)

#     log_file.close()
#     renderer.close()
#     print("Data logged successfully and environment closed.")

# if __name__ == "__main__":
#     main()

# import json
# import time
# from grid_env_v2 import GridEnvV2


# import json
# import time
# import pygame
# from grid_env_v2 import GridEnvV2
# from renderer_v2 import GridRendererV2

# def main():
#     env = GridEnvV2(width=8, height=8)
    
#     # We ask for the instruction in the terminal *before* taking over with Pygame
#     print("--- Language-Conditioned Teleoperation ---")
#     instruction = input("Enter the task instruction (e.g., 'unlock the door and move the blue box to the target'): ")
    
#     obs = env.reset(instruction=instruction)
    
#     # Initialize the renderer
#     renderer = GridRendererV2(width=8, height=8)
#     renderer.render(obs)
    
#     log_file = open('language_demo_data.jsonl', 'a')
    
#     print("\nPygame window is now active! Please click on the window to control the agent.")
#     print("Controls: Arrow Keys or W/A/S/D (Move), P (Pick), Space (Place/Interact), Esc (Quit)")
    
#     running = True
#     while running:
#         action = None
        
#         # Pygame Event Loop
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type == pygame.KEYDOWN:
#                 # Support both WASD and Arrow Keys for movement
#                 if event.key in [pygame.K_UP, pygame.K_w]: action = 0
#                 elif event.key in [pygame.K_DOWN, pygame.K_s]: action = 1
#                 elif event.key in [pygame.K_LEFT, pygame.K_a]: action = 2
#                 elif event.key in [pygame.K_RIGHT, pygame.K_d]: action = 3
#                 elif event.key == pygame.K_p: action = 4
#                 elif event.key == pygame.K_SPACE: action = 5
#                 elif event.key == pygame.K_ESCAPE: running = False

#         # If a valid action key was pressed, step the environment
#         if action is not None:
#             next_obs, reward, done, info = env.step(action)
            
#             # Log the transition
#             transition = {
#                 "instruction": obs["instruction"],
#                 "state": obs,
#                 "action": action,
#                 "next_state": next_obs,
#                 "reward": reward,
#                 "feedback": info["feedback"]
#             }
#             log_file.write(json.dumps(transition) + "\n")
            
#             print(f"Action: {action} | Feedback: {info['feedback']} | Reward: {reward}")
            
#             # Update the state and the Pygame display
#             obs = next_obs
#             renderer.render(obs)
            
#             if done:
#                 print("\nTask Completed Successfully!")
#                 time.sleep(1.5) # Pause to let the human see the final success state
#                 running = False

#     log_file.close()
#     renderer.close()
#     print("Data logged successfully and environment closed.")

# if __name__ == "__main__":
#     main()

# # def main():
# #     env = GridEnvV2(width=8, height=8)
    
# #     # In a real setup, you would hook up the Pygame renderer here
# #     print("--- Language-Conditioned Teleoperation ---")
# #     instruction = input("Enter the task instruction (e.g., 'unlock the door and move the blue box to the target'): ")
    
# #     obs = env.reset(instruction=instruction)
    
# #     log_file = open('language_demo_data.jsonl', 'a')
    
# #     print("\nControls: w/s/a/d (Move), p (Pick), space (Place/Interact), q (Quit)")
    
# #     running = True
# #     while running:
# #         # Simple terminal visualizer for debugging
# #         print(f"\nState: Agent{obs['agent']}, Inv:{obs['inventory']}, DoorLocked:{obs['door_locked']}")
# #         print(f"Goal: {obs['instruction']}")
        
# #         val = input("Action: ")
# #         action = None
# #         if val == 'w': action = 0
# #         elif val == 's': action = 1
# #         elif val == 'a': action = 2
# #         elif val == 'd': action = 3
# #         elif val == 'p': action = 4
# #         elif val == ' ': action = 5
# #         elif val == 'q': break
# #         else: continue

# #         next_obs, reward, done, info = env.step(action)
        
# #         # Log the transition
# #         transition = {
# #             "instruction": obs["instruction"],
# #             "state": obs,
# #             "action": action,
# #             "next_state": next_obs,
# #             "reward": reward,
# #             "feedback": info["feedback"]
# #         }
# #         log_file.write(json.dumps(transition) + "\n")
        
# #         print(f"Feedback: {info['feedback']} | Reward: {reward}")
# #         obs = next_obs
        
# #         if done:
# #             print("Task Completed Successfully!")
# #             break

# #     log_file.close()

# if __name__ == "__main__":
#     main()