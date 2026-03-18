import time
import pygame
import torch
from grid_env_simple import SimplePickPlaceEnv
from renderer_simple import SimpleRenderer
from agent import SimpleLanguageAgent, state_dict_to_tensor, simple_tokenize

def main():
    # 1. Load the trained model
    agent = SimpleLanguageAgent()
    try:
        agent.load_state_dict(torch.load("trained_agent.pth", weights_only=True))
        print("Loaded trained model successfully.")
    except FileNotFoundError:
        print("Model not found. Please run train.py first.")
        return

    # 2. Setup Environment
    env = SimplePickPlaceEnv(width=8, height=8)
    renderer = SimpleRenderer(width=8, height=8, title="AI Evaluation Mode")
    
    # Initialize the first layout
    obs = env.reset()
    renderer.render(obs)
    
    while True:
        # Keep Pygame's event queue moving so the window doesn't freeze
        pygame.event.pump() 
        
        print("\n" + "="*40)
        print("Look at the Pygame window to see the current layout.")
        instruction = input("Enter a task (or type 'r' to reshuffle, 'quit' to exit): ")
        
        # Handle User Intent
        if instruction.lower() in ['quit', 'q', 'exit']:
            break
        elif instruction.lower() == 'r':
            obs = env.reset()
            renderer.render(obs)
            print("--> Environment reshuffled!")
            continue # Loop back and ask for instruction again
        elif instruction.strip() == "":
            print("Instruction cannot be empty.")
            continue
            
        # 3. Process the valid instruction
        text_tokens = torch.tensor(simple_tokenize(instruction), dtype=torch.long)
        
        done = False
        step_count = 0
        max_steps = 30 
        
        print("\nAgent is executing...")
        
        # 4. Agent Execution Loop
        while not done and step_count < max_steps:
            pygame.event.pump() 
            
            # Map state to tensor and ask neural network for action
            state_tensor = state_dict_to_tensor(obs)
            action = agent.act(state_tensor, text_tokens)
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            print(f"Step {step_count}: AI chose Action {action} -> {info['feedback']}")
            
            # Visualize the agent's move
            renderer.render(obs)
            time.sleep(0.4) 
            
            step_count += 1
            
            # Check if agent decided to finish the task
            if action == 5: # PLACE action
                print("--> Agent chose to place an item. Episode finished.")
                done = True

        print("Waiting 2 seconds before the next round...")
        time.sleep(2)
        
        # Automatically reset the environment for the next task
        obs = env.reset()
        renderer.render(obs)

    renderer.close()
    print("Evaluation ended.")

if __name__ == "__main__":
    main()

# import time
# import torch
# from grid_env_simple import SimplePickPlaceEnv
# from renderer_simple import SimpleRenderer
# from agent import SimpleLanguageAgent, state_dict_to_tensor, simple_tokenize

# def main():
#     # 1. Load the trained model
#     agent = SimpleLanguageAgent()
#     try:
#         agent.load_state_dict(torch.load("trained_agent.pth", weights_only=True))
#         print("Loaded trained model.")
#     except FileNotFoundError:
#         print("Model not found. Please run train.py first.")
#         return

#     # 2. Setup Environment
#     env = SimplePickPlaceEnv(width=8, height=8)
#     renderer = SimpleRenderer(width=8, height=8)
    
#     while True:
#         obs = env.reset()
#         renderer.render(obs)
        
#         print("\n" + "="*40)
#         instruction = input("Enter a task for the AI to perform (or 'quit'): ")
#         if instruction.lower() in ['quit', 'q']:
#             break
            
#         # Tokenize the instruction for the agent
#         text_tokens = torch.tensor(simple_tokenize(instruction), dtype=torch.long)
        
#         done = False
#         step_count = 0
#         max_steps = 30 # Prevent infinite loops
        
#         print("\nAgent is running...")
#         while not done and step_count < max_steps:
#             # Convert state to tensor
#             state_tensor = state_dict_to_tensor(obs)
            
#             # Agent decides action
#             action = agent.act(state_tensor, text_tokens)
            
#             # Environment steps
#             obs, reward, done, info = env.step(action)
            
#             print(f"Step {step_count}: AI chose Action {action} -> {info['feedback']}")
            
#             # Visualize
#             renderer.render(obs)
#             time.sleep(0.4) # Slow down so you can watch it move
            
#             step_count += 1
            
#             # Because this environment has no hard-coded "done" condition (since goals are text-based),
#             # we manually stop if the agent places an item, to evaluate if it did it correctly.
#             if action == 5: # 5 is PLACE
#                 print("\nAgent chose to place an item. Stopping episode for review.")
#                 done = True

#     renderer.close()
#     print("Evaluation ended.")

# if __name__ == "__main__":
#     main()