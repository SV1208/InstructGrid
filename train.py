import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from agent import SimpleLanguageAgent, state_dict_to_tensor, simple_tokenize

class BCDataset(Dataset):
    def __init__(self, jsonl_file, vocab_size=500):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Convert state to spatial tensor
        state_tensor = state_dict_to_tensor(item["state"])
        
        # 2. Tokenize text (This will now safely pad/truncate to length 15)
        tokens = simple_tokenize(item["instruction"])
        text_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # 3. Extract action label
        action_label = torch.tensor(item["action"], dtype=torch.long)
        
        return state_tensor, text_tensor, action_label

def main():
    dataset = BCDataset('language_demo_data.jsonl')
    
    if len(dataset) == 0:
        print("Dataset is empty! Run language_logger.py first.")
        return
        
    # batch_size=16 requires all tensors to be exactly the same shape. 
    # Our updated tokenizer ensures this.
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    agent = SimpleLanguageAgent()
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 20
    print(f"Training on {len(dataset)} transitions...")
    
    for epoch in range(epochs):
        agent.train()
        total_loss = 0
        for states, texts, actions in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            logits = agent(states, texts)
            
            # Compute loss
            loss = criterion(logits, actions)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    # Save the weights so evaluate.py can load them
    torch.save(agent.state_dict(), "trained_agent.pth")
    print("\nModel successfully saved as 'trained_agent.pth'")

if __name__ == "__main__":
    main()