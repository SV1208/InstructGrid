import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def simple_tokenize(text, vocab_size=500, max_length=15):
    """Tokenizes text and pads/truncates it to a strictly fixed length."""
    # 1. Split text into words and hash them into vocabulary indices
    tokens = [hash(word) % vocab_size for word in text.lower().split()]
    
    # 2. Fix the "Length 0" RNN error: Provide a default token if empty
    if len(tokens) == 0:
        tokens = [0]
        
    # 3. Fix the DataLoader batching error: Pad or Truncate to max_length
    if len(tokens) < max_length:
        # Pad with 0s if the instruction is short
        tokens = tokens + [0] * (max_length - len(tokens))
    else:
        # Truncate if the instruction is too long
        tokens = tokens[:max_length]
        
    return tokens

def state_dict_to_tensor(state_dict, width=8, height=8):
    """Maps the dictionary into a 5-channel 2D spatial tensor."""
    # Channels: 0:Agent, 1:RedBox, 2:BlueBox, 3:RedTarget, 4:BlueTarget
    tensor = np.zeros((5, height, width), dtype=np.float32)
    
    def set_pos(channel, pos):
        if pos[0] != -1 and pos[1] != -1: 
            tensor[channel, pos[1], pos[0]] = 1.0

    set_pos(0, state_dict['agent'])
    set_pos(3, state_dict['red_target'])
    set_pos(4, state_dict['blue_target'])
    
    # Ensure boxes follow the agent if they are in the inventory
    if state_dict["inventory"] == "red_box":
        set_pos(1, state_dict['agent'])
    else:
        set_pos(1, state_dict['red_box'])
        
    if state_dict["inventory"] == "blue_box":
        set_pos(2, state_dict['agent'])
    else:
        set_pos(2, state_dict['blue_box'])
    
    return torch.tensor(tensor)

class SimpleLanguageAgent(nn.Module):
    def __init__(self, vocab_size=500, embedding_dim=32, grid_channels=5, hidden_dim=128, num_actions=6):
        super().__init__()
        
        # Text encoding
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.text_rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # Spatial encoding
        self.conv1 = nn.Conv2d(grid_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.state_fc = nn.Linear(32 * 8 * 8, hidden_dim)
        
        # Fusion
        self.fusion_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, state_tensor, text_tokens):
        # 1. Process Text
        embedded_text = self.word_embedding(text_tokens)
        _, text_hidden = self.text_rnn(embedded_text)
        text_features = text_hidden.squeeze(0) 
        
        # 2. Process Image/Map
        x = F.relu(self.conv1(state_tensor))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        state_features = F.relu(self.state_fc(x)) 
        
        # 3. Fuse & Output
        fused = torch.cat((state_features, text_features), dim=1) 
        x = F.relu(self.fusion_fc1(fused))
        return self.policy_head(x) 

    def act(self, state_tensor, text_tokens):
        """Returns highest probability action during evaluation."""
        self.eval()
        with torch.no_grad():
            # Add a batch dimension of 1 since we are processing a single state
            logits = self.forward(state_tensor.unsqueeze(0), text_tokens.unsqueeze(0))
            action = torch.argmax(logits, dim=1).item()
        return action