"""
models/
-------
Experimental model versions. All are drop-in replacements for agent.py.

  agent_v2.py        Stage 1 — instruction every step + LayerNorm + Dropout
  agent_v3.py        Stage 2 — cross-attention fusion
  agent_v3_lstm.py   Stage 2 — LSTM temporal policy head
  agent_v4_film.py   Stage 3 — FiLM conditioning + auxiliary loss
"""
from models.agent_v2 import AgentV2
from models.agent_v3 import AgentV3
from models.agent_v3_lstm import AgentV3LSTM
from models.agent_v4_film import AgentV4FiLM
