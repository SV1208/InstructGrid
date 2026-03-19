"""
utils/tokenizer.py
------------------
Simple word-level tokenizer shared by all model versions.

Builds a vocabulary from the training data and converts instructions
to fixed-length integer token sequences.

Usage:
    tok = Tokenizer()
    tok.build_from_file("language_demo_data.jsonl")
    tok.save("utils/vocab.json")

    # Later:
    tok = Tokenizer.load("utils/vocab.json")
    tokens = tok.encode("place the red box in the blue zone", max_len=20)
    # → torch.LongTensor of shape (20,)
"""

import json
import re
import torch


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Tokenizer:
    def __init__(self):
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN}

    # ------------------------------------------------------------------
    # Building vocabulary
    # ------------------------------------------------------------------

    def build_from_file(self, jsonl_path: str, min_freq: int = 1):
        """Build vocab from all instruction fields in a .jsonl demo file."""
        from collections import Counter
        counter = Counter()
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("instruction"):
                    for w in self._tokenize_str(entry["instruction"]):
                        counter[w] += 1
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
        print(f"[Tokenizer] Vocabulary size: {len(self.word2idx)}")

    def build_from_list(self, instructions: list):
        """Build vocab from a list of instruction strings."""
        for instr in instructions:
            for w in self._tokenize_str(instr):
                if w not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str, max_len: int = 20) -> torch.Tensor:
        """Return a LongTensor of shape (max_len,), zero-padded."""
        words = self._tokenize_str(text)
        ids   = [self.word2idx.get(w, 1) for w in words]   # 1 = UNK
        ids   = ids[:max_len]
        ids  += [0] * (max_len - len(ids))                  # 0 = PAD
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids) -> str:
        return " ".join(self.idx2word.get(int(i), UNK_TOKEN)
                        for i in token_ids if int(i) != 0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.word2idx, f, indent=2)
        print(f"[Tokenizer] Vocab saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        tok = cls()
        with open(path) as f:
            tok.word2idx = json.load(f)
        tok.word2idx = {k: int(v) for k, v in tok.word2idx.items()}
        tok.idx2word = {v: k for k, v in tok.word2idx.items()}
        print(f"[Tokenizer] Loaded vocab ({len(tok.word2idx)} tokens) from {path}")
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_str(text: str) -> list:
        return re.findall(r"[a-z0-9]+", text.lower())
