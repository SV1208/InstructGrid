"""utils/tokenizer.py — word-level tokenizer shared by all models."""

import json, re
import torch

PAD, UNK = "<PAD>", "<UNK>"


class Tokenizer:
    def __init__(self):
        self.w2i = {PAD: 0, UNK: 1}
        self.i2w = {0: PAD, 1: UNK}

    # -- building -------------------------------------------------------

    def build_from_files(self, paths: list):
        from collections import Counter
        counter = Counter()
        for path in paths:
            try:
                with open(path) as f:
                    for line in f:
                        e = json.loads(line)
                        if e.get("instruction"):
                            for w in self._tok(e["instruction"]):
                                counter[w] += 1
            except FileNotFoundError:
                pass
        for w in counter:
            if w not in self.w2i:
                idx = len(self.w2i)
                self.w2i[w] = idx
                self.i2w[idx] = w
        print(f"[Tokenizer] vocab size: {len(self.w2i)}")

    def build_from_catalogue(self, catalogue: dict):
        """Build from the full instruction catalogue (covers every phrasing)."""
        for phrasings in catalogue.values():
            for phrase in phrasings:
                for w in self._tok(phrase):
                    if w not in self.w2i:
                        idx = len(self.w2i)
                        self.w2i[w] = idx
                        self.i2w[idx] = w
        print(f"[Tokenizer] vocab size (catalogue): {len(self.w2i)}")

    # -- encode / decode ------------------------------------------------

    def encode(self, text: str, max_len: int = 20) -> torch.Tensor:
        ids = [self.w2i.get(w, 1) for w in self._tok(text)][:max_len]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        return " ".join(self.i2w.get(int(i), UNK)
                        for i in ids if int(i) != 0)

    # -- persistence ----------------------------------------------------

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.w2i, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        tok = cls()
        with open(path) as f:
            tok.w2i = {k: int(v) for k, v in json.load(f).items()}
        tok.i2w = {v: k for k, v in tok.w2i.items()}
        print(f"[Tokenizer] loaded {len(tok.w2i)} tokens from {path}")
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.w2i)

    @staticmethod
    def _tok(text: str) -> list:
        return re.findall(r"[a-z0-9]+", text.lower())
