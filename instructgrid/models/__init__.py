"""models/ — registry of all model classes."""

from models.model_cnn_gru       import CNNGRUAgent
from models.model_attention      import AttentionAgent
from models.model_lstm_attention import LSTMAttentionAgent
from models.model_film           import FiLMAgent

MODEL_REGISTRY = {
    "cnn_gru":        CNNGRUAgent,
    "attention":      AttentionAgent,
    "lstm_attention": LSTMAttentionAgent,
    "film":           FiLMAgent,
}

ALL_MODEL_NAMES = list(MODEL_REGISTRY.keys())


def build_model(name: str, vocab_size: int, cfg: dict):
    cls    = MODEL_REGISTRY[name]
    kwargs = dict(
        vocab_size  = vocab_size,
        embed_dim   = cfg.get("embed_dim",   64),
        hidden_dim  = cfg.get("hidden_dim",  256),
        dropout     = cfg.get("dropout",     0.2),
    )
    if name in ("attention", "lstm_attention", "film"):
        kwargs["cnn_channels"] = cfg.get("cnn_channels", 64)
        kwargs["num_heads"]    = cfg.get("num_heads",     4)
    if name == "lstm_attention":
        kwargs["lstm_hidden"]     = cfg.get("lstm_hidden",     256)
        kwargs["num_lstm_layers"] = cfg.get("num_lstm_layers",   1)
    return cls(**kwargs)
