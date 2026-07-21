"""Model configuration for MiniLLM."""

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class MiniLLMConfig:
    name: str = "minillm"
    description: str = ""

    vocab_size: int = 8000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    intermediate_size: int = 1536
    dropout: float = 0.1

    norm_type: str = "rmsnorm"
    activation: str = "swiglu"
    pos_embedding: str = "rope"
    bias: bool = False
    weight_tying: bool = True

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        assert self.vocab_size > 0
        assert self.block_size > 0
        assert self.n_layer > 0

    @classmethod
    def from_yaml(cls, path: str) -> "MiniLLMConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "intermediate_size": self.intermediate_size,
            "dropout": self.dropout,
            "bias": self.bias,
            "weight_tying": self.weight_tying,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
        }
