"""Transformer block with Pre-RMSNorm, SwiGLU, and residual connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import MiniLLMConfig
from model.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), rope_freqs)
        x = x + self.mlp(self.ln_2(x))
        return x
