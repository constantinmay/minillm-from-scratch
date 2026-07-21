"""Causal self-attention with Rotary Positional Embeddings (RoPE)."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import MiniLLMConfig


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute cosine and sine frequencies for RoPE.

    Returns:
        Tensor of shape (max_seq_len, head_dim) where first half is cos, second half is sin.
    """
    assert head_dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim // 2)
    cos_freqs = torch.cos(angles)
    sin_freqs = torch.sin(angles)
    # First half is cos, second half is sin: [cos0, cos1, ..., sin0, sin1, ...]
    return torch.cat([cos_freqs, sin_freqs], dim=-1)  # (max_seq_len, head_dim)


def apply_rotary_emb(
    x: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to a tensor.

    Args:
        x: (B, n_head, T, head_dim)
        freqs: (T, head_dim) precomputed cos/sin frequencies

    Returns:
        Tensor of same shape as x with rotary embedding applied.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2

    cos = freqs[:, :half]  # (T, half)
    sin = freqs[:, half:]  # (T, half)

    # Reshape for broadcasting: (1, 1, T, half)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., :half]
    x2 = x[..., half:]

    rotated = torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )
    return rotated


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o_proj._is_residual = True

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, rope_freqs)
        k = apply_rotary_emb(k, rope_freqs)

        # Scaled dot-product attention with causal mask
        # Use PyTorch 2.0+ efficient SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.config.dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.o_proj(attn_output))
        return output
