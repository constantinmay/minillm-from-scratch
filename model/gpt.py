"""MiniLLM: LLaMA-style decoder-only Transformer language model."""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import MiniLLMConfig
from model.block import TransformerBlock, RMSNorm
from model.attention import precompute_rope_freqs


class MiniLLM(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = RMSNorm(config.n_embd, config.rms_norm_eps)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        # Precompute RoPE frequencies
        rope_freqs = precompute_rope_freqs(config.head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, '_is_residual'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Token embeddings (no positional embedding table — RoPE handles positions)
        x = self.dropout(self.token_embedding(input_ids))

        # Get RoPE frequencies for current sequence length
        rope_freqs = self.rope_freqs[:T]

        # Transformer blocks
        for block in self.layers:
            x = block(x, rope_freqs)

        # Final norm and project to logits
        x = self.ln_f(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if targets is not None:
            # Targets are already shifted by 1 in the dataset
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
