"""Shared training utilities for MiniLLM."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import json
import yaml

from model.config import MiniLLMConfig


def get_device() -> str:
    """Return 'cuda' if available else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_optimizer(model, lr, weight_decay, betas=(0.9, 0.95)):
    """AdamW with weight decay only on 2D+ params (weight matrices).

    This follows the GPT-2 / GPT-3 approach: weight decay is applied only to
    parameters that are at least 2-dimensional (i.e. weight matrices, not biases
    or layer norms).
    """
    # Separate parameters into those that should have weight decay and those that shouldn't
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=False)
    return optimizer


def create_scheduler(optimizer, max_steps, warmup_steps, max_lr, min_lr):
    """Cosine LR with linear warmup. Returns LambdaLR."""
    max_lr = float(max_lr)
    min_lr = float(min_lr)
    warmup_steps = int(warmup_steps)
    max_steps = int(max_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())
            return min_lr + (max_lr - min_lr) * cosine_decay / max_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def save_checkpoint(model, optimizer, scheduler, step, path, config):
    """Save dict with model_state_dict, optimizer_state_dict, scheduler_state_dict, step, config."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": config if isinstance(config, dict) else config.to_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):
    """Load checkpoint dict."""
    return torch.load(path, map_location=device, weights_only=False)


class AvgMetric:
    """Running average tracker."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counts = []

    def update(self, value, n=1):
        self.values.append(value * n)
        self.counts.append(n)

    @property
    def avg(self):
        if not self.counts or sum(self.counts) == 0:
            return 0.0
        return sum(self.values) / sum(self.counts)
