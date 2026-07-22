"""Shared, beginner-readable training and reproducibility utilities."""

import datetime as dt
import hashlib
import json
import os
import platform
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from model.config import MiniLLMConfig


def get_device() -> str:
    """Return 'cuda' if available else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow deterministic kernels."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Return an explicitly seeded generator for DataLoader shuffling."""
    return torch.Generator().manual_seed(int(seed))


def require_file(path: str, label: str) -> Path:
    """Resolve a required file or fail before expensive initialization begins."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Required {label} was not found: {resolved}. "
            "Check the configured path and create/download the required artifact first."
        )
    return resolved


def validate_training_config(cfg: dict, block_size: int) -> None:
    """Validate common positive-valued training fields."""
    values = {
        "block_size": block_size,
        "batch_size": cfg.get("batch_size"),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps"),
        "max_steps": cfg.get("max_steps"),
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _numpy_state_to_dict(state: tuple) -> dict:
    return {
        "bit_generator": state[0],
        "state": state[1].tolist(),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _numpy_state_from_dict(state: dict) -> tuple:
    return (
        state["bit_generator"],
        np.asarray(state["state"], dtype=np.uint32),
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )


def capture_rng_state() -> dict:
    """Capture RNG state in a checkpoint-friendly representation."""
    state = {
        "python": random.getstate(),
        "numpy": _numpy_state_to_dict(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    """Restore any RNG states present in a v2 checkpoint."""
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(_numpy_state_from_dict(state["numpy"]))
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


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

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if max_lr <= 0 or not 0 <= min_lr <= max_lr:
        raise ValueError("learning rates must satisfy 0 <= min_lr <= max_lr")

    min_lr_ratio = min_lr / max_lr

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # LambdaLR expects a multiplier, not an absolute learning rate.
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def get_git_commit() -> str:
    """Return the current commit without failing outside a Git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    path,
    config,
    *,
    scaler=None,
    dataloader_state: Optional[dict] = None,
    training_config: Optional[dict] = None,
    checkpoint_type: str = "training",
):
    """Save a versioned training checkpoint while retaining legacy field names."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    model_config = config if isinstance(config, dict) else config.to_dict()
    checkpoint = {
        "checkpoint_version": 2,
        "project": "minillm-from-scratch",
        "checkpoint_type": checkpoint_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "dataloader_state": dataloader_state,
        "step": int(step),
        "config": model_config,
        "model_config": model_config,
        "training_config": training_config,
        "rng_state": capture_rng_state(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "git_commit": get_git_commit(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):
    """Load old or v2 checkpoint dictionaries with a clear missing-file error."""
    resolved = require_file(path, "checkpoint")
    checkpoint = torch.load(resolved, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {resolved}")
    return checkpoint


def checkpoint_model_state(checkpoint: dict) -> dict:
    """Extract model weights from supported training/inference formats."""
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def validate_checkpoint_model_config(
    checkpoint: dict, expected: MiniLLMConfig, label: str
) -> None:
    """Compare architecture metadata when present; legacy raw state dicts remain valid."""
    saved = checkpoint.get("model_config", checkpoint.get("config"))
    if not isinstance(saved, dict):
        return
    expected_dict = expected.to_dict()
    keys = (
        "vocab_size", "block_size", "n_layer", "n_head", "n_embd",
        "intermediate_size", "bias", "weight_tying",
    )
    mismatches = [
        f"{key}: checkpoint={saved[key]!r}, expected={expected_dict[key]!r}"
        for key in keys
        if key in saved and saved[key] != expected_dict[key]
    ]
    if mismatches:
        raise ValueError(f"{label} model config is incompatible: " + "; ".join(mismatches))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: str | Path, root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name


def write_run_manifest(
    output_dir: str | Path,
    *,
    config_path: str | None,
    config: dict,
    seed: int,
    device: str,
    checkpoint_paths: list[str] | None = None,
    command: str | None = None,
) -> Path:
    """Write non-secret run metadata for training or evaluation."""
    root = Path(__file__).resolve().parents[1]
    checkpoints = [Path(path) for path in (checkpoint_paths or [])]
    relative_checkpoints = [_relative_path(path, root) for path in checkpoints]
    checksums = {
        _relative_path(path, root): sha256_file(path)
        for path in checkpoints
        if path.is_file()
    }
    manifest = {
        "git_commit": get_git_commit(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command or " ".join(sys.argv),
        "config_path": _relative_path(config_path, root) if config_path else None,
        "config": config,
        "seed": int(seed),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "checkpoint_paths": relative_checkpoints,
        "checkpoint_sha256": checksums,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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
