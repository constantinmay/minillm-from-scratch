"""Export a training checkpoint as a verified inference-only artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from model.config import MiniLLMConfig
from model.gpt import MiniLLM
from train.common import checkpoint_model_state, load_checkpoint, sha256_file


INFERENCE_FORMAT = "minillm-inference"
INFERENCE_VERSION = 1


def checkpoint_format(checkpoint: dict[str, Any]) -> str:
    """Return a short, user-facing checkpoint format label."""
    if checkpoint.get("checkpoint_format") == INFERENCE_FORMAT:
        return f"{INFERENCE_FORMAT}-v{checkpoint.get('checkpoint_version', '?')}"
    version = checkpoint.get("checkpoint_version")
    return f"training-v{version}" if version is not None else "legacy-training"


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> MiniLLMConfig:
    """Read an embedded model config without guessing missing architecture values."""
    config = checkpoint.get("model_config", checkpoint.get("config"))
    if not isinstance(config, dict):
        raise ValueError(
            "Checkpoint does not contain a reliable model_config/config dictionary; "
            "the architecture cannot be inferred safely."
        )
    try:
        return MiniLLMConfig(**config)
    except (AssertionError, TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint model configuration is invalid: {exc}") from exc


def model_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract model parameters from inference, v2, or legacy checkpoints."""
    if checkpoint.get("checkpoint_format") == INFERENCE_FORMAT:
        state = checkpoint.get("model_state")
    else:
        state = checkpoint_model_state(checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain a non-empty model state dictionary")
    return state


def load_model_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    fallback_config: MiniLLMConfig | None = None,
) -> tuple[MiniLLM, dict[str, Any], MiniLLMConfig]:
    """Load any supported checkpoint into a model for inference.

    ``fallback_config`` exists for old raw state dictionaries used by the demo.
    Release export never supplies it because exported metadata must be derived
    from the source checkpoint itself.
    """
    checkpoint = load_checkpoint(str(path), device=device)
    embedded_config = checkpoint.get("model_config", checkpoint.get("config"))
    if isinstance(embedded_config, dict):
        config = model_config_from_checkpoint(checkpoint)
    elif fallback_config is not None:
        config = fallback_config
    else:
        config = model_config_from_checkpoint(checkpoint)
    model = MiniLLM(config).to(device)
    model.load_state_dict(model_state_from_checkpoint(checkpoint), strict=True)
    model.eval()
    return model, checkpoint, config


def _verification_input(config: MiniLLMConfig, device: str) -> torch.Tensor:
    length = min(4, config.block_size)
    values = torch.arange(length, dtype=torch.long, device=device)
    return values.remainder(config.vocab_size).unsqueeze(0)


@torch.no_grad()
def verify_export(
    source_model: MiniLLM,
    exported_path: str | Path,
    *,
    device: str = "cpu",
) -> float:
    """Reload an export and return the maximum absolute logits difference."""
    exported_model, exported, exported_config = load_model_checkpoint(
        exported_path, device=device
    )
    if exported.get("checkpoint_format") != INFERENCE_FORMAT:
        raise ValueError("Exported checkpoint has an unexpected checkpoint_format")
    if exported.get("checkpoint_version") != INFERENCE_VERSION:
        raise ValueError("Exported checkpoint has an unsupported checkpoint_version")
    if exported_config.to_dict() != source_model.config.to_dict():
        raise ValueError("Exported model configuration differs from the source")

    source_state = source_model.state_dict()
    exported_state = exported_model.state_dict()
    if source_state.keys() != exported_state.keys():
        raise ValueError("Exported parameter names differ from the source")
    for name, source_tensor in source_state.items():
        exported_tensor = exported_state[name]
        if source_tensor.shape != exported_tensor.shape:
            raise ValueError(f"Exported parameter shape differs for {name}")
        if not torch.equal(source_tensor, exported_tensor):
            raise ValueError(f"Exported parameter values differ for {name}")

    input_ids = _verification_input(source_model.config, device)
    source_logits = source_model(input_ids)["logits"]
    exported_logits = exported_model(input_ids)["logits"]
    if not torch.allclose(source_logits, exported_logits, rtol=1e-6, atol=1e-7):
        raise ValueError("Exported model logits differ from the source checkpoint")
    return float((source_logits - exported_logits).abs().max().item())


def export_inference_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    *,
    training_stage: str,
    tokenizer_path: str | Path | None = None,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create, verify, and checksum an inference-only checkpoint."""
    source = Path(input_path).expanduser()
    output = Path(output_path).expanduser()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    if not source.is_file():
        raise FileNotFoundError(f"Input checkpoint was not found: {source.resolve()}")
    if not training_stage.strip():
        raise ValueError("training_stage must not be empty")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if not overwrite and (output.exists() or checksum_path.exists()):
        raise FileExistsError(
            f"Output already exists: {output if output.exists() else checksum_path}. "
            "Use --overwrite to replace it."
        )

    tokenizer_metadata = None
    if tokenizer_path is not None:
        tokenizer = Path(tokenizer_path).expanduser()
        if not tokenizer.is_file():
            raise FileNotFoundError(f"Tokenizer was not found: {tokenizer.resolve()}")
        tokenizer_metadata = {
            "filename": tokenizer.name,
            "sha256": sha256_file(tokenizer),
        }

    source_model, source_checkpoint, config = load_model_checkpoint(
        source, device=device
    )
    source_model.eval()
    artifact = {
        "checkpoint_format": INFERENCE_FORMAT,
        "checkpoint_version": INFERENCE_VERSION,
        "model_state": source_model.state_dict(),
        "model_config": config.to_dict(),
        "training_stage": training_stage.strip(),
        "source_checkpoint_sha256": sha256_file(source),
        "source_checkpoint_step": source_checkpoint.get("step"),
        "source_git_commit": source_checkpoint.get("git_commit", "unknown"),
        "parameter_count": source_model.num_params(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tokenizer": tokenizer_metadata,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        torch.save(artifact, temp_path)
        max_abs_logit_diff = verify_export(source_model, temp_path, device=device)
        os.replace(temp_path, output)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    output_sha256 = sha256_file(output)
    checksum_path.write_text(
        f"{output_sha256}  {output.name}\n", encoding="utf-8"
    )
    return {
        "output": output,
        "checksum_path": checksum_path,
        "sha256": output_sha256,
        "parameter_count": source_model.num_params(),
        "max_abs_logit_diff": max_abs_logit_diff,
        "checkpoint_format": checkpoint_format(artifact),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Training checkpoint to export")
    parser.add_argument("--output", required=True, help="Inference checkpoint path")
    parser.add_argument("--stage", required=True, help="Training stage label")
    parser.add_argument("--tokenizer", help="Optional tokenizer file for checksum metadata")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = export_inference_checkpoint(
        args.input,
        args.output,
        training_stage=args.stage,
        tokenizer_path=args.tokenizer,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"Exported: {result['output']}")
    print(f"Format: {result['checkpoint_format']}")
    print(f"Parameters: {result['parameter_count']:,}")
    print(f"SHA256: {result['sha256']}")
    print(f"Checksum: {result['checksum_path']}")
    print(f"Maximum absolute logits difference: {result['max_abs_logit_diff']:.3g}")


if __name__ == "__main__":
    main()
