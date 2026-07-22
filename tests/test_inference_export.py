"""CPU-only tests for verified inference checkpoint export."""

import hashlib

import pytest
import torch

from model.config import MiniLLMConfig
from model.gpt import MiniLLM
from scripts.export_inference_checkpoint import (
    INFERENCE_FORMAT,
    export_inference_checkpoint,
    load_model_checkpoint,
)
from train.common import create_optimizer, save_checkpoint


FORBIDDEN_TRAINING_FIELDS = {
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "rng_state",
    "dataloader_state",
    "training_config",
}


def tiny_config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=8,
        intermediate_size=16,
        dropout=0.0,
    )


def assert_export_matches(source_model, output, source_step):
    exported = torch.load(output, map_location="cpu", weights_only=False)
    assert exported["checkpoint_format"] == INFERENCE_FORMAT
    assert exported["checkpoint_version"] == 1
    assert exported["model_config"] == source_model.config.to_dict()
    assert exported["source_checkpoint_step"] == source_step
    assert FORBIDDEN_TRAINING_FIELDS.isdisjoint(exported)

    exported_model, _, _ = load_model_checkpoint(output, device="cpu")
    for expected, actual in zip(source_model.parameters(), exported_model.parameters()):
        assert torch.equal(expected, actual)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    source_model.eval()
    assert torch.allclose(
        source_model(input_ids)["logits"],
        exported_model(input_ids)["logits"],
        rtol=1e-6,
        atol=1e-7,
    )

    sidecar = output.with_suffix(output.suffix + ".sha256")
    expected_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    assert sidecar.read_text(encoding="utf-8") == f"{expected_hash}  {output.name}\n"


def test_checkpoint_v2_exports_without_training_state(tmp_path):
    config = tiny_config()
    model = MiniLLM(config)
    optimizer = create_optimizer(model, 1e-3, 0.0)
    source = tmp_path / "v2.pt"
    save_checkpoint(
        model,
        optimizer,
        scheduler=None,
        step=7,
        path=str(source),
        config=config,
        dataloader_state={"batches_consumed": 2},
        training_config={"private_path": "must-not-be-exported"},
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    output = tmp_path / "model_inference.pt"

    result = export_inference_checkpoint(
        source,
        output,
        training_stage="instruction_sft",
        tokenizer_path=tokenizer,
    )

    assert result["max_abs_logit_diff"] == 0.0
    assert_export_matches(model, output, source_step=7)
    exported = torch.load(output, map_location="cpu", weights_only=False)
    assert exported["tokenizer"]["filename"] == "tokenizer.json"
    assert len(exported["tokenizer"]["sha256"]) == 64
    assert "private_path" not in str(exported)


def test_legacy_checkpoint_exports_when_config_is_embedded(tmp_path):
    config = tiny_config()
    model = MiniLLM(config)
    source = tmp_path / "legacy.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict(), "step": 3},
        source,
    )
    output = tmp_path / "legacy_inference.pt"

    export_inference_checkpoint(source, output, training_stage="base")

    assert_export_matches(model, output, source_step=3)


def test_export_refuses_overwrite_and_missing_config(tmp_path):
    config = tiny_config()
    model = MiniLLM(config)
    source = tmp_path / "legacy.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict()}, source
    )
    output = tmp_path / "inference.pt"
    export_inference_checkpoint(source, output, training_stage="base")

    with pytest.raises(FileExistsError, match="--overwrite"):
        export_inference_checkpoint(source, output, training_stage="base")

    no_config = tmp_path / "no_config.pt"
    torch.save(model.state_dict(), no_config)
    with pytest.raises(ValueError, match="cannot be inferred safely"):
        export_inference_checkpoint(
            no_config, tmp_path / "invalid.pt", training_stage="unknown"
        )
