import pytest
import torch

from demo_compare import (
    build_prompt,
    load_demo_model,
    missing_checkpoint_message,
    parse_models,
    resolve_device,
)
from model.config import MiniLLMConfig
from model.gpt import MiniLLM


def test_demo_builds_current_plain_text_templates():
    assert build_prompt("continuation", "A story.") == (
        "Instruction: Continue the story.\nInput: A story.\nResponse:"
    )
    qa = build_prompt("qa", "Lily ran.", question="Who ran?")
    assert "Question: Who ran?" in qa
    assert qa.endswith("Response:")
    keyword = build_prompt("keywords", "A story.", keywords=["bird", "home"])
    assert '"bird"' in keyword and '"home"' in keyword
    sentence = build_prompt("sentence_count", "A story.", sentence_count=2)
    assert "exactly 2 sentences" in sentence


def test_demo_rejects_missing_task_arguments():
    with pytest.raises(ValueError):
        build_prompt("qa", "A story.")
    with pytest.raises(ValueError):
        build_prompt("keywords", "A story.")
    with pytest.raises(ValueError):
        build_prompt("sentence_count", "A story.", sentence_count=0)


def test_demo_parses_named_models():
    models = parse_models(["A=one.pt", "B=two.pt"])
    assert list(models.items()) == [("A", "one.pt"), ("B", "two.pt")]


def test_demo_loads_inference_checkpoint_and_reports_metadata(tmp_path):
    config = MiniLLMConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=8,
        intermediate_size=16,
        dropout=0.0,
    )
    expected = MiniLLM(config)
    path = tmp_path / "inference.pt"
    torch.save(
        {
            "checkpoint_format": "minillm-inference",
            "checkpoint_version": 1,
            "model_state": expected.state_dict(),
            "model_config": config.to_dict(),
            "training_stage": "instruction_sft",
        },
        path,
    )

    actual, _, info = load_demo_model(str(path), MiniLLMConfig(), "cpu")

    assert info["format"] == "minillm-inference-v1"
    assert info["training_stage"] == "instruction_sft"
    assert info["device"] == "cpu"
    assert len(info["sha256"]) == 64
    assert all(
        torch.equal(left, right)
        for left, right in zip(expected.parameters(), actual.parameters())
    )

    legacy_path = tmp_path / "legacy_training.pt"
    torch.save({"model_state_dict": expected.state_dict(), "step": 4}, legacy_path)
    legacy_model, _, legacy_info = load_demo_model(
        str(legacy_path), config, "cpu"
    )
    assert legacy_info["format"] == "legacy-training"
    assert legacy_info["training_stage"] == "unknown"
    assert all(
        torch.equal(left, right)
        for left, right in zip(expected.parameters(), legacy_model.parameters())
    )


def test_demo_reports_all_missing_checkpoints(tmp_path):
    models = parse_models(
        [f"SFT={tmp_path / 'sft.pt'}", f"DPO={tmp_path / 'dpo.pt'}"]
    )
    message = missing_checkpoint_message(models)
    assert "SFT" in message and "DPO" in message
    assert "Train the corresponding models" in message
    assert "GitHub Release" in message


def test_demo_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="CUDA was requested"):
        resolve_device("cuda")
