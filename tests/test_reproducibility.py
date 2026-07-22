"""Tests for fail-fast initialization, seeds, manifests, and checkpoint v2."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.config import MiniLLMConfig
from model.gpt import MiniLLM
from train.common import (
    checkpoint_model_state,
    create_optimizer,
    create_scheduler,
    load_checkpoint,
    make_dataloader_generator,
    restore_rng_state,
    save_checkpoint,
    set_seed,
    write_run_manifest,
)
from train.dpo import DPODataset, load_dpo_checkpoint
from train.sft import SFTDataset, load_sft_initial_weights


class TinyTokenizer:
    def bos_id(self):
        return 1

    def eos_id(self):
        return 2

    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(char) % 20) for char in text]


def tiny_config():
    return MiniLLMConfig(
        vocab_size=32,
        block_size=12,
        n_layer=1,
        n_head=2,
        n_embd=8,
        intermediate_size=16,
        dropout=0.0,
    )


def test_set_seed_repeats_parameters_and_loader_order():
    set_seed(9)
    first = MiniLLM(tiny_config())
    set_seed(9)
    second = MiniLLM(tiny_config())
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))

    dataset = TensorDataset(torch.arange(20))
    order_a = torch.cat([
        batch[0] for batch in DataLoader(
            dataset, batch_size=4, shuffle=True,
            generator=make_dataloader_generator(7),
        )
    ])
    order_b = torch.cat([
        batch[0] for batch in DataLoader(
            dataset, batch_size=4, shuffle=True,
            generator=make_dataloader_generator(7),
        )
    ])
    order_c = torch.cat([
        batch[0] for batch in DataLoader(
            dataset, batch_size=4, shuffle=True,
            generator=make_dataloader_generator(8),
        )
    ])
    assert torch.equal(order_a, order_b)
    assert not torch.equal(order_a, order_c)


def test_sft_initialization_requires_checkpoint(tmp_path):
    config = tiny_config()
    model = MiniLLM(config)
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="allow_random_init"):
        load_sft_initial_weights(model, str(missing), "cpu", False, config)
    assert load_sft_initial_weights(model, str(missing), "cpu", True, config) is False

    checkpoint = tmp_path / "base.pt"
    expected = MiniLLM(config)
    torch.save({"model_state_dict": expected.state_dict(), "config": config.to_dict()}, checkpoint)
    assert load_sft_initial_weights(model, str(checkpoint), "cpu", False, config)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), expected.parameters()))


def test_dpo_initialization_requires_compatible_checkpoint(tmp_path):
    config = tiny_config()
    model = MiniLLM(config)
    with pytest.raises(FileNotFoundError, match="policy"):
        load_dpo_checkpoint(model, str(tmp_path / "missing.pt"), "cpu", config, "policy")

    incompatible = config.to_dict()
    incompatible["n_layer"] = 2
    path = tmp_path / "wrong.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": incompatible}, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_dpo_checkpoint(model, str(path), "cpu", config, "reference")


def test_datasets_fail_early_on_invalid_content(tmp_path):
    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text(json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="response"):
        SFTDataset(str(sft_path), TinyTokenizer(), max_length=12)

    dpo_path = tmp_path / "dpo.jsonl"
    dpo_path.write_text(
        json.dumps({"prompt": "p", "chosen": "same", "rejected": "same"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must differ"):
        DPODataset(str(dpo_path), TinyTokenizer(), max_length=12)


def test_checkpoint_v2_roundtrip_and_rng_restore(tmp_path):
    set_seed(21)
    config = tiny_config()
    model = MiniLLM(config)
    optimizer = create_optimizer(model, 1e-3, 0.0)
    scheduler = create_scheduler(optimizer, 2, 0, 1e-3, 1e-4)
    inputs = torch.randint(0, config.vocab_size, (1, 4))
    loss = model(inputs, targets=inputs)["loss"]
    loss.backward()
    optimizer.step()
    scheduler.step()

    path = tmp_path / "v2.pt"
    save_checkpoint(
        model, optimizer, scheduler, 1, str(path), config,
        training_config={"seed": 21}, checkpoint_type="test_training",
    )
    expected = (random.random(), np.random.random(), torch.rand(2))
    set_seed(999)
    checkpoint = load_checkpoint(str(path))
    restore_rng_state(checkpoint["rng_state"])
    actual = (random.random(), np.random.random(), torch.rand(2))

    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["step"] == 1
    assert checkpoint["optimizer_state_dict"]
    assert checkpoint["scheduler_state_dict"]
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "rng_state",
            "dataloader_state",
        }
    }
    assert str(Path.cwd().resolve()) not in json.dumps(metadata, default=str)

    legacy = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": model.state_dict(), "step": 4}, legacy)
    assert load_checkpoint(str(legacy))["step"] == 4

    minimal_v2 = tmp_path / "minimal_v2.pt"
    torch.save(
        {"checkpoint_version": 2, "model_state_dict": model.state_dict()},
        minimal_v2,
    )
    inference_checkpoint = load_checkpoint(str(minimal_v2), device="cpu")
    inference_model = MiniLLM(config)
    inference_model.load_state_dict(checkpoint_model_state(inference_checkpoint))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_checkpoint_maps_to_cpu(tmp_path):
    config = tiny_config()
    cuda_model = MiniLLM(config).cuda()
    path = tmp_path / "cuda_v2.pt"
    save_checkpoint(
        cuda_model,
        optimizer=None,
        scheduler=None,
        step=0,
        path=str(path),
        config=config,
        checkpoint_type="inference",
    )

    checkpoint = load_checkpoint(str(path), device="cpu")
    assert all(
        tensor.device.type == "cpu"
        for tensor in checkpoint_model_state(checkpoint).values()
    )
    cpu_model = MiniLLM(config)
    cpu_model.load_state_dict(checkpoint_model_state(checkpoint))


def test_checkpoint_v2_resume_matches_uninterrupted_next_step(tmp_path):
    """A CPU resume reproduces the next shuffled batch, loss, and parameters."""

    def make_training_state(seed):
        set_seed(seed)
        config = tiny_config()
        model = MiniLLM(config)
        optimizer = create_optimizer(model, 1e-3, 0.0)
        scheduler = create_scheduler(optimizer, 4, 0, 1e-3, 1e-4)
        generator = make_dataloader_generator(seed)
        inputs = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ])
        targets = torch.roll(inputs, shifts=-1, dims=1)
        indices = torch.arange(len(inputs))
        loader = DataLoader(
            TensorDataset(inputs, targets, indices),
            batch_size=2,
            shuffle=True,
            generator=generator,
        )
        return config, model, optimizer, scheduler, generator, loader

    def train_one_step(model, optimizer, scheduler, batch):
        input_ids, targets, indices = batch
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, targets=targets)["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach(), indices.clone()

    seed = 31
    config, model, optimizer, scheduler, generator, loader = make_training_state(seed)
    epoch_generator_state = generator.get_state()
    first_iterator = iter(loader)
    train_one_step(model, optimizer, scheduler, next(first_iterator))

    path = tmp_path / "resume_v2.pt"
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        1,
        str(path),
        config,
        dataloader_state={
            "epoch_generator_state": epoch_generator_state,
            "batches_consumed": 1,
        },
        training_config={"seed": seed},
        checkpoint_type="test_training",
    )

    expected_batch = next(first_iterator)
    expected_loss, expected_order = train_one_step(
        model, optimizer, scheduler, expected_batch
    )
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    (
        resumed_config,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_generator,
        resumed_loader,
    ) = make_training_state(seed=999)
    checkpoint = load_checkpoint(str(path), device="cpu")
    resumed_model.load_state_dict(checkpoint["model_state_dict"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint.get("rng_state"))
    dataloader_state = checkpoint["dataloader_state"]
    resumed_generator.set_state(dataloader_state["epoch_generator_state"])

    resumed_iterator = iter(resumed_loader)
    for _ in range(dataloader_state["batches_consumed"]):
        next(resumed_iterator)
    resumed_batch = next(resumed_iterator)
    resumed_loss, resumed_order = train_one_step(
        resumed_model, resumed_optimizer, resumed_scheduler, resumed_batch
    )

    assert resumed_config.to_dict() == config.to_dict()
    assert torch.equal(resumed_order, expected_order)
    assert torch.allclose(resumed_loss, expected_loss, rtol=1e-6, atol=1e-7)
    for actual, expected in zip(resumed_model.parameters(), expected_parameters):
        assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_run_manifest_records_relative_paths_and_checksum(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    path = write_run_manifest(
        tmp_path / "run",
        config_path="configs/train_sft.yaml",
        config={"seed": 42},
        seed=42,
        device="cpu",
        checkpoint_paths=[str(checkpoint)],
        command="python train/sft.py",
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["device"] == "cpu"
    relative_checkpoint = manifest["checkpoint_paths"][0]
    assert relative_checkpoint.endswith("model.pt")
    assert not Path(relative_checkpoint).is_absolute()
    assert len(manifest["checkpoint_sha256"][relative_checkpoint]) == 64
    assert "Users" not in path.read_text(encoding="utf-8")
