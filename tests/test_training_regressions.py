"""Regression tests for training configuration and initialization bugs."""

import json
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import MiniLLMConfig
from model.generation import generate_batch
from model.gpt import MiniLLM
from train.common import create_scheduler
from train.dpo import DPODataset


class FakeTokenizer:
    def bos_id(self):
        return 1

    def eos_id(self):
        return 2

    def encode(self, text, add_special_tokens=False):
        return [10 + ord(char) for char in text]


def test_dpo_pairs_use_shifted_response_targets(tmp_path):
    path = tmp_path / "dpo.jsonl"
    path.write_text(
        json.dumps({"prompt": "A", "chosen": "BC", "rejected": "DE"}) + "\n",
        encoding="utf-8",
    )
    dataset = DPODataset(str(path), FakeTokenizer(), max_length=16)
    input_ids, labels = dataset._tokenize_pair("A", "BC")

    prompt_ids = FakeTokenizer().encode("A ", add_special_tokens=False)
    response_ids = FakeTokenizer().encode("BC", add_special_tokens=False)
    assert labels[: len(prompt_ids)].tolist() == [-100] * len(prompt_ids)
    assert labels[len(prompt_ids) :].tolist() == response_ids + [2]
    assert input_ids.tolist() == [1] + prompt_ids + response_ids


def test_cosine_scheduler_ends_at_requested_min_lr():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=3e-4)
    scheduler = create_scheduler(
        optimizer, max_steps=10, warmup_steps=0, max_lr=3e-4, min_lr=3e-5
    )
    lr_multiplier = scheduler.lr_lambdas[0]

    assert lr_multiplier(0) == pytest.approx(1.0)
    assert lr_multiplier(5) == pytest.approx(0.55)
    assert lr_multiplier(10) == pytest.approx(0.1)
    assert 3e-4 * lr_multiplier(10) == pytest.approx(3e-5)


def test_residual_projections_receive_scaled_initialization():
    torch.manual_seed(7)
    config = MiniLLMConfig(
        vocab_size=128,
        block_size=16,
        n_layer=4,
        n_head=4,
        n_embd=128,
        intermediate_size=256,
        dropout=0.0,
    )
    model = MiniLLM(config)
    q_std = model.layers[0].attn.q_proj.weight.std().item()
    o_std = model.layers[0].attn.o_proj.weight.std().item()
    down_std = model.layers[0].mlp.down_proj.weight.std().item()
    expected_residual_std = 0.02 / math.sqrt(2 * config.n_layer)

    assert model.layers[0].attn.o_proj._is_residual is True
    assert model.layers[0].mlp.down_proj._is_residual is True
    assert q_std == pytest.approx(0.02, rel=0.05)
    assert o_std == pytest.approx(expected_residual_std, rel=0.05)
    assert down_std == pytest.approx(expected_residual_std, rel=0.05)


def test_fixed_architecture_options_fail_loudly():
    with pytest.raises(ValueError, match="activation"):
        MiniLLMConfig(activation="gelu")


def test_checkpoint_config_preserves_architecture_fields():
    config = MiniLLMConfig(description="test config")
    saved = config.to_dict()

    assert saved["description"] == "test config"
    assert saved["norm_type"] == "rmsnorm"
    assert saved["activation"] == "swiglu"
    assert saved["pos_embedding"] == "rope"


def test_generate_batch_does_not_leak_padding_or_prompt_tokens():
    class GenerationTokenizer:
        def encode_batch(self, prompts):
            return [[10], [20, 21, 22]]

        def eos_id(self):
            return 2

        def decode(self, ids, skip_special_tokens=True):
            return ",".join(str(token_id) for token_id in ids)

    class EosModel:
        def __init__(self):
            self.config = type("Config", (), {"block_size": 16})()

        def eval(self):
            return self

        def __call__(self, input_ids):
            batch, length = input_ids.shape
            logits = torch.full((batch, length, 8), -100.0)
            logits[:, :, 2] = 100.0
            return {"logits": logits}

    results = generate_batch(
        EosModel(),
        GenerationTokenizer(),
        ["short", "long"],
        max_new_tokens=1,
        device="cpu",
    )

    assert results == ["2", "2"]
