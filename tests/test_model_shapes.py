"""Tests for model forward pass shapes and configuration."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import MiniLLMConfig
from model.gpt import MiniLLM


@pytest.fixture
def tiny_config():
    return MiniLLMConfig(
        name="test-tiny",
        vocab_size=1000,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128,
        intermediate_size=512,
        dropout=0.0,
        bias=False,
        weight_tying=True,
    )


@pytest.fixture
def model(tiny_config):
    return MiniLLM(tiny_config)


class TestModelShapes:

    def test_forward_logits_shape(self, model, tiny_config):
        B, T = 2, 32
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, T))
        output = model(input_ids)
        assert "logits" in output
        assert output["logits"].shape == (B, T, tiny_config.vocab_size)

    def test_forward_no_targets_no_loss(self, model, tiny_config):
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))
        output = model(input_ids)
        assert "loss" not in output

    def test_forward_with_targets_returns_loss(self, model, tiny_config):
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 32))
        output = model(input_ids, targets=input_ids)
        assert "loss" in output
        assert output["loss"].dim() == 0
        assert output["loss"].item() > 0

    def test_logits_shape_matches_vocab(self, model, tiny_config):
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 16))
        logits = model(input_ids)["logits"]
        assert logits.shape[-1] == tiny_config.vocab_size

    def test_weight_tying(self, model):
        assert model.token_embedding.weight is model.lm_head.weight

    def test_no_bias_in_linear_layers(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert module.bias is None, f"{name} has bias but should not"

    def test_parameter_count(self, model):
        n = model.num_params()
        assert n > 0
        # Tiny config with 2 layers, 128 dim, 1000 vocab ~650K params
        assert 100_000 < n < 5_000_000

    def test_gradient_flows(self, model, tiny_config):
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
        output = model(input_ids, targets=input_ids)
        output["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
