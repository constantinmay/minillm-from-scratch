"""Tests for causal masking: future tokens must not affect past logits."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import MiniLLMConfig
from model.gpt import MiniLLM


@pytest.fixture
def model():
    config = MiniLLMConfig(
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
    torch.manual_seed(42)
    return MiniLLM(config)


class TestCausalMask:

    def test_future_token_does_not_affect_past(self, model):
        """Changing the last token should not affect logits at earlier positions."""
        torch.manual_seed(0)
        input_ids = torch.randint(0, 1000, (1, 32))

        model.eval()
        with torch.no_grad():
            logits1 = model(input_ids)["logits"]

        # Modify the last token
        modified = input_ids.clone()
        modified[0, -1] = (modified[0, -1] + 1) % 1000

        with torch.no_grad():
            logits2 = model(modified)["logits"]

        # All positions except the last should be identical
        assert torch.allclose(logits1[0, :-1], logits2[0, :-1], atol=1e-5), \
            "Changing future token affected past logits — causal mask broken!"

        # The last position should differ (since its input changed)
        assert not torch.allclose(logits1[0, -1], logits2[0, -1], atol=1e-5), \
            "Logits at changed position are identical — something is wrong."

    def test_middle_token_change_only_affects_subsequent(self, model):
        """Changing a token at position t should only affect logits at positions >= t."""
        torch.manual_seed(0)
        input_ids = torch.randint(0, 1000, (1, 32))
        model.eval()

        change_pos = 15

        with torch.no_grad():
            logits1 = model(input_ids)["logits"]

        modified = input_ids.clone()
        modified[0, change_pos] = (modified[0, change_pos] + 1) % 1000

        with torch.no_grad():
            logits2 = model(modified)["logits"]

        # Positions before change_pos should be unaffected
        assert torch.allclose(logits1[0, :change_pos], logits2[0, :change_pos], atol=1e-5), \
            f"Token change at position {change_pos} affected earlier positions!"

        # Positions at and after change_pos should differ
        assert not torch.allclose(logits1[0, change_pos], logits2[0, change_pos], atol=1e-5), \
            f"Token change at position {change_pos} did not affect its own position!"
