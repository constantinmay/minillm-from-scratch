"""Tests for DPO (Direct Preference Optimization) loss computation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DPO loss pure functions (duplicated for test isolation)
# ---------------------------------------------------------------------------

def compute_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute mean log-probability of target tokens given logits.

    Args:
        logits: (B, T, V) raw logits from the model.
        targets: (B, T) target token IDs.

    Returns:
        (B,) tensor of mean log-probabilities per sequence.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log-probs for the target tokens
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # Average over sequence length
    return target_log_probs.mean(dim=-1)


def dpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    ref_chosen_logprobs: torch.Tensor,
    ref_rejected_logprobs: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute DPO loss.

    Loss = -mean(log(sigmoid(beta * (log_pi_chosen - log_ref_chosen
                                     - log_pi_rejected + log_ref_rejected))))

    Args:
        policy_chosen_logprobs: (B,) mean log-probs of chosen under policy.
        policy_rejected_logprobs: (B,) mean log-probs of rejected under policy.
        ref_chosen_logprobs: (B,) mean log-probs of chosen under reference.
        ref_rejected_logprobs: (B,) mean log-probs of rejected under reference.
        beta: DPO temperature parameter.

    Returns:
        Scalar loss tensor.
    """
    chosen_rewards = beta * (policy_chosen_logprobs - ref_chosen_logprobs)
    rejected_rewards = beta * (policy_rejected_logprobs - ref_rejected_logprobs)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


# ---------------------------------------------------------------------------
# Tiny model helper for logprob tests
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Minimal 1-layer model for testing logprob computation."""

    def __init__(self, vocab_size=20, embed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDPOLoss:

    def test_dpo_loss_chosen_preferred(self):
        """When chosen has higher logprob than rejected, loss should be small."""
        # Policy prefers chosen; reference is neutral
        policy_chosen = torch.tensor([-0.5, -0.3])
        policy_rejected = torch.tensor([-2.0, -1.5])
        ref_chosen = torch.tensor([-1.0, -1.0])
        ref_rejected = torch.tensor([-1.0, -1.0])

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0)

        assert loss.item() < 0.5, (
            f"Loss should be small when chosen is preferred, got {loss.item():.4f}"
        )

    def test_dpo_loss_reversed(self):
        """When rejected has higher logprob, loss should be larger."""
        policy_chosen = torch.tensor([-2.0, -1.5])
        policy_rejected = torch.tensor([-0.5, -0.3])
        ref_chosen = torch.tensor([-1.0, -1.0])
        ref_rejected = torch.tensor([-1.0, -1.0])

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0)

        assert loss.item() > 0.5, (
            f"Loss should be large when rejected is preferred, got {loss.item():.4f}"
        )

    def test_dpo_loss_equal(self):
        """When all logprobs equal, loss should be log(2)."""
        # chosen_rewards - rejected_rewards = 0 => -log(sigmoid(0)) = log(2)
        logprobs = torch.tensor([-1.0, -1.0])
        loss = dpo_loss(logprobs, logprobs, logprobs, logprobs, beta=1.0)

        expected = math.log(2.0)
        assert abs(loss.item() - expected) < 1e-5, (
            f"Loss should be log(2)={expected:.4f}, got {loss.item():.4f}"
        )

    def test_dpo_loss_beta_scaling(self):
        """Larger beta amplifies margin effect."""
        policy_chosen = torch.tensor([-0.5])
        policy_rejected = torch.tensor([-2.0])
        ref_chosen = torch.tensor([-1.0])
        ref_rejected = torch.tensor([-1.0])

        loss_low_beta = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )
        loss_high_beta = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=5.0
        )

        # Higher beta => margin amplified => chosen even more preferred => lower loss
        assert loss_high_beta.item() < loss_low_beta.item(), (
            f"Higher beta should decrease loss when chosen is preferred. "
            f"Low beta loss={loss_low_beta.item():.4f}, high beta loss={loss_high_beta.item():.4f}"
        )

    def test_compute_logprobs(self):
        """Test logprob computation with a small model."""
        torch.manual_seed(42)
        vocab_size = 20
        model = TinyModel(vocab_size=vocab_size, embed_dim=16)

        input_ids = torch.randint(0, vocab_size, (2, 8))
        logits = model(input_ids)

        # Compute logprobs manually
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        expected = target_log_probs.mean(dim=-1)

        result = compute_logprobs(logits, input_ids)

        assert torch.allclose(result, expected, atol=1e-6), (
            f"compute_logprobs result {result} does not match expected {expected}"
        )

        # Logprobs should be negative (log of probability <= 0)
        assert (result < 0).all(), "Log-probabilities should be negative"
