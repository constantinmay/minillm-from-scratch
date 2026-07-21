"""Tests for SFT label masking: prompt tokens should have labels=-100."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch


# ---------------------------------------------------------------------------
# Pure-function SFT template helper (duplicated for test isolation)
# ---------------------------------------------------------------------------

SFT_PROMPT_TEMPLATE = "<|prompt|>"
SFT_RESPONSE_TEMPLATE = "<|response|>"
SFT_END_TEMPLATE = "<|end|>"


def build_sft_input_and_labels(
    prompt_text: str,
    response_text: str,
    encode_fn,
    prompt_token_id: int = 0,
    response_token_id: int = 1,
    end_token_id: int = 2,
):
    """Build input_ids and labels for SFT training.

    The full sequence is: [prompt_token] + prompt_ids + [response_token] +
    response_ids + [end_token].

    Labels: -100 for the prompt portion (including prompt special token),
    real ids for the response portion.

    Args:
        prompt_text: Raw prompt string.
        response_text: Raw response string.
        encode_fn: Callable that encodes a string to list[int] of token ids.
        prompt_token_id: ID for the prompt special token.
        response_token_id: ID for the response special token.
        end_token_id: ID for the end special token.

    Returns:
        (input_ids, labels) as 1-D lists of ints.
    """
    prompt_ids = encode_fn(prompt_text)
    response_ids = encode_fn(response_text)

    input_ids = (
        [prompt_token_id]
        + prompt_ids
        + [response_token_id]
        + response_ids
        + [end_token_id]
    )

    # Prompt portion: from prompt_token through response_token (inclusive)
    prompt_len = 1 + len(prompt_ids) + 1  # prompt_tok + prompt_ids + response_tok

    labels = [-100] * prompt_len + response_ids + [end_token_id]

    assert len(input_ids) == len(labels)
    return input_ids, labels


# ---------------------------------------------------------------------------
# Simple mock encode function for testing
# ---------------------------------------------------------------------------

def _mock_encode(text: str) -> list:
    """Deterministic mock encoder: each char gets a unique id starting at 10."""
    return [ord(c) % 100 + 10 for c in text]


class TestSFTLabelMasking:

    def test_sft_label_masking(self):
        """Prompt tokens have labels=-100, response tokens have real labels."""
        prompt = "Hello"
        response = "World"
        input_ids, labels = build_sft_input_and_labels(
            prompt, response, _mock_encode
        )

        # Sanity: lengths match
        assert len(input_ids) == len(labels)

        # The prompt portion includes: [prompt_tok] + prompt_ids + [response_tok]
        # = 1 + 5 + 1 = 7 tokens with labels=-100
        prompt_len = 1 + len(_mock_encode(prompt)) + 1
        for i in range(prompt_len):
            assert labels[i] == -100, (
                f"Position {i} is in the prompt portion but has label {labels[i]} instead of -100"
            )

        # The response portion should have real token IDs
        for i in range(prompt_len, len(labels)):
            assert labels[i] != -100, (
                f"Position {i} is in the response portion but has label -100"
            )

    def test_sft_response_labels_match_input(self):
        """Response labels should match the corresponding response token IDs."""
        prompt = "What is 2+2?"
        response = "The answer is 4."
        input_ids, labels = build_sft_input_and_labels(
            prompt, response, _mock_encode
        )

        response_ids = _mock_encode(response)
        # Response labels = response_ids + [end_token_id]
        expected_response_labels = response_ids + [2]

        prompt_len = 1 + len(_mock_encode(prompt)) + 1
        actual_response_labels = labels[prompt_len:]

        assert actual_response_labels == expected_response_labels

    def test_sft_empty_prompt(self):
        """Empty prompt still masks the special tokens."""
        prompt = ""
        response = "Hi"
        input_ids, labels = build_sft_input_and_labels(
            prompt, response, _mock_encode
        )

        # Prompt portion: [prompt_tok] + [] + [response_tok] = 2 tokens
        assert labels[0] == -100
        assert labels[1] == -100
        # Remaining labels are real
        for i in range(2, len(labels)):
            assert labels[i] != -100

    def test_sft_total_length(self):
        """Total length should be prompt_special + prompt + response_special + response + end_special."""
        prompt = "A"
        response = "B C"
        input_ids, labels = build_sft_input_and_labels(
            prompt, response, _mock_encode
        )

        p_len = len(_mock_encode(prompt))
        r_len = len(_mock_encode(response))
        expected_total = 1 + p_len + 1 + r_len + 1
        assert len(input_ids) == expected_total
        assert len(labels) == expected_total
