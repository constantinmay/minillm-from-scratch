"""Tests for MiniLLM tokenizer.

These tests require a trained tokenizer file. They will be automatically
skipped if the tokenizer has not been trained yet.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tokenizer", "minillm_tokenizer.json"
)
TOKENIZER_PATH = os.path.normpath(TOKENIZER_PATH)

pytestmark = pytest.mark.skipif(
    not os.path.exists(TOKENIZER_PATH),
    reason="Tokenizer not trained yet",
)


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tokenizer once for all tests in this module."""
    from tokenizer.tokenizer_utils import MiniLLMTokenizer
    return MiniLLMTokenizer(TOKENIZER_PATH)


class TestTokenizer:

    def test_encode_decode_roundtrip(self, tokenizer):
        """Encode then decode produces non-empty output with key content preserved."""
        texts = [
            "The cat sat on the mat.",
            "A little girl ran in the park.",
            "The sun is bright today.",
        ]
        for text in texts:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            assert len(ids) > 0, f"Encode returned empty for '{text}'"
            assert len(decoded) > 0, f"Decode returned empty for '{text}'"
            # With small vocab, subword splits may occur; verify common short words survive
            for word in ["cat", "mat", "ran", "park", "sun", "day"]:
                if word in text.lower():
                    assert word in decoded.lower(), (
                        f"Key word '{word}' lost: '{text}' -> '{decoded}'"
                    )

    def test_special_tokens(self, tokenizer):
        """Verify special tokens exist with correct IDs."""
        pad_id = tokenizer.pad_id()
        unk_id = tokenizer.unk_id()
        bos_id = tokenizer.bos_id()
        eos_id = tokenizer.eos_id()

        assert isinstance(pad_id, int) and pad_id >= 0, f"Invalid pad_id: {pad_id}"
        assert isinstance(unk_id, int) and unk_id >= 0, f"Invalid unk_id: {unk_id}"
        assert isinstance(bos_id, int) and bos_id >= 0, f"Invalid bos_id: {bos_id}"
        assert isinstance(eos_id, int) and eos_id >= 0, f"Invalid eos_id: {eos_id}"

        special_ids = {pad_id, unk_id, bos_id, eos_id}
        assert len(special_ids) == 4, f"Special token IDs should be distinct: {special_ids}"

        assert tokenizer.token_to_id("<pad>") == pad_id
        assert tokenizer.token_to_id("<unk>") == unk_id
        assert tokenizer.token_to_id("<bos>") == bos_id
        assert tokenizer.token_to_id("<eos>") == eos_id

    def test_vocab_size(self, tokenizer):
        """Tokenizer has a valid vocab size (> special tokens count)."""
        vs = tokenizer.vocab_size
        assert vs >= 4, f"Vocab size should be >= 4 (special tokens), got {vs}"

    def test_batch_encode_decode(self, tokenizer):
        """Batch encode/decode works correctly."""
        texts = [
            "The cat sat on the mat.",
            "A little dog ran fast.",
            "The bird flew away.",
        ]
        batch_ids = tokenizer.encode_batch(texts)
        assert len(batch_ids) == len(texts)

        decoded_batch = tokenizer.decode_batch(batch_ids)
        assert len(decoded_batch) == len(texts)

        for original, ids, decoded in zip(texts, batch_ids, decoded_batch):
            assert isinstance(ids, list)
            assert all(isinstance(i, int) for i in ids)
            assert len(decoded) > 0, f"Decode returned empty for '{original}'"
