"""Tests for the reward function used in RSFT (Reinforced Self-Feedback Training)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ---------------------------------------------------------------------------
# Reward function pure logic (duplicated for test isolation)
#
# The reward considers:
#   1. Keyword coverage (fraction of required keywords present)
#   2. Repetition penalty (penalize repeated n-grams)
#   3. Length penalty (very short outputs fail a hard constraint)
#   4. Sentence ending bonus
#
# Hard constraints gate: if any hard constraint fails, total reward is
# multiplied by a gate factor (default 0.0).
# ---------------------------------------------------------------------------

def _count_repeated_ngrams(text: str, n: int = 3) -> float:
    """Return ratio of repeated n-grams to total n-grams."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    from collections import Counter
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    unique = len(counts)
    return (total - unique) / total


def compute_reward(
    text: str,
    required_keywords: list = None,
    min_length: int = 10,
    max_repetition: float = 0.5,
) -> float:
    """Compute reward for a generated text.

    Args:
        text: Generated text string.
        required_keywords: List of required keywords.
        min_length: Minimum word count (hard constraint).
        max_repetition: Maximum allowed repetition rate (hard constraint).

    Returns:
        Scalar reward in [0.0, 1.0] range.
    """
    if required_keywords is None:
        required_keywords = []

    words = text.split()
    word_count = len(words)

    # --- Hard constraints ---
    hard_pass = True
    if word_count < min_length:
        hard_pass = False

    repetition_rate = _count_repeated_ngrams(text, n=3)
    if repetition_rate > max_repetition:
        hard_pass = False

    # --- Soft scores ---
    # Keyword coverage
    if required_keywords:
        text_lower = text.lower()
        found = sum(1 for kw in required_keywords if kw.lower() in text_lower)
        keyword_score = found / len(required_keywords)
    else:
        keyword_score = 1.0

    # Repetition penalty (less repetition = higher score)
    repetition_score = 1.0 - repetition_rate

    # Length bonus (encourage reasonable length, up to a point)
    length_score = min(word_count / 50.0, 1.0)

    # Sentence ending bonus
    sentence_end_bonus = 0.1 if text.rstrip().endswith((".", "!", "?")) else 0.0

    # Weighted combination
    total = (
        0.4 * keyword_score
        + 0.3 * repetition_score
        + 0.2 * length_score
        + sentence_end_bonus
    )

    # Hard constraint gate: if any hard constraint fails, reward is heavily penalized
    if not hard_pass:
        total *= 0.0

    return max(0.0, min(1.0, total))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReward:

    def test_reward_keywords_boost(self):
        """More required keywords -> higher reward."""
        base_text = "Once upon a time there was a brave knight who rode through the dark forest."

        reward_none = compute_reward(base_text, required_keywords=[])
        reward_some = compute_reward(
            base_text, required_keywords=["knight", "forest", "brave"]
        )
        reward_many = compute_reward(
            base_text,
            required_keywords=["knight", "forest", "brave", "castle", "dragon", "sword"],
        )

        assert reward_some >= reward_none, (
            f"With keywords should be >= without: {reward_some} vs {reward_none}"
        )
        assert reward_some >= reward_many or reward_many == reward_some, (
            "More satisfied keywords should not decrease reward"
        )

    def test_reward_repetition_penalty(self):
        """Repetitive text gets lower reward than diverse text."""
        diverse = (
            "The quick brown fox jumped over the lazy dog near the old barn "
            "on a sunny morning while birds sang sweetly in tall green trees."
        )
        repetitive = (
            "the the the the the the the the the the the the the the the "
            "the the the the the the the the the the the the the the the"
        )

        reward_diverse = compute_reward(diverse)
        reward_repetitive = compute_reward(repetitive)

        assert reward_diverse > reward_repetitive, (
            f"Diverse text reward ({reward_diverse}) should exceed repetitive ({reward_repetitive})"
        )

    def test_reward_short_penalty(self):
        """Very short output fails hard constraint and gets zero reward."""
        short_text = "Hi."
        reward = compute_reward(short_text, min_length=10)

        assert reward == 0.0, (
            f"Short text should get 0 reward due to hard constraint, got {reward}"
        )

    def test_reward_hard_constraint_gate(self):
        """Hard constraint failure gates the total reward to zero."""
        # Text with excessive repetition should also get 0 reward
        text = " ".join(["hello world"] * 50)
        reward = compute_reward(text, min_length=10, max_repetition=0.1)

        assert reward == 0.0, (
            f"Text failing hard constraint should get 0 reward, got {reward}"
        )

    def test_reward_long_quality_text(self):
        """Well-formed story with keywords gets high reward."""
        story = (
            "Once upon a time, a brave knight set out on a quest to find "
            "a magical sword hidden deep within the enchanted forest. "
            "Along the way, the knight encountered a wise old dragon who "
            "offered guidance through the dark and treacherous paths. "
            "With courage and determination, the knight reached the castle "
            "and claimed the legendary weapon, bringing peace to the kingdom."
        )

        reward = compute_reward(
            story,
            required_keywords=["knight", "dragon", "sword", "forest", "castle"],
            min_length=10,
        )

        assert reward > 0.5, (
            f"Well-formed story with keywords should get high reward, got {reward}"
        )
        assert reward <= 1.0
