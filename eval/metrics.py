"""Text generation evaluation metrics for MiniLLM."""

import math
from collections import Counter
from typing import List, Optional


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss: exp(loss)."""
    return math.exp(loss)


def compute_keyword_coverage(texts: List[str], keyword_lists: List[List[str]]) -> float:
    """Average fraction of required keywords present in each text.

    Args:
        texts: List of generated text strings.
        keyword_lists: Parallel list where keyword_lists[i] is the list of
            required keywords for texts[i].

    Returns:
        Average keyword coverage across all texts (0.0 to 1.0).
    """
    if not texts:
        return 0.0

    coverages = []
    for text, keywords in zip(texts, keyword_lists):
        if not keywords:
            coverages.append(1.0)
            continue
        text_lower = text.lower()
        found = sum(1 for kw in keywords if kw.lower() in text_lower)
        coverages.append(found / len(keywords))

    return sum(coverages) / len(coverages)


def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Extract n-grams from a list of tokens."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_repetition_rate(texts: List[str], ngram: int = 3) -> float:
    """Average ratio of repeated n-grams to total n-grams.

    For each text, computes (total_ngrams - unique_ngrams) / total_ngrams.
    Returns 0.0 for texts shorter than ngram tokens.

    Args:
        texts: List of generated text strings.
        ngram: N-gram order for repetition detection.

    Returns:
        Average repetition rate (0.0 = no repetition, 1.0 = all repeated).
    """
    if not texts:
        return 0.0

    rates = []
    for text in texts:
        tokens = text.split()
        ngrams = _get_ngrams(tokens, ngram)
        if not ngrams:
            rates.append(0.0)
            continue
        counts = Counter(ngrams)
        total = len(ngrams)
        unique = len(counts)
        repeated = total - unique
        rates.append(repeated / total)

    return sum(rates) / len(rates)


def compute_distinct_n(texts: List[str], n: int = 1) -> float:
    """Unique n-grams divided by total n-grams.

    Higher values indicate more diverse text.

    Args:
        texts: List of generated text strings.
        n: N-gram order.

    Returns:
        Average distinct-n ratio (0.0 to 1.0).
    """
    if not texts:
        return 0.0

    ratios = []
    for text in texts:
        tokens = text.split()
        ngrams = _get_ngrams(tokens, n)
        if not ngrams:
            ratios.append(0.0)
            continue
        ratios.append(len(set(ngrams)) / len(ngrams))

    return sum(ratios) / len(ratios)


def compute_avg_length(texts: List[str]) -> float:
    """Average word count across texts.

    Args:
        texts: List of generated text strings.

    Returns:
        Average number of words per text.
    """
    if not texts:
        return 0.0
    return sum(len(text.split()) for text in texts) / len(texts)


def compute_sentence_end_rate(texts: List[str]) -> float:
    """Fraction of texts ending with a sentence-ending punctuation mark (. ! ?).

    Args:
        texts: List of generated text strings.

    Returns:
        Fraction ending with proper sentence punctuation.
    """
    if not texts:
        return 0.0
    endings = (".", "!", "?")
    ended = sum(1 for text in texts if text.rstrip().endswith(endings))
    return ended / len(texts)


def compute_all_metrics(
    texts: List[str],
    loss: Optional[float] = None,
    keyword_lists: Optional[List[List[str]]] = None,
) -> dict:
    """Compute all evaluation metrics and return a dictionary.

    Args:
        texts: List of generated text strings.
        loss: Optional cross-entropy loss for perplexity computation.
        keyword_lists: Optional per-text keyword lists for coverage.

    Returns:
        Dictionary with all computed metrics.
    """
    metrics = {}

    if loss is not None:
        metrics["perplexity"] = compute_perplexity(loss)

    metrics["distinct_1"] = compute_distinct_n(texts, n=1)
    metrics["distinct_2"] = compute_distinct_n(texts, n=2)
    metrics["distinct_3"] = compute_distinct_n(texts, n=3)
    metrics["repetition_3"] = compute_repetition_rate(texts, ngram=3)
    metrics["avg_length"] = compute_avg_length(texts)
    metrics["sentence_end_rate"] = compute_sentence_end_rate(texts)

    if keyword_lists is not None:
        metrics["keyword_coverage"] = compute_keyword_coverage(texts, keyword_lists)

    return metrics
