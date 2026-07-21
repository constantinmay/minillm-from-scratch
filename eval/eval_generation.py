"""Text generation evaluation for MiniLLM."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from model.generation import generate_batch
from eval.metrics import compute_all_metrics


def evaluate_generation(
    model,
    tokenizer,
    prompts,
    max_new_tokens=150,
    temperature=0.8,
    top_k=40,
    device="cuda",
    keyword_lists=None,
):
    """Generate texts from prompts and compute all metrics.

    Args:
        model: MiniLLM model instance.
        tokenizer: MiniLLMTokenizer instance.
        prompts: List of prompt strings.
        max_new_tokens: Maximum new tokens to generate per prompt.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        device: Device string.
        keyword_lists: Optional per-prompt keyword lists for coverage metric.

    Returns:
        Dictionary with:
            - 'texts': list of generated text strings (completions only)
            - 'metrics': dictionary of computed metrics
    """
    model.eval()

    generated_texts = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    metrics = compute_all_metrics(
        generated_texts,
        loss=None,
        keyword_lists=keyword_lists,
    )

    return {"texts": generated_texts, "metrics": metrics}


def save_generation_samples(prompts, generated_texts, save_path):
    """Save prompt-generation pairs to a markdown file.

    Args:
        prompts: List of prompt strings.
        generated_texts: Parallel list of generated text strings.
        save_path: Path to the output markdown file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# Generation Samples\n\n")
        for i, (prompt, gen) in enumerate(zip(prompts, generated_texts), 1):
            f.write(f"## Sample {i}\n\n")
            f.write(f"**Prompt:** {prompt}\n\n")
            f.write(f"**Generated:** {gen}\n\n")
            f.write("---\n\n")
