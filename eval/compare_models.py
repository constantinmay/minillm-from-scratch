"""Compare multiple trained MiniLLM checkpoints (Base / SFT / DPO / RSFT)."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from model.config import MiniLLMConfig
from model.gpt import MiniLLM
from tokenizer.tokenizer_utils import MiniLLMTokenizer
from eval.eval_loss import evaluate_loss
from eval.eval_generation import evaluate_generation, save_generation_samples


def _load_model(checkpoint_path, config, device):
    """Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        config: MiniLLMConfig instance.
        device: Target device.

    Returns:
        Loaded MiniLLM model on device.
    """
    model = MiniLLM(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Support both raw state_dict and wrapped {"model": ...} checkpoints
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def _load_prompts(prompts_path):
    """Load evaluation prompts from a text file (one per line) or JSON list.

    Args:
        prompts_path: Path to prompts file.

    Returns:
        List of prompt strings.
    """
    with open(prompts_path, "r", encoding="utf-8") as f:
        if prompts_path.endswith(".json"):
            return json.load(f)
        return [line.strip() for line in f if line.strip()]


def compare_models(
    model_paths_dict,
    tokenizer_path,
    config_path,
    prompts_path,
    output_dir="results/",
    device="cuda",
    max_new_tokens=150,
    temperature=0.8,
    top_k=40,
    eval_dataloader=None,
    max_eval_batches=None,
):
    """Compare multiple model checkpoints (Base/SFT/DPO/RSFT).

    For each model: compute loss (if dataloader provided), generate texts,
    and compute all metrics. Save comparison table and generation samples.

    Args:
        model_paths_dict: Mapping of display name to checkpoint path,
            e.g. {"Base": "checkpoints/base.pt", "SFT": "checkpoints/sft.pt"}.
        tokenizer_path: Path to the tokenizer JSON file.
        config_path: Path to the model config YAML file.
        prompts_path: Path to prompts file (one prompt per line, or JSON list).
        output_dir: Directory to write results into.
        device: Device string for inference.
        max_new_tokens: Max tokens to generate per prompt.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        eval_dataloader: Optional DataLoader for loss evaluation.
        max_eval_batches: Optional cap on evaluation batches.

    Returns:
        Dictionary mapping model name -> results dict (with 'loss_metrics',
        'generation_metrics', 'texts').
    """
    config = MiniLLMConfig.from_yaml(config_path)
    tokenizer = MiniLLMTokenizer(tokenizer_path)
    prompts = _load_prompts(prompts_path)

    tables_dir = os.path.join(output_dir, "tables")
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    all_results = {}

    for name, ckpt_path in model_paths_dict.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({ckpt_path})")
        print(f"{'='*60}")

        model = _load_model(ckpt_path, config, device)

        entry = {}

        # --- Loss evaluation (optional) ---
        if eval_dataloader is not None:
            loss_metrics = evaluate_loss(
                model, eval_dataloader, device=device, max_batches=max_eval_batches
            )
            entry["loss_metrics"] = loss_metrics
            print(f"  Loss: {loss_metrics['loss']:.4f}  PPL: {loss_metrics['perplexity']:.2f}")

        # --- Generation evaluation ---
        gen_result = evaluate_generation(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        entry["generation_metrics"] = gen_result["metrics"]
        entry["texts"] = gen_result["texts"]

        print(f"  Avg length:    {gen_result['metrics']['avg_length']:.1f}")
        print(f"  Distinct-1:    {gen_result['metrics']['distinct_1']:.4f}")
        print(f"  Repetition-3:  {gen_result['metrics']['repetition_3']:.4f}")
        print(f"  Sent-end rate: {gen_result['metrics']['sentence_end_rate']:.4f}")

        # Save generation samples
        sample_path = os.path.join(samples_dir, f"{name.lower()}_samples.md")
        save_generation_samples(prompts, gen_result["texts"], sample_path)

        all_results[name] = entry

    # Build and save comparison table
    comparison = {}
    for name, entry in all_results.items():
        row = {}
        if "loss_metrics" in entry:
            row["loss"] = entry["loss_metrics"]["loss"]
            row["perplexity"] = entry["loss_metrics"]["perplexity"]
        row.update(entry["generation_metrics"])
        comparison[name] = row

    table_path = os.path.join(tables_dir, "model_comparison.json")
    with open(table_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\nComparison table saved to {table_path}")
    print(f"Generation samples saved to {samples_dir}/")

    return all_results
