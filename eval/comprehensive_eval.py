"""Reproducible, multi-dimensional evaluation for MiniLLM checkpoints.

This module deliberately separates metrics with different meanings:

* held-out TinyStories next-token NLL/perplexity (language modeling),
* response-only SFT NLL/perplexity (continuation modeling),
* DPO preference accuracy/margin/loss (preference alignment), and
* fixed-seed generation metrics (surface quality, diversity, and efficiency).

It also exports randomized A/B pairs for human or LLM-as-a-judge review.  No
single metric is treated as an overall model score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from eval.metrics import compute_all_metrics
from model.config import MiniLLMConfig
from model.generation import generate
from model.gpt import MiniLLM
from tokenizer.tokenizer_utils import MiniLLMTokenizer
from train.dpo import DPODataset, collate_dpo, compute_logprobs, dpo_loss
from train.pretrain import PretrainDataset
from train.sft import SFTDataset, collate_sft


DEFAULT_MODELS = OrderedDict(
    [
        ("Base", "checkpoints/base.pt"),
        ("SFT", "checkpoints/sft.pt"),
        ("DPO", "checkpoints/dpo.pt"),
        ("RSFT", "checkpoints/rsft.pt"),
    ]
)


def load_prompt_records(path: str) -> List[dict]:
    """Load JSON, JSONL, or plain-text prompts into normalized records."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            data = json.load(handle)
            if isinstance(data, dict):
                data = data.get("prompts", [])
            for item in data:
                records.append(item if isinstance(item, dict) else {"prompt": str(item)})
        elif path.endswith(".jsonl"):
            for line in handle:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    records.append(item if isinstance(item, dict) else {"prompt": str(item)})
        else:
            records = [{"prompt": line.strip()} for line in handle if line.strip()]

    normalized = []
    for item in records:
        if "prompt" not in item:
            raise ValueError(f"Prompt record has no 'prompt' field: {item}")
        normalized.append(
            {
                "type": item.get("type", item.get("task_type", "unspecified")),
                "prompt": str(item["prompt"]),
                "required_words": list(item.get("required_words", [])),
                "required_sentence_count": item.get("required_sentence_count"),
                "answer": item.get("answer"),
                "reference_response": item.get("response"),
            }
        )
    return normalized


def load_model(path: str, config: MiniLLMConfig, device: str) -> MiniLLM:
    model = MiniLLM(config).to(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_lm_nll(
    model: MiniLLM,
    data_path: str,
    block_size: int,
    batch_size: int,
    device: str,
    max_batches: int | None,
) -> dict:
    dataset = PretrainDataset(data_path, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    loss_sum = 0.0
    token_count = 0
    batch_count = 0
    for batch_index, (input_ids, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        loss = model(input_ids, targets=targets)["loss"]
        tokens = targets.numel()
        loss_sum += loss.item() * tokens
        token_count += tokens
        batch_count += 1
    nll = loss_sum / token_count if token_count else float("inf")
    return {
        "nll": nll,
        "perplexity": math.exp(nll) if math.isfinite(nll) else float("inf"),
        "tokens": token_count,
        "batches": batch_count,
    }


@torch.no_grad()
def evaluate_response_nll(
    model: MiniLLM,
    dataset: SFTDataset,
    batch_size: int,
    device: str,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sft,
    )
    loss_sum = 0.0
    token_count = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        tokens = int((labels != -100).sum().item())
        loss = model(input_ids, targets=labels)["loss"]
        loss_sum += loss.item() * tokens
        token_count += tokens
    nll = loss_sum / token_count if token_count else float("inf")
    return {
        "nll": nll,
        "perplexity": math.exp(nll) if math.isfinite(nll) else float("inf"),
        "tokens": token_count,
    }


@torch.no_grad()
def evaluate_preferences(
    model: MiniLLM,
    reference_model: MiniLLM,
    dataset: DPODataset,
    batch_size: int,
    device: str,
    beta: float,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dpo,
    )
    correct = 0
    total = 0
    margins: List[torch.Tensor] = []
    losses: List[Tuple[float, int]] = []
    for chosen_ids, chosen_labels, rejected_ids, rejected_labels in loader:
        chosen_ids = chosen_ids.to(device)
        chosen_labels = chosen_labels.to(device)
        rejected_ids = rejected_ids.to(device)
        rejected_labels = rejected_labels.to(device)

        chosen_logps = compute_logprobs(model, chosen_ids, chosen_labels)
        rejected_logps = compute_logprobs(model, rejected_ids, rejected_labels)
        ref_chosen = compute_logprobs(reference_model, chosen_ids, chosen_labels)
        ref_rejected = compute_logprobs(reference_model, rejected_ids, rejected_labels)
        margin = chosen_logps - rejected_logps
        batch_n = chosen_ids.size(0)
        correct += int((margin > 0).sum().item())
        total += batch_n
        margins.append(margin.cpu())
        loss = dpo_loss(
            chosen_logps, rejected_logps, ref_chosen, ref_rejected, beta=beta
        )
        losses.append((loss.item(), batch_n))

    all_margins = torch.cat(margins) if margins else torch.empty(0)
    return {
        "accuracy": correct / total if total else 0.0,
        "mean_logprob_margin": all_margins.mean().item() if total else 0.0,
        "median_logprob_margin": all_margins.median().item() if total else 0.0,
        "dpo_loss_vs_base": (
            sum(loss * count for loss, count in losses) / total if total else float("inf")
        ),
        "pairs": total,
    }


def count_sentences(text: str) -> int:
    return sum(text.count(mark) for mark in ".!?")


def normalize_short_answer(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def required_keyword_coverage(texts, keyword_lists) -> Tuple[float | None, int]:
    """Average coverage only over prompts that actually require keywords."""
    coverages = []
    for text, keywords in zip(texts, keyword_lists):
        if not keywords:
            continue
        text_lower = text.lower()
        coverages.append(
            sum(keyword.lower() in text_lower for keyword in keywords) / len(keywords)
        )
    return (sum(coverages) / len(coverages) if coverages else None, len(coverages))


@torch.no_grad()
def evaluate_generation_fixed_seed(
    model: MiniLLM,
    tokenizer: MiniLLMTokenizer,
    prompt_records: Sequence[dict],
    seeds: Sequence[int],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> Tuple[dict, List[dict]]:
    texts: List[str] = []
    keyword_lists: List[List[str]] = []
    samples: List[dict] = []
    eos_count = 0
    generated_token_count = 0
    exact_three_total = 0
    exact_three_correct = 0
    exact_sentence_total = 0
    exact_sentence_correct = 0
    qa_total = 0
    qa_exact = 0
    keyword_full_total = 0
    keyword_full_correct = 0
    start = time.perf_counter()

    for seed in seeds:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        for record in prompt_records:
            prompt_ids = [tokenizer.bos_id()] + tokenizer.encode(
                record["prompt"] + " ", add_special_tokens=False
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            output = generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=tokenizer.eos_id(),
                do_sample=True,
            )
            continuation_ids = output[0, len(prompt_ids) :].tolist()
            if continuation_ids and continuation_ids[-1] == tokenizer.eos_id():
                eos_count += 1
            generated_token_count += len(continuation_ids)
            text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            texts.append(text)
            keyword_lists.append(record["required_words"])
            if record["type"] == "format_control":
                exact_three_total += 1
                exact_three_correct += int(count_sentences(text) == 3)
            expected_sentences = record.get("required_sentence_count")
            if expected_sentences is not None:
                exact_sentence_total += 1
                exact_sentence_correct += int(
                    count_sentences(text) == int(expected_sentences)
                )
            if record["type"] == "question_answering" and record.get("answer"):
                qa_total += 1
                qa_exact += int(
                    normalize_short_answer(text)
                    == normalize_short_answer(str(record["answer"]))
                )
            if record["required_words"]:
                keyword_full_total += 1
                lowered = text.lower()
                keyword_full_correct += int(
                    all(
                        re.search(rf"\b{re.escape(word.lower())}\b", lowered)
                        for word in record["required_words"]
                    )
                )
            samples.append(
                {
                    "seed": seed,
                    "type": record["type"],
                    "prompt": record["prompt"],
                    "required_words": record["required_words"],
                    "required_sentence_count": record.get("required_sentence_count"),
                    "answer": record.get("answer"),
                    "reference_response": record.get("reference_response"),
                    "completion": text,
                    "generated_tokens": len(continuation_ids),
                }
            )

    elapsed = time.perf_counter() - start
    metrics = compute_all_metrics(texts, keyword_lists=None)
    keyword_coverage, keyword_prompt_count = required_keyword_coverage(
        texts, keyword_lists
    )
    by_prompt_type = {}
    for prompt_type in sorted({sample["type"] for sample in samples}):
        indices = [
            index for index, sample in enumerate(samples) if sample["type"] == prompt_type
        ]
        group_texts = [texts[index] for index in indices]
        group_keywords = [keyword_lists[index] for index in indices]
        group_metrics = compute_all_metrics(group_texts, keyword_lists=None)
        group_coverage, group_keyword_count = required_keyword_coverage(
            group_texts, group_keywords
        )
        group_metrics.update(
            {
                "empty_rate": sum(not text for text in group_texts) / len(group_texts),
                "keyword_coverage": group_coverage,
                "keyword_prompt_count": group_keyword_count,
            }
        )
        by_prompt_type[prompt_type] = group_metrics
    metrics.update(
        {
            "keyword_coverage": keyword_coverage,
            "keyword_prompt_count": keyword_prompt_count,
            "eos_stop_rate": eos_count / len(texts) if texts else 0.0,
            "empty_rate": sum(not text for text in texts) / len(texts) if texts else 0.0,
            "exact_three_sentence_rate": (
                exact_three_correct / exact_three_total if exact_three_total else None
            ),
            "exact_sentence_count_rate": (
                exact_sentence_correct / exact_sentence_total
                if exact_sentence_total else None
            ),
            "qa_exact_match": qa_exact / qa_total if qa_total else None,
            "keyword_all_success_rate": (
                keyword_full_correct / keyword_full_total
                if keyword_full_total else None
            ),
            "generated_tokens": generated_token_count,
            "generation_seconds": elapsed,
            "tokens_per_second": generated_token_count / elapsed if elapsed else 0.0,
            "by_prompt_type": by_prompt_type,
        }
    )
    return metrics, samples


def export_blind_pairs(
    samples_by_model: Mapping[str, Sequence[dict]], output_dir: Path, seed: int = 2026
) -> None:
    base_samples = samples_by_model["Base"]
    rng = random.Random(seed)
    pairs = []
    keys = []
    for challenger, challenger_samples in samples_by_model.items():
        if challenger == "Base":
            continue
        for index, (base, candidate) in enumerate(zip(base_samples, challenger_samples)):
            choices = [("Base", base["completion"]), (challenger, candidate["completion"])]
            rng.shuffle(choices)
            pair_id = f"{challenger.lower()}_{index:04d}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "prompt": base["prompt"],
                    "response_a": choices[0][1],
                    "response_b": choices[1][1],
                    "rubric": [
                        "grammar",
                        "coherence",
                        "consistency",
                        "prompt_relevance",
                        "creativity",
                        "age_appropriateness",
                    ],
                }
            )
            keys.append(
                {"pair_id": pair_id, "model_a": choices[0][0], "model_b": choices[1][0]}
            )

    with open(output_dir / "blind_judge_pairs.jsonl", "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with open(output_dir / "blind_judge_key.json", "w", encoding="utf-8") as handle:
        json.dump(keys, handle, indent=2, ensure_ascii=False)


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def write_markdown_report(results: Mapping[str, dict], metadata: dict, path: Path) -> None:
    lines = [
        "# MiniLLM Comprehensive Evaluation",
        "",
        "## Protocol",
        "",
        f"- Device: `{metadata['device']}`",
        f"- Seeds: `{metadata['seeds']}`",
        f"- Prompts: `{metadata['prompt_count']}`",
        f"- Generation: temperature={metadata['temperature']}, top_k={metadata['top_k']}, max_new_tokens={metadata['max_new_tokens']}",
        f"- TinyStories LM evaluation batches: `{metadata['max_lm_batches']}`",
        "",
        "Metrics from different sections have different meanings and must not be compared as one loss.",
        "",
        "## Comparable Results",
        "",
        "| Model | TinyStories NLL ↓ | TinyStories PPL ↓ | SFT response NLL ↓ | DPO pref. acc. ↑ | DPO margin ↑ | Distinct-2 ↑ | Repeat-3 ↓ | Empty ↓ | End rate ↑ | Keyword cov. ↑ | 3-sentence ↑ | tok/s ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lm = result["language_modeling"]
        sft = result["sft_response"]
        pref = result["preference"]
        gen = result["generation"]
        lines.append(
            f"| {name} | {fmt(lm['nll'])} | {fmt(lm['perplexity'], 2)} | "
            f"{fmt(sft['nll'])} | {fmt(pref['accuracy'])} | "
            f"{fmt(pref['mean_logprob_margin'])} | {fmt(gen['distinct_2'])} | "
            f"{fmt(gen['repetition_3'])} | {fmt(gen['empty_rate'])} | "
            f"{fmt(gen['sentence_end_rate'])} | {fmt(gen['keyword_coverage'])} | "
            f"{fmt(gen['exact_three_sentence_rate'])} | {fmt(gen['tokens_per_second'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- TinyStories PPL measures retained in-domain language modeling ability.",
            "- SFT response NLL measures likelihood of held-out target continuations.",
            "- Preference accuracy measures whether chosen continuations receive higher average log-probability than rejected continuations.",
            "- Distinct-n alone is not quality: random text can be diverse. Read it together with repetition, ending, keyword coverage, and blind judgments.",
            "- Keyword coverage is averaged only over prompts with explicit required words; unconstrained prompts are excluded.",
            "- Empty rate is crucial here because the training data uses raw story continuation rather than an instruction template.",
            "- `blind_judge_pairs.jsonl` is randomized; evaluators must not read `blind_judge_key.json` before scoring.",
            "",
            "## Generation Breakdown by Prompt Type",
            "",
            "| Model | Type | Samples | Avg words | Empty ↓ | End rate ↑ | Distinct-2 ↑ | Keyword cov. ↑ |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in results.items():
        for prompt_type, group in result["generation"]["by_prompt_type"].items():
            lines.append(
                f"| {name} | {prompt_type} | "
                f"{len(metadata['seeds']) * metadata['prompt_type_counts'][prompt_type]} | "
                f"{fmt(group['avg_length'], 1)} | {fmt(group['empty_rate'])} | "
                f"{fmt(group['sentence_end_rate'])} | {fmt(group['distinct_2'])} | "
                f"{fmt(group['keyword_coverage'])} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_summary(results: Mapping[str, dict], path: Path) -> None:
    rows = []
    for name, result in results.items():
        row = {"model": name}
        for section, values in result.items():
            if section == "samples":
                continue
            for key, value in values.items():
                row[f"{section}.{key}"] = value
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    fieldnames.remove("model")
    fieldnames.insert(0, "model")
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_models(values: Sequence[str] | None) -> OrderedDict:
    if not values:
        return DEFAULT_MODELS.copy()
    models = OrderedDict()
    for value in values:
        if "=" not in value:
            raise ValueError("--model must use NAME=CHECKPOINT format")
        name, path = value.split("=", 1)
        models[name] = path
    if "Base" not in models:
        raise ValueError("A model named 'Base' is required as the DPO reference")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="NAME=CHECKPOINT; repeatable")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/minillm_tokenizer.json")
    parser.add_argument("--lm-valid", default="data/processed/valid.bin")
    parser.add_argument("--sft-valid", default="data/instruction_sft/valid.jsonl")
    parser.add_argument("--dpo-valid", default="data/dpo_v2/valid.jsonl")
    parser.add_argument("--prompts", default="data/prompts/eval_prompts.jsonl")
    parser.add_argument("--output", default="results/comprehensive_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-lm-batches", type=int, default=100)
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--max-prompts-per-type",
        type=int,
        default=0,
        help="Evaluate at most N prompts per type; 0 uses all prompts.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--beta", type=float, default=0.05)
    args = parser.parse_args()

    models = parse_models(args.model)
    for name, path in models.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MiniLLMConfig.from_yaml(args.config)
    tokenizer = MiniLLMTokenizer(args.tokenizer)
    prompt_records = load_prompt_records(args.prompts)
    if args.max_prompts_per_type > 0:
        limited = []
        type_counts = Counter()
        for record in prompt_records:
            if type_counts[record["type"]] >= args.max_prompts_per_type:
                continue
            limited.append(record)
            type_counts[record["type"]] += 1
        prompt_records = limited
    sft_dataset = SFTDataset(args.sft_valid, tokenizer, config.block_size)
    dpo_dataset = DPODataset(args.dpo_valid, tokenizer, config.block_size)

    print(f"Device: {args.device}")
    print(f"Models: {', '.join(models)}")
    print(f"Prompts: {len(prompt_records)} x {len(seeds)} seeds")

    reference_model = load_model(models["Base"], config, args.device)
    results: OrderedDict[str, dict] = OrderedDict()
    samples_by_model: OrderedDict[str, List[dict]] = OrderedDict()

    for name, checkpoint_path in models.items():
        print(f"\n{'=' * 64}\nEvaluating {name}: {checkpoint_path}\n{'=' * 64}")
        model = reference_model if name == "Base" else load_model(
            checkpoint_path, config, args.device
        )
        lm_metrics = evaluate_lm_nll(
            model,
            args.lm_valid,
            config.block_size,
            args.batch_size,
            args.device,
            args.max_lm_batches,
        )
        print(f"TinyStories NLL={lm_metrics['nll']:.4f}, PPL={lm_metrics['perplexity']:.2f}")
        sft_metrics = evaluate_response_nll(
            model, sft_dataset, args.batch_size, args.device
        )
        print(f"SFT response NLL={sft_metrics['nll']:.4f}")
        preference_metrics = evaluate_preferences(
            model,
            reference_model,
            dpo_dataset,
            max(1, args.batch_size // 2),
            args.device,
            args.beta,
        )
        print(
            f"Preference accuracy={preference_metrics['accuracy']:.3f}, "
            f"margin={preference_metrics['mean_logprob_margin']:.4f}"
        )
        generation_metrics, samples = evaluate_generation_fixed_seed(
            model,
            tokenizer,
            prompt_records,
            seeds,
            args.device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        )
        print(
            f"Distinct-2={generation_metrics['distinct_2']:.3f}, "
            f"repeat-3={generation_metrics['repetition_3']:.3f}, "
            f"end-rate={generation_metrics['sentence_end_rate']:.3f}"
        )
        results[name] = {
            "language_modeling": lm_metrics,
            "sft_response": sft_metrics,
            "preference": preference_metrics,
            "generation": generation_metrics,
        }
        samples_by_model[name] = samples
        (output_dir / f"{name.lower()}_samples.json").write_text(
            json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if name != "Base":
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    metadata = {
        "device": args.device,
        "models": models,
        "seeds": seeds,
        "prompt_count": len(prompt_records),
        "prompt_type_counts": {
            prompt_type: sum(record["type"] == prompt_type for record in prompt_records)
            for prompt_type in sorted({record["type"] for record in prompt_records})
        },
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "max_lm_batches": args.max_lm_batches,
        "lm_valid": args.lm_valid,
        "sft_valid": args.sft_valid,
        "dpo_valid": args.dpo_valid,
    }
    payload = {"metadata": metadata, "results": results}
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv_summary(results, output_dir / "evaluation_summary.csv")
    write_markdown_report(results, metadata, output_dir / "evaluation_report.md")
    export_blind_pairs(samples_by_model, output_dir)
    print(f"\nEvaluation artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
