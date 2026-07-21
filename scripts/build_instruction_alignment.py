"""Generate one candidate pool and export matched Instruction-DPO/RSFT data.

Each instruction task has an objectively auditable reward.  DPO receives the
highest- versus lowest-reward distinct candidate; RSFT receives the same
highest-reward candidate only when it satisfies the task's hard constraint.
Generation is append-only and supports --resume for time-limited GPU jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from model.config import MiniLLMConfig
from model.generation import generate
from model.gpt import MiniLLM
from scripts.build_instruction_sft import split_sentences
from tokenizer.tokenizer_utils import MiniLLMTokenizer
from train.common import get_device, load_checkpoint, load_config


TASKS = (
    "continuation",
    "keyword_story",
    "sentence_count",
    "question_answering",
)


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalized_tokens(prediction)
    target = normalized_tokens(answer)
    if not predicted or not target:
        return float(predicted == target)
    common = Counter(predicted) & Counter(target)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def surface_quality(text: str) -> tuple[float, dict]:
    words = normalized_tokens(text)
    word_count = len(words)
    unique_ratio = len(set(words)) / word_count if word_count else 0.0
    ends_properly = bool(text.strip()) and text.strip()[-1] in ".!?"
    if word_count >= 3:
        trigrams = [tuple(words[i : i + 3]) for i in range(word_count - 2)]
        repeat_ratio = 1.0 - len(set(trigrams)) / len(trigrams)
    else:
        repeat_ratio = 0.0
    length_score = min(word_count / 12.0, 1.0) if word_count else 0.0
    score = (
        0.30 * float(ends_properly)
        + 0.25 * unique_ratio
        + 0.25 * (1.0 - repeat_ratio)
        + 0.20 * length_score
    )
    return score, {
        "word_count": word_count,
        "unique_ratio": round(unique_ratio, 4),
        "repeat_3gram_ratio": round(repeat_ratio, 4),
        "ends_properly": ends_properly,
    }


def score_instruction_response(record: dict, response: str) -> dict:
    """Return a [0, 1] task reward plus a strict success flag."""
    task = record["task_type"]
    response = response.strip()
    quality, details = surface_quality(response)
    hard_pass = False

    if task == "question_answering":
        answer = str(record.get("answer", record.get("response", "")))
        exact = normalized_tokens(response) == normalized_tokens(answer)
        f1 = token_f1(response, answer)
        concise = float(0 < len(normalized_tokens(response)) <= 5)
        reward = 0.75 * float(exact) + 0.20 * f1 + 0.05 * concise
        hard_pass = exact
        details.update({"exact_match": exact, "token_f1": round(f1, 4)})

    elif task == "sentence_count":
        expected = int(record["required_sentence_count"])
        actual = len(split_sentences(response)) if response else 0
        distance_score = max(0.0, 1.0 - abs(actual - expected) / max(expected, 1))
        hard_pass = actual == expected and bool(response)
        reward = 0.70 * float(hard_pass) + 0.15 * distance_score + 0.15 * quality
        details.update({"expected_sentences": expected, "actual_sentences": actual})

    elif task == "keyword_story":
        required = [str(word).lower() for word in record.get("required_words", [])]
        lowered = response.lower()
        covered = [
            word
            for word in required
            if re.search(rf"\b{re.escape(word)}\b", lowered)
        ]
        coverage = len(covered) / len(required) if required else 1.0
        hard_pass = bool(required) and len(covered) == len(required)
        reward = 0.75 * coverage + 0.25 * quality
        details.update(
            {
                "required_words": required,
                "covered_words": covered,
                "keyword_coverage": round(coverage, 4),
            }
        )

    elif task == "continuation":
        hard_pass = (
            details["word_count"] >= 8
            and details["ends_properly"]
            and details["repeat_3gram_ratio"] < 0.30
        )
        reward = quality

    else:
        raise ValueError(f"Unsupported task type: {task}")

    return {
        "reward": round(max(0.0, min(float(reward), 1.0)), 4),
        "hard_pass": bool(hard_pass),
        "details": details,
    }


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def select_balanced_records(
    records: Iterable[dict], total: int, valid_fraction: float, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    by_task = {task: [] for task in TASKS}
    used_source_groups: set[str] = set()
    shuffled = list(records)
    rng.shuffle(shuffled)
    for record in shuffled:
        task = record.get("task_type")
        source_group = record.get("source_group", record.get("source_id", record["id"]))
        if task not in by_task or source_group in used_source_groups:
            continue
        by_task[task].append(record)
        used_source_groups.add(source_group)

    base = total // len(TASKS)
    remainder = total % len(TASKS)
    selected = []
    for task_index, task in enumerate(TASKS):
        count = base + int(task_index < remainder)
        task_records = by_task[task][:count]
        if len(task_records) != count:
            raise RuntimeError(f"Only {len(task_records)}/{count} unique {task} prompts")
        valid_count = max(1, round(count * valid_fraction))
        for index, record in enumerate(task_records):
            copy = dict(record)
            copy["alignment_split"] = "valid" if index < valid_count else "train"
            selected.append(copy)
    rng.shuffle(selected)
    return selected


def load_completed(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    records = load_jsonl(path)
    return records, {record["id"] for record in records}


@torch.no_grad()
def generate_candidate_pool(
    records: list[dict],
    output_path: Path,
    model_path: str,
    model_config_path: str,
    tokenizer_path: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    resume: bool,
    device: str | None,
) -> list[dict]:
    device = device or get_device()
    completed_records, completed_ids = load_completed(output_path) if resume else ([], set())
    mode = "a" if resume and output_path.exists() else "w"
    pending = [record for record in records if record["id"] not in completed_ids]
    print(
        f"Using {device}; {len(completed_ids)} complete, "
        f"{len(pending)} prompts pending, k={k}"
    )

    config = MiniLLMConfig.from_yaml(model_config_path)
    tokenizer = MiniLLMTokenizer(tokenizer_path)
    model = MiniLLM(config).to(device)
    checkpoint = load_checkpoint(model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for record in tqdm(pending, desc="Instruction candidates"):
            prompt_ids = [tokenizer.bos_id()] + tokenizer.encode(
                record["prompt"] + " ", add_special_tokens=False
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            candidates = []
            seen: set[str] = set()
            for _ in range(k):
                output = generate(
                    model,
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=tokenizer.eos_id(),
                    do_sample=True,
                )
                generated_ids = output[0, len(prompt_ids) :].tolist()
                response = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
                normalized = " ".join(normalized_tokens(response))
                if normalized in seen:
                    continue
                seen.add(normalized)
                score = score_instruction_response(record, response)
                candidates.append({"response": response, **score})
            if not candidates:
                candidates.append(
                    {
                        "response": "",
                        **score_instruction_response(record, ""),
                    }
                )
            result = {
                "id": record["id"],
                "source_id": record.get("source_id"),
                "source_group": record.get("source_group"),
                "split": record["alignment_split"],
                "task_type": record["task_type"],
                "prompt": record["prompt"],
                "reference_response": record.get("response"),
                "required_words": record.get("required_words", []),
                "required_sentence_count": record.get("required_sentence_count"),
                "answer": record.get("answer"),
                "candidates": candidates,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            completed_records.append(result)
    return completed_records


def choose_pair(candidates: list[dict], min_reward_gap: float) -> tuple[dict, dict] | None:
    ordered = sorted(
        candidates,
        key=lambda candidate: (candidate["hard_pass"], candidate["reward"]),
        reverse=True,
    )
    chosen = ordered[0]
    rejected = next(
        (
            candidate
            for candidate in reversed(ordered)
            if " ".join(normalized_tokens(candidate["response"]))
            != " ".join(normalized_tokens(chosen["response"]))
        ),
        None,
    )
    if rejected is None or chosen["reward"] - rejected["reward"] < min_reward_gap:
        return None
    return chosen, rejected


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_alignment_data(
    candidate_records: list[dict], output_dir: Path, min_reward_gap: float
) -> dict:
    dpo = {"train": [], "valid": []}
    rsft = {"train": [], "valid": []}
    candidate_stats = Counter()
    for record in candidate_records:
        split = record["split"]
        task = record["task_type"]
        candidates = record["candidates"]
        candidate_stats[f"{task}:prompts"] += 1
        candidate_stats[f"{task}:hard_pass_prompts"] += int(
            any(candidate["hard_pass"] for candidate in candidates)
        )

        pair = choose_pair(candidates, min_reward_gap)
        if pair is not None:
            chosen, rejected = pair
            dpo[split].append(
                {
                    "id": record["id"],
                    "task_type": task,
                    "prompt": record["prompt"],
                    "chosen": chosen["response"],
                    "rejected": rejected["response"],
                    "chosen_reward": chosen["reward"],
                    "rejected_reward": rejected["reward"],
                    "reward_gap": round(chosen["reward"] - rejected["reward"], 4),
                }
            )

        passing = [candidate for candidate in candidates if candidate["hard_pass"]]
        if passing:
            best = max(passing, key=lambda candidate: candidate["reward"])
            rsft[split].append(
                {
                    "id": record["id"],
                    "task_type": task,
                    "prompt": record["prompt"],
                    "response": best["response"],
                    "reward": best["reward"],
                }
            )

    for split in ("train", "valid"):
        write_jsonl(output_dir / f"dpo_{split}.jsonl", dpo[split])
        write_jsonl(output_dir / f"rsft_{split}.jsonl", rsft[split])

    stats = {
        "candidate_prompts": len(candidate_records),
        "candidate_stats": dict(sorted(candidate_stats.items())),
        "min_reward_gap": min_reward_gap,
        "dpo": {
            split: {
                "total": len(dpo[split]),
                "tasks": dict(Counter(item["task_type"] for item in dpo[split])),
            }
            for split in ("train", "valid")
        },
        "rsft": {
            split: {
                "total": len(rsft[split]),
                "tasks": dict(Counter(item["task_type"] for item in rsft[split])),
            }
            for split in ("train", "valid")
        },
    }
    with (output_dir / "statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/instruction_alignment.yaml")
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    source_path = Path(cfg["source_path"])
    output_dir = Path(cfg["output_dir"])
    candidate_path = output_dir / "candidates.jsonl"
    num_prompts = args.num_prompts or int(cfg.get("num_prompts", 1000))
    k = args.k or int(cfg.get("k", 8))

    if args.export_only:
        candidate_records = load_jsonl(candidate_path)
    else:
        selected = select_balanced_records(
            load_jsonl(source_path),
            num_prompts,
            float(cfg.get("valid_fraction", 0.1)),
            int(cfg.get("seed", 42)),
        )
        candidate_records = generate_candidate_pool(
            selected,
            candidate_path,
            cfg["model_path"],
            cfg.get("model_config", "configs/model_config.yaml"),
            cfg.get("tokenizer_path", "tokenizer/minillm_tokenizer.json"),
            k,
            int(cfg.get("max_new_tokens", 80)),
            float(cfg.get("temperature", 0.8)),
            int(cfg.get("top_k", 40)),
            int(cfg.get("seed", 42)),
            args.resume,
            cfg.get("device"),
        )

    stats = export_alignment_data(
        candidate_records, output_dir, float(cfg.get("min_reward_gap", 0.1))
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Alignment data written to {output_dir}")


if __name__ == "__main__":
    main()
