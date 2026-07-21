"""Export conservative, objectively verifiable DPO pairs from a candidate pool.

The exporter deliberately excludes unconstrained continuation prompts.  A pair
is eligible only when its chosen response satisfies the task's hard constraint
and its rejected response fails it.  Eligible pairs are downsampled to the same
count per task before a source-disjoint train/validation split is written.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_instruction_alignment import load_jsonl, normalized_tokens, write_jsonl


STRICT_TASKS = ("question_answering", "sentence_count", "keyword_story")


def choose_strict_pair(candidates: list[dict]) -> tuple[dict, dict] | None:
    """Choose the best hard pass and worst distinct hard failure."""
    passing = sorted(
        (candidate for candidate in candidates if candidate.get("hard_pass")),
        key=lambda candidate: candidate["reward"],
        reverse=True,
    )
    failing = sorted(
        (candidate for candidate in candidates if not candidate.get("hard_pass")),
        key=lambda candidate: candidate["reward"],
    )
    for chosen in passing:
        chosen_text = " ".join(normalized_tokens(chosen["response"]))
        for rejected in failing:
            if " ".join(normalized_tokens(rejected["response"])) != chosen_text:
                return chosen, rejected
    return None


def build_strict_dpo_records(
    candidate_records: list[dict], valid_fraction: float, seed: int
) -> tuple[dict[str, list[dict]], dict]:
    """Return balanced train/valid pairs and an auditable selection summary."""
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    rng = random.Random(seed)
    eligible: dict[str, list[dict]] = {task: [] for task in STRICT_TASKS}

    for record in candidate_records:
        task = record.get("task_type")
        if task not in eligible:
            continue
        pair = choose_strict_pair(record.get("candidates", []))
        if pair is None:
            continue
        chosen, rejected = pair
        eligible[task].append(
            {
                "id": record["id"],
                "source_group": record.get("source_group"),
                "task_type": task,
                "prompt": record["prompt"],
                "chosen": chosen["response"],
                "rejected": rejected["response"],
                "chosen_reward": chosen["reward"],
                "rejected_reward": rejected["reward"],
                "reward_gap": round(chosen["reward"] - rejected["reward"], 4),
            }
        )

    per_task = min(len(records) for records in eligible.values())
    if per_task < 2:
        raise RuntimeError(
            "At least two strict pairs per task are required; got "
            + ", ".join(f"{task}={len(records)}" for task, records in eligible.items())
        )

    output = {"train": [], "valid": []}
    valid_count = max(1, round(per_task * valid_fraction))
    for task in STRICT_TASKS:
        records = eligible[task][:]
        rng.shuffle(records)
        selected = records[:per_task]
        output["valid"].extend(selected[:valid_count])
        output["train"].extend(selected[valid_count:])
    rng.shuffle(output["train"])
    rng.shuffle(output["valid"])

    stats = {
        "candidate_prompts": len(candidate_records),
        "policy": {
            "tasks": list(STRICT_TASKS),
            "chosen": "hard_pass=true",
            "rejected": "hard_pass=false",
            "continuation_excluded": True,
            "balancing": "downsample_to_smallest_task",
        },
        "eligible_pairs": {task: len(eligible[task]) for task in STRICT_TASKS},
        "selected_per_task": per_task,
        "valid_per_task": valid_count,
        "train": {
            "total": len(output["train"]),
            "tasks": dict(Counter(item["task_type"] for item in output["train"])),
        },
        "valid": {
            "total": len(output["valid"]),
            "tasks": dict(Counter(item["task_type"] for item in output["valid"])),
        },
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", default="data/instruction_alignment/candidates.jsonl"
    )
    parser.add_argument("--output-dir", default="data/instruction_dpo_v2")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records, stats = build_strict_dpo_records(
        load_jsonl(Path(args.candidates)), args.valid_fraction, args.seed
    )
    write_jsonl(output_dir / "train.jsonl", records["train"])
    write_jsonl(output_dir / "valid.jsonl", records["valid"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
