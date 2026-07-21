"""Build a reproducible, leakage-safe TinyStories instruction SFT dataset.

The builder derives four objectively checkable tasks from real TinyStories:
story continuation, required-keyword continuation, exact sentence count, and
extractive "who" question answering.  Source stories are split before examples
are derived, so variants of one story cannot cross validation/test boundaries.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASK_RATIOS = {
    "continuation": 0.35,
    "keyword_story": 0.25,
    "sentence_count": 0.20,
    "question_answering": 0.20,
}

TRAIN_INSTRUCTIONS = {
    "continuation": (
        "Continue the story.",
        "Write what happens next.",
        "Finish the following story.",
        "Add the next part of the story.",
    ),
    "keyword_story": (
        'Continue the story and use the words {words}.',
        'Write what happens next. Include the words {words}.',
        'Finish the story using the words {words}.',
    ),
    "sentence_count": (
        "Continue the story in exactly {count} {unit}.",
        "Write exactly {count} {unit} about what happens next.",
        "Finish the story with exactly {count} {unit}.",
    ),
    "question_answering": (
        "Answer the question using the story.",
        "Read the story and answer the question.",
        "Give a short answer based only on the story.",
    ),
}

# Held-out wording makes the test set measure paraphrase generalization too.
TEST_INSTRUCTIONS = {
    "continuation": ("What happens next in this story?",),
    "keyword_story": ('Complete the story. Your answer must contain {words}.',),
    "sentence_count": ("Tell the next part using only {count} {unit}.",),
    "question_answering": ("Based on the passage, give the answer.",),
}

STOPWORDS = {
    "about", "after", "again", "also", "and", "because", "before", "but",
    "came", "could", "did", "down", "each", "from", "gave", "good", "had",
    "has", "have", "her", "here", "him", "his", "into", "just", "little",
    "looked", "made", "more", "much", "not", "one", "only", "other", "out",
    "said", "saw", "she", "some", "that", "the", "their", "them", "then",
    "there", "they", "this", "time", "too", "very", "wanted", "was", "went",
    "were", "what", "when", "where", "which", "who", "with", "would", "you",
}

BAD_SUBJECTS = {
    "A", "An", "And", "As", "At", "But", "Finally", "He", "Her", "His",
    "I", "If", "In", "It", "Mom", "Mother", "Once", "One", "She", "So",
    "Suddenly", "The", "Then", "There", "They", "This", "We", "When", "While", "You",
}

PREDICATE_STARTS = {
    "asked", "brought", "called", "carried", "climbed", "cried", "decided",
    "did", "felt", "found", "gave", "got", "had", "has", "heard", "helped",
    "is", "jumped", "knew", "laughed", "liked", "lived", "looked", "loved",
    "made", "opened", "played", "pulled", "ran", "replied", "said", "sat",
    "saw", "smiled", "started", "told", "took", "tried", "walked", "wanted",
    "was", "went", "wished",
}


def split_sentences(text: str) -> list[str]:
    """Split simple English prose while retaining terminal punctuation."""
    text = re.sub(r"\s+", " ", text.strip())
    # Python look-behind cannot contain the optional closing quote, so insert
    # an explicit separator after terminal punctuation instead.
    marked = re.sub(r'([.!?]["\']?)\s+', r"\1\n", text)
    parts = marked.splitlines()
    return [part.strip() for part in parts if part.strip()]


def read_stories(path: Path) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            story = line.strip()
            if story:
                yield line_number, story


def reservoir_sample(
    stories: Iterable[tuple[int, str]], size: int, rng: random.Random
) -> list[tuple[int, str]]:
    """Uniformly sample a large source file without loading it into memory."""
    reservoir: list[tuple[int, str]] = []
    for seen, item in enumerate(stories, 1):
        if len(reservoir) < size:
            reservoir.append(item)
        else:
            position = rng.randrange(seen)
            if position < size:
                reservoir[position] = item
    rng.shuffle(reservoir)
    return reservoir


def allocate_counts(
    total: int, task_ratios: dict[str, float] | None = None
) -> dict[str, int]:
    task_ratios = task_ratios or TASK_RATIOS
    if not task_ratios or any(task not in BUILDERS for task in task_ratios):
        raise ValueError(f"Unsupported task ratios: {task_ratios}")
    ratio_sum = sum(task_ratios.values())
    if ratio_sum <= 0:
        raise ValueError("Task ratios must sum to a positive value")
    normalized = {task: ratio / ratio_sum for task, ratio in task_ratios.items()}
    counts = {task: int(total * ratio) for task, ratio in normalized.items()}
    first_task = next(iter(normalized))
    counts[first_task] += total - sum(counts.values())
    return counts


def choose_split(sentences: list[str], rng: random.Random) -> tuple[str, list[str]] | None:
    if len(sentences) < 3:
        return None
    prompt_count = rng.choice((1, 2)) if len(sentences) >= 4 else 1
    remaining = sentences[prompt_count:]
    input_text = " ".join(sentences[:prompt_count])
    if not remaining or input_text.count('"') % 2:
        return None
    return input_text, remaining


def clean_response(sentences: list[str], max_sentences: int = 5) -> str | None:
    response = " ".join(sentences[:max_sentences]).strip()
    words = response.split()
    if not 10 <= len(words) <= 110 or response.count('"') % 2:
        return None
    if response[-1:] not in '.!?"\'':
        return None
    return response


def format_prompt(instruction: str, input_text: str, question: str | None = None) -> str:
    fields = [f"Instruction: {instruction}", f"Input: {input_text}"]
    if question:
        fields.append(f"Question: {question}")
    fields.append("Response:")
    return "\n".join(fields)


def instruction_for(
    task: str,
    split: str,
    rng: random.Random,
    **values: object,
) -> tuple[str, int]:
    bank = TEST_INSTRUCTIONS if split == "test" else TRAIN_INSTRUCTIONS
    variants = bank[task]
    variant_id = rng.randrange(len(variants))
    return variants[variant_id].format(**values), variant_id


def base_record(
    source_id: str,
    task: str,
    prompt: str,
    response: str,
    variant_id: int,
    **metadata: object,
) -> dict:
    source_name, line_number = source_id.rsplit(":", 1)
    record = {
        "id": "",  # Assigned after deterministic shuffling.
        "source_id": source_id,
        # Raw local files retain TinyStories paragraph newlines. Group nearby
        # lines conservatively so paragraphs from one story cannot cross
        # validation/test splits.
        "source_group": f"{source_name}:block-{int(line_number) // 10}",
        "task_type": task,
        "instruction_variant": variant_id,
        "prompt": prompt,
        "response": response,
    }
    record.update(metadata)
    return record


def build_continuation(
    source_id: str, story: str, split: str, rng: random.Random
) -> dict | None:
    divided = choose_split(split_sentences(story), rng)
    if divided is None:
        return None
    input_text, remaining = divided
    response = clean_response(remaining)
    if response is None:
        return None
    instruction, variant = instruction_for("continuation", split, rng)
    return base_record(
        source_id, "continuation", format_prompt(instruction, input_text), response, variant
    )


def content_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    unique: list[str] = []
    seen: set[str] = set()
    for word in words:
        normalized = word.lower()
        if len(normalized) < 4 or normalized in STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def build_keyword_story(
    source_id: str, story: str, split: str, rng: random.Random
) -> dict | None:
    divided = choose_split(split_sentences(story), rng)
    if divided is None:
        return None
    input_text, remaining = divided
    response = clean_response(remaining)
    if response is None:
        return None
    candidates = content_words(response)
    if len(candidates) < 2:
        return None
    required = sorted(rng.sample(candidates, k=2))
    displayed = '"' + '" and "'.join(required) + '"'
    instruction, variant = instruction_for(
        "keyword_story", split, rng, words=displayed
    )
    return base_record(
        source_id,
        "keyword_story",
        format_prompt(instruction, input_text),
        response,
        variant,
        required_words=required,
    )


def build_sentence_count(
    source_id: str, story: str, split: str, rng: random.Random
) -> dict | None:
    divided = choose_split(split_sentences(story), rng)
    if divided is None:
        return None
    input_text, remaining = divided
    count = rng.randint(1, min(3, len(remaining)))
    response = clean_response(remaining[:count], max_sentences=count)
    if response is None or len(split_sentences(response)) != count:
        return None
    unit = "sentence" if count == 1 else "sentences"
    instruction, variant = instruction_for(
        "sentence_count", split, rng, count=count, unit=unit
    )
    return base_record(
        source_id,
        "sentence_count",
        format_prompt(instruction, input_text),
        response,
        variant,
        required_sentence_count=count,
    )


def build_question_answering(
    source_id: str, story: str, split: str, rng: random.Random
) -> dict | None:
    sentences = split_sentences(story)
    if len(sentences) < 2:
        return None
    candidates: list[tuple[str, str]] = []
    for sentence in sentences[:6]:
        match = re.match(r'^([A-Z][a-z]{2,})\s+(.+)$', sentence)
        if not match or match.group(1) in BAD_SUBJECTS:
            continue
        answer, predicate = match.groups()
        predicate = predicate.rstrip('.!?"\'')
        predicate_words = predicate.split()
        if (
            len(predicate_words) < 2
            or predicate_words[0].lower() not in PREDICATE_STARTS
            or answer.lower() in predicate.lower().split()
            or '"' in predicate
        ):
            continue
        candidates.append((answer, f"Who {predicate}?"))
    if not candidates:
        return None
    answer, question = rng.choice(candidates)
    input_text = " ".join(sentences[: min(5, len(sentences))])
    if len(input_text.split()) > 120 or input_text.count('"') % 2:
        return None
    instruction, variant = instruction_for("question_answering", split, rng)
    return base_record(
        source_id,
        "question_answering",
        format_prompt(instruction, input_text, question),
        answer + ".",
        variant,
        answer=answer,
        question=question,
    )


BUILDERS: dict[str, Callable[[str, str, str, random.Random], dict | None]] = {
    "continuation": build_continuation,
    "keyword_story": build_keyword_story,
    "sentence_count": build_sentence_count,
    "question_answering": build_question_answering,
}


def validate_record(record: dict, tokenizer=None, max_length: int = 256) -> str | None:
    prompt = record["prompt"].strip()
    response = record["response"].strip()
    if not prompt or not response:
        return "empty"
    if not prompt.endswith("Response:"):
        return "bad_template"
    if tokenizer is not None:
        token_count = len(tokenizer.encode(prompt + " " + response, add_special_tokens=False))
        if token_count + 2 > max_length:
            return "too_many_tokens"
    if record["task_type"] == "keyword_story":
        lowered = response.lower()
        if not all(re.search(rf"\b{re.escape(word)}\b", lowered) for word in record["required_words"]):
            return "missing_keyword"
    if record["task_type"] == "sentence_count":
        if len(split_sentences(response)) != record["required_sentence_count"]:
            return "wrong_sentence_count"
    if record["task_type"] == "question_answering":
        input_text = prompt.split("\nInput: ", 1)[1].split("\nQuestion:", 1)[0]
        if record["answer"].lower() not in input_text.lower():
            return "ungrounded_answer"
    return None


def build_split(
    source_name: str,
    stories: list[tuple[int, str]],
    split: str,
    total: int,
    rng: random.Random,
    tokenizer=None,
    max_length: int = 256,
    task_ratios: dict[str, float] | None = None,
) -> tuple[list[dict], Counter]:
    requested = allocate_counts(total, task_ratios)
    records: list[dict] = []
    rejected: Counter = Counter()
    for task, target in requested.items():
        candidates = stories.copy()
        rng.shuffle(candidates)
        accepted = 0
        for line_number, story in candidates:
            source_id = f"{source_name}:{line_number}"
            record = BUILDERS[task](source_id, story, split, rng)
            if record is None:
                rejected[f"{task}:not_constructible"] += 1
                continue
            reason = validate_record(record, tokenizer, max_length)
            if reason:
                rejected[f"{task}:{reason}"] += 1
                continue
            records.append(record)
            accepted += 1
            if accepted == target:
                break
        if accepted != target:
            raise RuntimeError(
                f"Could only build {accepted}/{target} {task} records for {split}. "
                "Increase --train-pool-size/--eval-pool-size or reduce split size."
            )
    rng.shuffle(records)
    for index, record in enumerate(records, 1):
        record["id"] = f"{split}_{index:06d}"
    return records, rejected


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(
    records_by_split: dict[str, list[dict]],
    rejected: Counter,
    seed: int,
    task_ratios: dict[str, float] | None = None,
) -> dict:
    source_sets = {
        split: {record["source_group"] for record in records}
        for split, records in records_by_split.items()
    }
    overlaps = {
        "train_valid": len(source_sets["train"] & source_sets["valid"]),
        "train_test": len(source_sets["train"] & source_sets["test"]),
        "valid_test": len(source_sets["valid"] & source_sets["test"]),
    }
    result = {
        "seed": seed,
        "task_ratios": task_ratios or TASK_RATIOS,
        "source_overlaps": overlaps,
    }
    result["splits"] = {}
    for split, records in records_by_split.items():
        lengths = [len(record["response"].split()) for record in records]
        result["splits"][split] = {
            "total": len(records),
            "tasks": dict(Counter(record["task_type"] for record in records)),
            "unique_sources": len(source_sets[split]),
            "average_response_words": round(sum(lengths) / len(lengths), 2),
            "min_response_words": min(lengths),
            "max_response_words": max(lengths),
        }
    result["rejected_candidates"] = dict(sorted(rejected.items()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", default="data/raw/TinyStories-train.txt")
    parser.add_argument("--eval-source", default="data/raw/TinyStories-valid.txt")
    parser.add_argument("--output-dir", default="data/instruction_sft")
    parser.add_argument("--tokenizer-path", default="tokenizer/minillm_tokenizer.json")
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--valid-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--train-pool-size", type=int, default=50_000)
    parser.add_argument("--eval-pool-size", type=int, default=12_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task-mix",
        choices=("multitask", "continuation_only"),
        default="multitask",
    )
    args = parser.parse_args()

    from tokenizer.tokenizer_utils import MiniLLMTokenizer

    rng = random.Random(args.seed)
    task_ratios = (
        {"continuation": 1.0}
        if args.task_mix == "continuation_only"
        else TASK_RATIOS
    )
    tokenizer = MiniLLMTokenizer(args.tokenizer_path)
    print(f"Sampling source stories (seed={args.seed})...")
    train_pool = reservoir_sample(
        read_stories(Path(args.train_source)), args.train_pool_size, rng
    )
    eval_pool = reservoir_sample(
        read_stories(Path(args.eval_source)), args.eval_pool_size, rng
    )
    valid_pool = [item for item in eval_pool if (item[0] // 10) % 2 == 0]
    test_pool = [item for item in eval_pool if (item[0] // 10) % 2 == 1]

    records_by_split: dict[str, list[dict]] = {}
    rejected: Counter = Counter()
    for split, pool, size, source_name in (
        ("train", train_pool, args.train_size, "TinyStories-train"),
        ("valid", valid_pool, args.valid_size, "TinyStories-valid"),
        ("test", test_pool, args.test_size, "TinyStories-valid"),
    ):
        records, split_rejected = build_split(
            source_name,
            pool,
            split,
            size,
            rng,
            tokenizer,
            args.max_length,
            task_ratios,
        )
        records_by_split[split] = records
        rejected.update(split_rejected)
        print(f"Built {len(records):,} {split} records")

    output_dir = Path(args.output_dir)
    for split, records in records_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)

    stats = summarize(records_by_split, rejected, args.seed, task_ratios)
    if any(stats["source_overlaps"].values()):
        raise RuntimeError(f"Source leakage detected: {stats['source_overlaps']}")
    with (output_dir / "statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Dataset written to {output_dir}")


if __name__ == "__main__":
    main()
