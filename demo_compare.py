"""Compare selected MiniLLM checkpoints on one controlled instruction."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch

from model.config import MiniLLMConfig
from model.generation import generate
from model.gpt import MiniLLM
from scripts.export_inference_checkpoint import (
    checkpoint_format,
    load_model_checkpoint,
)
from tokenizer.tokenizer_utils import MiniLLMTokenizer
from train.common import sha256_file


DEFAULT_MODELS = (
    ("InstructionSFT", "checkpoints/instruction_sft/sft.pt"),
    ("DPOv2", "checkpoints/instruction_dpo_v2/dpo_step_200.pt"),
    ("RSFT", "checkpoints/instruction_rsft/rsft.pt"),
)


def build_prompt(
    task: str,
    input_text: str,
    question: str | None = None,
    keywords: list[str] | None = None,
    sentence_count: int | None = None,
) -> str:
    if task == "raw":
        return input_text
    if task == "continuation":
        instruction = "Continue the story."
    elif task == "qa":
        if not question:
            raise ValueError("--question is required for task=qa")
        return (
            "Instruction: Answer the question using the story.\n"
            f"Input: {input_text}\nQuestion: {question}\nResponse:"
        )
    elif task == "keywords":
        words = [word.strip() for word in (keywords or []) if word.strip()]
        if not words:
            raise ValueError("--keywords is required for task=keywords")
        quoted = " and ".join(f'\"{word}\"' for word in words)
        instruction = f"Continue the story and use the words {quoted}."
    elif task == "sentence_count":
        if sentence_count is None or sentence_count < 1:
            raise ValueError("--sentences must be positive for task=sentence_count")
        instruction = f"Continue the story in exactly {sentence_count} sentences."
    else:
        raise ValueError(f"Unsupported task: {task}")
    return f"Instruction: {instruction}\nInput: {input_text}\nResponse:"


def parse_models(values: list[str] | None) -> OrderedDict[str, str]:
    models: OrderedDict[str, str] = OrderedDict()
    if not values:
        models.update(DEFAULT_MODELS)
        return models
    for value in values:
        if "=" not in value:
            raise ValueError("--model must use NAME=CHECKPOINT format")
        name, path = value.split("=", 1)
        models[name] = path
    return models


def resolve_device(value: str) -> str:
    """Resolve auto/cpu/cuda without silently ignoring an unavailable CUDA request."""
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available; use --device cpu or auto")
    return value


def missing_checkpoint_message(models: OrderedDict[str, str]) -> str | None:
    """Return a concise actionable message when requested checkpoints are absent."""
    missing = [(name, path) for name, path in models.items() if not Path(path).is_file()]
    if not missing:
        return None
    lines = ["Missing required checkpoint file(s):"]
    lines.extend(f"  - {name}: {path}" for name, path in missing)
    lines.append(
        "Train the corresponding models or download verified inference checkpoints "
        "from a GitHub Release. Release assets will be published with the next version."
    )
    return "\n".join(lines)


def load_demo_model(
    checkpoint_path: str,
    fallback_config: MiniLLMConfig,
    device: str,
) -> tuple[MiniLLM, dict, dict]:
    """Load a training or inference checkpoint and return display metadata."""
    model, checkpoint, _ = load_model_checkpoint(
        checkpoint_path,
        device=device,
        fallback_config=fallback_config,
    )
    info = {
        "path": checkpoint_path,
        "format": checkpoint_format(checkpoint),
        "training_stage": checkpoint.get(
            "training_stage", checkpoint.get("checkpoint_type", "unknown")
        ),
        "sha256": sha256_file(checkpoint_path),
        "device": device,
    }
    return model, checkpoint, info


@torch.no_grad()
def generate_one(
    checkpoint_path: str,
    prompt: str,
    config: MiniLLMConfig,
    tokenizer: MiniLLMTokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> tuple[str, dict]:
    model, _, info = load_demo_model(checkpoint_path, config, device)

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    prompt_ids = [tokenizer.bos_id()] + tokenizer.encode(
        prompt + " ", add_special_tokens=False
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-5),
        top_k=top_k,
        eos_token_id=tokenizer.eos_id(),
        do_sample=temperature > 0,
    )
    response_ids = output[0, len(prompt_ids) :].tolist()
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return response, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("continuation", "qa", "keywords", "sentence_count", "raw"),
        default="continuation",
    )
    parser.add_argument("--input", help="Story/input text; prompted interactively if omitted")
    parser.add_argument("--question")
    parser.add_argument("--keywords", help="Comma-separated required words")
    parser.add_argument("--sentences", type=int)
    parser.add_argument("--model", action="append", help="NAME=CHECKPOINT; repeatable")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/minillm_tokenizer.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_text = args.input or input("Input/story> ").strip()
    keyword_list = args.keywords.split(",") if args.keywords else None
    prompt = build_prompt(
        args.task, input_text, args.question, keyword_list, args.sentences
    )
    models = parse_models(args.model)
    missing_message = missing_checkpoint_message(models)
    if missing_message:
        raise SystemExit(missing_message)
    device = resolve_device(args.device)
    config = MiniLLMConfig.from_yaml(args.config)
    tokenizer = MiniLLMTokenizer(args.tokenizer)

    print(f"\nPrompt\n{'-' * 72}\n{prompt}\n")
    for name, checkpoint_path in models.items():
        response, info = generate_one(
            checkpoint_path,
            prompt,
            config,
            tokenizer,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.seed,
        )
        print(
            f"{name}\n{'-' * 72}\n"
            f"Checkpoint: {info['path']}\n"
            f"Format: {info['format']}\n"
            f"Training stage: {info['training_stage']}\n"
            f"SHA256: {info['sha256']}\n"
            f"Device: {info['device']}\n\n"
            f"{response}\n"
        )


if __name__ == "__main__":
    main()
