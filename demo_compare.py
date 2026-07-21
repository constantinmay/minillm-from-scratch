"""Compare selected MiniLLM checkpoints on one controlled instruction."""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch

from model.config import MiniLLMConfig
from model.generation import generate
from model.gpt import MiniLLM
from tokenizer.tokenizer_utils import MiniLLMTokenizer


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
) -> str:
    model = MiniLLM(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state)
    model.eval()

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
    return response


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    config = MiniLLMConfig.from_yaml(args.config)
    tokenizer = MiniLLMTokenizer(args.tokenizer)

    print(f"\nPrompt\n{'-' * 72}\n{prompt}\n")
    for name, checkpoint_path in models.items():
        response = generate_one(
            checkpoint_path,
            prompt,
            config,
            tokenizer,
            args.device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.seed,
        )
        print(f"{name}\n{'-' * 72}\n{response}\n")


if __name__ == "__main__":
    main()
