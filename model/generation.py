"""Text generation utilities for MiniLLM."""

from typing import List, Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    do_sample: bool = True,
) -> torch.Tensor:
    """Autoregressive generation.

    Args:
        model: MiniLLM model.
        input_ids: (B, T) prompt token IDs.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. 1.0 = standard, <1 = sharper, >1 = more random.
        top_k: If set, only sample from top-k logits.
        top_p: If set, nucleus sampling threshold.
        eos_token_id: Stop generation when this token is produced.
        do_sample: If False, use greedy decoding.

    Returns:
        (B, T + generated_len) token IDs.
    """
    model.eval()
    device = input_ids.device

    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        idx_cond = input_ids if input_ids.size(1) <= model.config.block_size else input_ids[:, -model.config.block_size:]

        logits = model(idx_cond)["logits"]
        logits = logits[:, -1, :]  # (B, V)

        if not do_sample:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return input_ids


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: Optional[float] = None,
    device: str = "cuda",
) -> List[str]:
    """Generate text for a batch of prompts.

    Args:
        model: MiniLLM model.
        tokenizer: MiniLLMTokenizer instance.
        prompts: List of prompt strings.
        max_new_tokens: Max tokens to generate per prompt.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        device: Device to run on.

    Returns:
        List of generated text strings (prompt + completion).
    """
    model.eval()
    encoded = tokenizer.encode_batch(prompts)

    max_prompt_len = max(len(t) for t in encoded)
    padded = []
    for tokens in encoded:
        pad_len = max_prompt_len - len(tokens)
        padded.append([tokenizer.pad_id()] * pad_len + tokens)

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)

    output_ids = generate(
        model, input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_id(),
    )

    results = []
    for i, prompt_tokens in enumerate(encoded):
        prompt_len = len(prompt_tokens)
        generated = output_ids[i, prompt_len:].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        results.append(text)

    return results
