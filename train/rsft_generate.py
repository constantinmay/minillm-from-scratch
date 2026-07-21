"""RSFT (Reward-Selected Fine-Tuning) generation pipeline for MiniLLM.

For each prompt, generate k candidates, score with a rule-based reward function,
select the best, and save as JSONL for re-training via SFT.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import re
from collections import Counter
from typing import List, Dict, Optional

import torch
from tqdm import tqdm

from model.config import MiniLLMConfig
from model.gpt import MiniLLM
from model.generation import generate
from tokenizer.tokenizer_utils import MiniLLMTokenizer
from train.common import get_device, load_config, load_checkpoint


def score_response(response: str, required_words: Optional[List[str]] = None) -> Dict:
    """Three-layer reward function for scoring generated responses.

    Uses only string/text analysis, no ML models.

    Layer 1 - Hard constraints:
        - word_count >= 10
        - repeat_3gram < 30%
        - at least 1 required word (if provided)

    Layer 2 - Quality signals:
        - keyword coverage (fraction of required words present)
        - length appropriateness (not too short, not too long)
        - sentence completeness (ends with period/question/exclamation)
        - coherence (no excessive repetition of consecutive words)

    Layer 3 - Progressive penalties:
        - repetition penalty (unique word ratio)
        - template collapse penalty (penalize generic/boilerplate phrases)
        - oversimplification penalty (penalize very short sentences)

    Args:
        response: The generated text to score.
        required_words: Optional list of keywords that should appear.

    Returns:
        Dict with 'total_reward', 'hard_pass', and 'details'.
    """
    if required_words is None:
        required_words = []

    response = response.strip()
    words = response.split()
    word_count = len(words)

    # ---------------------------------------------------------------
    # Layer 1: Hard constraints (must all pass for hard_pass=True)
    # ---------------------------------------------------------------
    hard_pass = True
    hard_details = {}

    # Word count >= 10
    min_word_count = 10
    length_ok = word_count >= min_word_count
    hard_details["length_ok"] = length_ok
    if not length_ok:
        hard_pass = False

    # Repeat 3-gram ratio < 30%
    if word_count >= 3:
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append(tuple(words[i : i + 3]))
        if trigrams:
            trigram_counts = Counter(trigrams)
            max_repeat = max(trigram_counts.values())
            repeat_3gram_ratio = max_repeat / len(trigrams)
        else:
            repeat_3gram_ratio = 0.0
    else:
        repeat_3gram_ratio = 0.0

    repeat_ok = repeat_3gram_ratio < 0.3
    hard_details["repeat_3gram_ratio"] = repeat_3gram_ratio
    hard_details["repeat_ok"] = repeat_ok
    if not repeat_ok:
        hard_pass = False

    # At least 1 required word (if required_words provided)
    if required_words:
        response_lower = response.lower()
        has_required = any(w.lower() in response_lower for w in required_words)
        hard_details["has_required_word"] = has_required
        if not has_required:
            hard_pass = False
    else:
        hard_details["has_required_word"] = True

    # ---------------------------------------------------------------
    # Layer 2: Quality signals
    # ---------------------------------------------------------------
    quality_reward = 0.0
    quality_details = {}

    # Keyword coverage: fraction of required words present
    if required_words:
        response_lower = response.lower()
        covered = sum(1 for w in required_words if w.lower() in response_lower)
        keyword_coverage = covered / len(required_words)
    else:
        keyword_coverage = 1.0  # no requirements, full coverage
    quality_reward += keyword_coverage * 0.3
    quality_details["keyword_coverage"] = keyword_coverage

    # Length appropriateness: reward 20-200 words, penalize extremes
    if 20 <= word_count <= 200:
        length_score = 1.0
    elif 10 <= word_count < 20:
        length_score = (word_count - 10) / 10.0  # ramp from 0 to 1
    elif word_count > 200:
        length_score = max(0.0, 1.0 - (word_count - 200) / 200.0)
    else:
        length_score = 0.0
    quality_reward += length_score * 0.2
    quality_details["length_score"] = length_score

    # Sentence completeness: ends with . ? or !
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = len(sentences)
    ends_properly = len(response) > 0 and response[-1] in ".!?"
    completeness_score = 0.5 if ends_properly else 0.0
    if num_sentences >= 2:
        completeness_score += 0.5
    quality_reward += completeness_score * 0.2
    quality_details["sentence_completeness"] = completeness_score

    # Coherence: penalize excessive consecutive word repetition
    consecutive_repeats = 0
    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            consecutive_repeats += 1
    if len(words) > 1:
        coherence_score = max(0.0, 1.0 - consecutive_repeats / (len(words) - 1))
    else:
        coherence_score = 0.0
    quality_reward += coherence_score * 0.3
    quality_details["coherence_score"] = coherence_score

    # ---------------------------------------------------------------
    # Layer 3: Progressive penalties
    # ---------------------------------------------------------------
    penalty = 0.0
    penalty_details = {}

    # Repetition penalty: unique word ratio
    if word_count > 0:
        unique_ratio = len(set(w.lower() for w in words)) / word_count
    else:
        unique_ratio = 0.0
    if unique_ratio < 0.4:
        rep_penalty = (0.4 - unique_ratio) * 2.0  # up to 0.8 penalty
    else:
        rep_penalty = 0.0
    penalty += rep_penalty
    penalty_details["repetition_penalty"] = rep_penalty

    # Template collapse penalty: penalize generic/boilerplate phrases
    generic_phrases = [
        "i don't know",
        "i am not sure",
        "i can't help",
        "as an ai",
        "as a language model",
        "i'm sorry",
        "i cannot",
    ]
    response_lower = response.lower()
    generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
    template_penalty = min(generic_count * 0.3, 1.0)
    penalty += template_penalty
    penalty_details["template_collapse_penalty"] = template_penalty

    # Oversimplification penalty: penalize very short average sentence length
    if num_sentences > 0:
        avg_sentence_len = word_count / num_sentences
        if avg_sentence_len < 3:
            simple_penalty = (3.0 - avg_sentence_len) / 3.0 * 0.5
        else:
            simple_penalty = 0.0
    else:
        simple_penalty = 0.3
    penalty += simple_penalty
    penalty_details["oversimplification_penalty"] = simple_penalty

    # ---------------------------------------------------------------
    # Total reward
    # ---------------------------------------------------------------
    # If hard constraints fail, heavily reduce reward
    if hard_pass:
        total_reward = quality_reward - penalty
    else:
        total_reward = (quality_reward - penalty) * 0.1  # heavy discount

    total_reward = max(total_reward, 0.0)  # floor at 0

    return {
        "total_reward": round(total_reward, 4),
        "hard_pass": hard_pass,
        "details": {
            "word_count": word_count,
            "num_sentences": num_sentences,
            "unique_ratio": round(unique_ratio, 4) if word_count > 0 else 0.0,
            "hard_constraints": hard_details,
            "quality": quality_details,
            "penalties": penalty_details,
        },
    }


def rsft_generate(
    model_path: str,
    tokenizer_path: str,
    prompts_path: str,
    output_path: str,
    model_config_path: str = "configs/model_config.yaml",
    k: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    device: Optional[str] = None,
):
    """RSFT generation pipeline.

    For each prompt, generate k candidates, score with reward, select best.
    Save as JSONL for re-training via SFT.

    Args:
        model_path: Path to model checkpoint.
        tokenizer_path: Path to tokenizer JSON file.
        prompts_path: Path to JSONL file with prompts (each line has 'prompt' field).
        output_path: Path to write output JSONL file.
        model_config_path: Path to model config YAML.
        k: Number of candidates per prompt.
        max_new_tokens: Max tokens to generate per candidate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        device: Device to use (auto-detected if None).
    """
    if device is None:
        device = get_device()
    print(f"Using device: {device}")

    # Load model
    model_config = MiniLLMConfig.from_yaml(model_config_path)
    model = MiniLLM(model_config).to(device)
    checkpoint = load_checkpoint(model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {model_path}")

    # Load tokenizer
    tokenizer = MiniLLMTokenizer(tokenizer_path)

    # Load prompts
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append(item)

    print(f"Loaded {len(prompts)} prompts, generating {k} candidates each")

    # Generate and score
    results = []
    for entry in tqdm(prompts, desc="RSFT Generate"):
        prompt = entry["prompt"]
        required_words = None  # Don't require prompt words in continuation

        # Encode prompt (no special tokens to avoid EOS issues)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Generate k candidates
        candidates = []
        for _ in range(k):
            output_ids = generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=tokenizer.eos_id(),
                do_sample=True,
            )
            generated_tokens = output_ids[0, len(prompt_ids):].tolist()
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Score the response
            score = score_response(generated_text, required_words)
            candidates.append({
                "response": generated_text,
                "reward": score["total_reward"],
                "hard_pass": score["hard_pass"],
                "details": score["details"],
            })

        # Select best candidate
        # Prefer candidates that pass hard constraints, then highest reward
        passing = [c for c in candidates if c["hard_pass"]]
        if passing:
            best = max(passing, key=lambda c: c["reward"])
        else:
            best = max(candidates, key=lambda c: c["reward"])

        results.append({
            "prompt": prompt,
            "response": best["response"],
            "reward": best["reward"],
            "hard_pass": best["hard_pass"],
            "num_candidates": k,
        })

    # Save output JSONL
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary stats
    total = len(results)
    passing = sum(1 for r in results if r["hard_pass"])
    avg_reward = sum(r["reward"] for r in results) / total if total > 0 else 0.0
    print(f"\nRSFT Generation Complete:")
    print(f"  Total prompts: {total}")
    print(f"  Hard-pass rate: {passing}/{total} ({100*passing/total:.1f}%)" if total > 0 else "  No prompts")
    print(f"  Average reward: {avg_reward:.4f}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSFT Generation Pipeline for MiniLLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rsft_generate.yaml",
        help="Path to RSFT config YAML",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides config)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (overrides config)",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default=None,
        help="Path to prompts JSONL (overrides config)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output JSONL (overrides config)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of candidates per prompt (overrides config)",
    )
    args = parser.parse_args()

    # Load config file if it exists, otherwise use defaults
    cfg = {}
    if os.path.exists(args.config):
        cfg = load_config(args.config)

    rsft_generate(
        model_path=args.model_path or cfg.get("model_path", "checkpoints/sft.pt"),
        tokenizer_path=args.tokenizer_path or cfg.get("tokenizer_path", "tokenizer/minillm_tokenizer.json"),
        prompts_path=args.prompts_path or cfg.get("prompts_path", "data/sft/train.jsonl"),
        output_path=args.output_path or cfg.get("output_path", "data/rsft/rsft_train.jsonl"),
        model_config_path=cfg.get("model_config", "configs/model_config.yaml"),
        k=args.k or cfg.get("k", 4),
        max_new_tokens=cfg.get("max_new_tokens", 128),
        temperature=cfg.get("temperature", 0.8),
        top_k=cfg.get("top_k", 40),
    )
