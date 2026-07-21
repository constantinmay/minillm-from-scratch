"""Build improved DPO dataset v3.

Principle: chosen from pretrain distribution, rejected from model generation.
- Chosen: real TinyStories continuation (ground truth quality)
- Rejected: model-generated continuation (naturally worse but still realistic)
- Quality gap is moderate and natural, not artificial garbage
"""
import sys, os, re, json, random, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from model.config import MiniLLMConfig
from model.gpt import MiniLLM
from model.generation import generate
from tokenizer.tokenizer_utils import MiniLLMTokenizer


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def quality_score(text):
    words = text.split()
    if len(words) < 5:
        return 0.0
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    ends_ok = 1.0 if text.strip()[-1] in '.!?"\'' else 0.0
    length_score = min(len(words) / 40.0, 1.0)
    return unique_ratio * 0.4 + ends_ok * 0.2 + length_score * 0.2 + (1.0 - min(text.count("  ") / max(len(words), 1), 0.5)) * 0.2


def build_dpo_data(
    stories_path,
    output_path,
    ckpt_path,
    model_config_path,
    tokenizer_path,
    num_samples=500,
    num_candidates=4,
    max_new_tokens=80,
    seed=42,
):
    rng = random.Random(seed)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MiniLLMTokenizer(tokenizer_path)
    model_config = MiniLLMConfig.from_yaml(model_config_path)
    model = MiniLLM(model_config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded from {ckpt_path}")

    # Load stories
    print(f"Loading stories from {stories_path}...")
    with open(stories_path, "r", encoding="utf-8") as f:
        stories = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(stories)} stories")
    rng.shuffle(stories)

    pairs = []
    for story in stories:
        if len(pairs) >= num_samples:
            break

        sentences = split_sentences(story)
        if len(sentences) < 4:
            continue

        n_prompt = rng.choice([1, 2])
        prompt = " ".join(sentences[:n_prompt])
        real_cont = " ".join(sentences[n_prompt:])

        # Quality filter on real continuation
        if len(real_cont.split()) < 20:
            continue
        if quality_score(real_cont) < 0.4:
            continue

        # Generate model candidates
        prompt_ids = tok.encode(prompt + " ", add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        candidates = []
        for _ in range(num_candidates):
            with torch.no_grad():
                out = generate(
                    model, input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.9,
                    top_k=50,
                    eos_token_id=tok.eos_id(),
                )
            text = tok.decode(out[0].tolist(), skip_special_tokens=True)
            gen = text[len(prompt)+1:] if text.startswith(prompt) else text
            score = quality_score(gen)
            candidates.append((gen, score))

        # Pick worst as rejected (but must be a real model output, not empty)
        candidates.sort(key=lambda x: x[1])
        rejected_text, rejected_score = candidates[0]

        if len(rejected_text.split()) < 5:
            continue

        chosen_score = quality_score(real_cont)

        pairs.append({
            "prompt": prompt,
            "chosen": real_cont,
            "rejected": rejected_text,
            "chosen_score": round(chosen_score, 3),
            "rejected_score": round(rejected_score, 3),
            "gap": round(chosen_score - rejected_score, 3),
        })

        if len(pairs) % 100 == 0:
            print(f"  {len(pairs)} / {num_samples}")

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            # Remove score fields for training
            train_item = {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
            f.write(json.dumps(train_item, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(pairs)} DPO pairs to {output_path}")
    gaps = [p["gap"] for p in pairs]
    print(f"Quality gap: avg={sum(gaps)/len(gaps):.3f}, min={min(gaps):.3f}, max={max(gaps):.3f}")

    # Print samples
    print("\n--- Samples ---")
    for p in pairs[:3]:
        print(f"Prompt:    {p['prompt'][:80]}")
        print(f"Chosen:    {p['chosen'][:80]}  (score={p['chosen_score']})")
        print(f"Rejected:  {p['rejected'][:80]}  (score={p['rejected_score']})")
        print(f"Gap:       {p['gap']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ckpt", default="checkpoints_v3/base.pt")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/minillm_tokenizer.json")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dpo_data(
        args.input, args.output,
        args.ckpt, args.model_config, args.tokenizer,
        args.num_samples, args.num_candidates, seed=args.seed,
    )
