"""Build improved SFT dataset v2.

Principle: data entirely from pretrain distribution.
- Prompt: first 1-2 sentences of real TinyStories
- Response: remaining sentences from the SAME story
- Only stories with 4+ sentences, response 20+ words
- Small dataset (500 train, 50 valid) to minimize gradient impact
"""
import sys, os, re, json, random, argparse


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def quality_score(text):
    words = text.split()
    if len(words) < 10:
        return 0.0
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    ends_ok = 1.0 if text.strip()[-1] in '.!?"\'' else 0.0
    return unique_ratio * 0.5 + ends_ok * 0.3 + min(len(words) / 50.0, 1.0) * 0.2


def build_sft_data(input_path, output_path, num_samples, seed=42):
    rng = random.Random(seed)

    print(f"Loading stories from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
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

        # Split at sentence boundary
        n_prompt = rng.choice([1, 2])
        prompt = " ".join(sentences[:n_prompt])
        response = " ".join(sentences[n_prompt:])

        # Quality filters
        if len(response.split()) < 20:
            continue
        if quality_score(response) < 0.4:
            continue

        pairs.append({"prompt": prompt, "response": response})

    rng.shuffle(pairs)
    pairs = pairs[:num_samples]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} pairs to {output_path}")

    # Stats
    resp_words = [len(p["response"].split()) for p in pairs]
    print(f"Response length: avg={sum(resp_words)/len(resp_words):.1f}, "
          f"min={min(resp_words)}, max={max(resp_words)}")

    # Samples
    print("\n--- Samples ---")
    for p in pairs[:3]:
        print(f"Prompt:   {p['prompt'][:80]}")
        print(f"Response: {p['response'][:80]}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_sft_data(args.input, args.output, args.num_samples, args.seed)
