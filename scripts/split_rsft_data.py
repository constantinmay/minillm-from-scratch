"""Split RSFT data into train/valid sets."""
import json, random

random.seed(42)

with open("data/rsft/rsft_train.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

random.shuffle(data)
split = int(len(data) * 0.9)
train = data[:split]
valid = data[split:]

# Convert to SFT format (prompt/response)
for name, subset in [("train", train), ("valid", valid)]:
    path = f"data/rsft/rsft_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps({
                "prompt": item["prompt"],
                "response": item["response"],
            }, ensure_ascii=False) + "\n")
    print(f"{name}: {len(subset)} samples -> {path}")
