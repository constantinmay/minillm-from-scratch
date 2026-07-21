"""Tokenize text data and pack into fixed-length blocks as numpy memmap.

Supports two modes:
  1. --source hf        Download TinyStories from HuggingFace
  2. --train_file / --valid_file   Use local text files

Usage examples:
  # From HuggingFace TinyStories
  python scripts/prepare_pretrain_data.py \
      --source hf \
      --tokenizer_path tokenizer/tokenizer.json \
      --output_dir data/processed

  # From local text files
  python scripts/prepare_pretrain_data.py \
      --source local \
      --train_file data/raw/toy_train.txt \
      --valid_file data/raw/toy_valid.txt \
      --tokenizer_path tokenizer/tokenizer.json \
      --output_dir data/processed
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Allow imports from the project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from tokenizer.tokenizer_utils import MiniLLMTokenizer


def tokenize_and_pack(
    input_path: str,
    tokenizer: MiniLLMTokenizer,
    output_path: str,
    block_size: int = 256,
) -> int:
    """Read text lines, tokenize all, pack into fixed-length blocks, save as uint16 numpy array.

    Args:
        input_path: Path to a text file (one document per line).
        tokenizer: MiniLLMTokenizer instance.
        output_path: Path to save the packed numpy array.
        block_size: Number of tokens per block.

    Returns:
        Number of complete blocks written.
    """
    print(f"  Reading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"  {len(lines)} lines read.")

    # Tokenize all lines, concatenating with EOS between documents
    all_ids: list[int] = []
    eos_id = tokenizer.eos_id()
    for i, line in enumerate(lines):
        ids = tokenizer.encode(line, add_special_tokens=True)
        all_ids.extend(ids)
        all_ids.append(eos_id)  # separator between documents
        if (i + 1) % 10000 == 0:
            print(f"    Tokenized {i + 1}/{len(lines)} lines ...")

    total_tokens = len(all_ids)
    n_blocks = total_tokens // block_size
    if n_blocks == 0:
        print(f"  WARNING: Only {total_tokens} tokens, fewer than one block_size={block_size}.")
        n_blocks = 0

    # Trim to exact multiple of block_size
    trimmed = all_ids[: n_blocks * block_size]
    arr = np.array(trimmed, dtype=np.uint16)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    arr.tofile(output_path)
    print(f"  Saved {n_blocks} blocks ({len(trimmed)} tokens) -> {output_path}")
    return n_blocks


def load_hf_tinystories(split: str, num_samples: int | None = None):
    """Load TinyStories from HuggingFace datasets.

    Args:
        split: 'train' or 'validation'.
        num_samples: Optional cap on number of samples.

    Returns:
        List of story strings.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package is required for --source hf.")
        print("  Install it with:  pip install datasets")
        sys.exit(1)

    print(f"  Downloading TinyStories split={split} ...")
    ds = load_dataset("roneneldan/TinyStories", split=split, trust_remote_code=True)
    stories = ds["text"]
    if num_samples is not None:
        stories = stories[:num_samples]
    print(f"  Loaded {len(stories)} stories.")
    return stories


def stories_to_temp_file(stories: list[str], path: str) -> str:
    """Write a list of story strings to a temp text file, one per line."""
    with open(path, "w", encoding="utf-8") as f:
        for s in stories:
            s_clean = s.replace("\n", " ").strip()
            if s_clean:
                f.write(s_clean + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text data and pack into fixed-length blocks for pretraining."
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["hf", "local"],
        default="local",
        help="Data source: 'hf' for HuggingFace TinyStories, 'local' for text files.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to local training text file (used when --source local).",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        default=None,
        help="Path to local validation text file (used when --source local).",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer/tokenizer.json",
        help="Path to the HuggingFace tokenizers JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save packed numpy arrays.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Number of tokens per packed block.",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Cap on number of training samples from HF (default: all).",
    )
    parser.add_argument(
        "--num_valid_samples",
        type=int,
        default=None,
        help="Cap on number of validation samples from HF (default: all).",
    )
    args = parser.parse_args()

    tokenizer = MiniLLMTokenizer(args.tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.source == "hf":
        # --- HF TinyStories mode ---
        train_stories = load_hf_tinystories("train", args.num_train_samples)
        valid_stories = load_hf_tinystories("validation", args.num_valid_samples)

        train_tmp = os.path.join(args.output_dir, "_tmp_train.txt")
        valid_tmp = os.path.join(args.output_dir, "_tmp_valid.txt")
        stories_to_temp_file(train_stories, train_tmp)
        stories_to_temp_file(valid_stories, valid_tmp)

        train_out = os.path.join(args.output_dir, "train.bin")
        valid_out = os.path.join(args.output_dir, "valid.bin")

        print("Processing training data:")
        n_train = tokenize_and_pack(train_tmp, tokenizer, train_out, args.block_size)
        print("Processing validation data:")
        n_valid = tokenize_and_pack(valid_tmp, tokenizer, valid_out, args.block_size)

        # Clean up temp files
        for tmp in (train_tmp, valid_tmp):
            if os.path.exists(tmp):
                os.remove(tmp)

    elif args.source == "local":
        # --- Local text files mode ---
        if not args.train_file:
            parser.error("--train_file is required when --source is 'local'.")
        if not os.path.exists(args.train_file):
            parser.error(f"Train file not found: {args.train_file}")

        train_out = os.path.join(args.output_dir, "train.bin")
        print("Processing training data:")
        n_train = tokenize_and_pack(args.train_file, tokenizer, train_out, args.block_size)

        n_valid = 0
        if args.valid_file:
            if not os.path.exists(args.valid_file):
                parser.error(f"Valid file not found: {args.valid_file}")
            valid_out = os.path.join(args.output_dir, "valid.bin")
            print("Processing validation data:")
            n_valid = tokenize_and_pack(args.valid_file, tokenizer, valid_out, args.block_size)

    # --- Summary ---
    meta_path = os.path.join(args.output_dir, "meta.json")
    import json
    meta = {
        "block_size": args.block_size,
        "vocab_size": tokenizer.vocab_size,
        "train_blocks": n_train,
        "valid_blocks": n_valid,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")
    print(f"  train_blocks: {n_train}")
    print(f"  valid_blocks: {n_valid}")
    print("Done.")


if __name__ == "__main__":
    main()
