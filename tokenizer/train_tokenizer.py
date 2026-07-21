"""Train a BPE tokenizer on text data."""

import argparse
import os
import sys

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer.tokenizer_utils import MiniLLMTokenizer


def train_bpe_tokenizer(
    input_files: list,
    save_path: str,
    vocab_size: int = 8000,
    min_frequency: int = 2,
):
    """Train a BPE tokenizer on the given text files."""
    special_tokens = MiniLLMTokenizer.SPECIAL_TOKENS

    tokenizer = Tokenizer(models.BPE(unk_token=MiniLLMTokenizer.UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {input_files}...")
    tokenizer.train(input_files, trainer)

    # Add BOS/EOS wrapping via post-processor
    bos_id = tokenizer.token_to_id(MiniLLMTokenizer.BOS_TOKEN)
    eos_id = tokenizer.token_to_id(MiniLLMTokenizer.EOS_TOKEN)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{MiniLLMTokenizer.BOS_TOKEN} $A {MiniLLMTokenizer.EOS_TOKEN}",
        special_tokens=[
            (MiniLLMTokenizer.BOS_TOKEN, bos_id),
            (MiniLLMTokenizer.EOS_TOKEN, eos_id),
        ],
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})")

    # Verify
    tok = MiniLLMTokenizer(save_path)
    test = "Once upon a time there was a little cat."
    encoded = tok.encode(test)
    decoded = tok.decode(encoded, skip_special_tokens=True)
    print(f"Verification: '{test}' -> {len(encoded)} tokens -> '{decoded}'")
    assert len(encoded) > 0, "Encode returned empty"
    assert len(decoded) > 0, "Decode returned empty"
    # With very small vocab, subword splits may alter whitespace
    for word in ["cat", "little", "time", "was"]:
        assert word in decoded, f"Key word '{word}' lost in roundtrip"
    print("Tokenizer roundtrip OK")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Input text file(s)")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--output", type=str, default="tokenizer/minillm_tokenizer.json")
    parser.add_argument("--min_frequency", type=int, default=2)
    args = parser.parse_args()

    train_bpe_tokenizer(args.input, args.output, args.vocab_size, args.min_frequency)


if __name__ == "__main__":
    main()
