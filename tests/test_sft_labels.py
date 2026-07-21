"""Regression tests for causal next-token alignment in SFT data."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.sft import SFTDataset, build_pretrain_targets, collate_sft


class FakeTokenizer:
    def bos_id(self):
        return 1

    def eos_id(self):
        return 2

    def encode(self, text, add_special_tokens=False):
        return [10 + ord(char) for char in text]


def make_dataset(tmp_path, prompt="Hi", response="World", max_length=64):
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps({"prompt": prompt, "response": response}) + "\n",
        encoding="utf-8",
    )
    return SFTDataset(str(path), FakeTokenizer(), max_length=max_length)


def test_sft_uses_shifted_next_token_labels(tmp_path):
    dataset = make_dataset(tmp_path)
    input_ids, labels = dataset[0]

    tokenizer = FakeTokenizer()
    prompt_ids = tokenizer.encode("Hi ", add_special_tokens=False)
    response_ids = tokenizer.encode("World", add_special_tokens=False)
    full_ids = [tokenizer.bos_id()] + prompt_ids + response_ids + [tokenizer.eos_id()]

    assert input_ids.tolist() == full_ids[:-1]
    assert labels[: len(prompt_ids)].tolist() == [-100] * len(prompt_ids)
    assert labels[len(prompt_ids) :].tolist() == response_ids + [tokenizer.eos_id()]

    # At the first trained position, the input is the final prompt token and
    # the label is the first response token--never the same sequence position.
    first_response_pos = len(prompt_ids)
    assert input_ids[first_response_pos].item() == prompt_ids[-1]
    assert labels[first_response_pos].item() == response_ids[0]
    assert input_ids[first_response_pos].item() != labels[first_response_pos].item()


def test_auxiliary_pretrain_targets_restore_shifted_prompt_only(tmp_path):
    short = make_dataset(tmp_path, prompt="A", response="BC")[0]
    long_path = tmp_path / "second.jsonl"
    long_path.write_text(
        json.dumps({"prompt": "Long", "response": "Answer"}) + "\n",
        encoding="utf-8",
    )
    long = SFTDataset(str(long_path), FakeTokenizer(), max_length=64)[0]

    input_ids, labels = collate_sft([short, long])
    targets = build_pretrain_targets(input_ids, labels)

    for row in range(2):
        trained = (labels[row] != -100).nonzero(as_tuple=False)
        first_response_pos = int(trained[0].item())
        assert torch.equal(
            targets[row, :first_response_pos],
            input_ids[row, 1 : first_response_pos + 1],
        )

        last_trained = int(trained[-1].item())
        assert (targets[row, last_trained + 1 :] == -100).all()


def test_long_prompt_still_leaves_a_response_target(tmp_path):
    dataset = make_dataset(tmp_path, prompt="P" * 100, response="R", max_length=8)
    input_ids, labels = dataset[0]

    assert len(input_ids) == 8
    assert len(labels) == 8
    assert labels[-1].item() != -100
