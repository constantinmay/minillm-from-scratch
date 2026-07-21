"""Tests for the comprehensive evaluation data plumbing."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.comprehensive_eval import (
    load_prompt_records,
    normalize_short_answer,
    parse_models,
    required_keyword_coverage,
)


def test_load_prompt_records_parses_jsonl_fields(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "keyword_story",
                "prompt": "Use cat and ball.",
                "required_words": ["cat", "ball"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_prompt_records(str(path))

    assert records == [
        {
            "type": "keyword_story",
            "prompt": "Use cat and ball.",
            "required_words": ["cat", "ball"],
            "required_sentence_count": None,
            "answer": None,
            "reference_response": None,
        }
    ]


def test_parse_models_requires_named_base():
    models = parse_models(["Base=base.pt", "SFT=sft.pt"])
    assert list(models) == ["Base", "SFT"]
    assert models["SFT"] == "sft.pt"


def test_keyword_coverage_excludes_unconstrained_prompts():
    coverage, count = required_keyword_coverage(
        ["no requested words", "a cat found a ball"],
        [[], ["cat", "ball", "garden"]],
    )

    assert count == 1
    assert coverage == 2 / 3


def test_short_answer_normalization_ignores_case_and_punctuation():
    assert normalize_short_answer("Lily.") == normalize_short_answer("  LILY! ")
