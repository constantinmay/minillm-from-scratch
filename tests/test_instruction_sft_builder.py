import random

from scripts.build_instruction_sft import (
    allocate_counts,
    build_continuation,
    build_keyword_story,
    build_question_answering,
    build_sentence_count,
    split_sentences,
    validate_record,
)


STORY = (
    "Lily found a little bird in the garden. "
    "The bird was cold and afraid. "
    "Lily carried the bird inside and gave it some food. "
    "Soon the bird felt warm and happy."
)


def test_split_sentences_retains_complete_sentences():
    sentences = split_sentences(STORY)
    assert len(sentences) == 4
    assert sentences[0].endswith(".")


def test_all_task_builders_produce_valid_records():
    builders = (
        build_continuation,
        build_keyword_story,
        build_sentence_count,
        build_question_answering,
    )
    for offset, builder in enumerate(builders):
        record = builder("source:1", STORY, "train", random.Random(10 + offset))
        assert record is not None
        assert validate_record(record) is None
        assert record["prompt"].endswith("Response:")


def test_test_instruction_wording_is_held_out():
    train = build_continuation("source:1", STORY, "train", random.Random(1))
    test = build_continuation("source:2", STORY, "test", random.Random(1))
    assert train is not None and test is not None
    assert train["prompt"].splitlines()[0] != test["prompt"].splitlines()[0]


def test_validator_detects_broken_constraints():
    keyword = build_keyword_story("source:1", STORY, "train", random.Random(3))
    assert keyword is not None
    keyword["response"] = "This answer contains none of the requested terms."
    assert validate_record(keyword) == "missing_keyword"

    sentence_count = build_sentence_count(
        "source:2", STORY, "train", random.Random(4)
    )
    assert sentence_count is not None
    sentence_count["response"] += " This is one extra sentence."
    assert validate_record(sentence_count) == "wrong_sentence_count"


def test_continuation_only_allocation_uses_the_full_budget():
    assert allocate_counts(123, {"continuation": 1.0}) == {"continuation": 123}
