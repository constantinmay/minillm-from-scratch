from scripts.build_instruction_alignment import (
    choose_pair,
    score_instruction_response,
    select_balanced_records,
)
from scripts.build_strict_dpo import build_strict_dpo_records, choose_strict_pair


def test_task_rewards_prefer_constraint_satisfaction():
    qa = {"task_type": "question_answering", "answer": "Lily"}
    assert score_instruction_response(qa, "Lily.")["hard_pass"]
    assert not score_instruction_response(qa, "Tom.")["hard_pass"]

    sentence = {"task_type": "sentence_count", "required_sentence_count": 2}
    exact = score_instruction_response(sentence, "One sentence. Two sentences.")
    wrong = score_instruction_response(sentence, "Only one sentence.")
    assert exact["hard_pass"]
    assert exact["reward"] > wrong["reward"]

    keyword = {
        "task_type": "keyword_story",
        "required_words": ["dog", "ball"],
    }
    full = score_instruction_response(keyword, "The dog found a red ball.")
    partial = score_instruction_response(keyword, "The dog ran home.")
    assert full["hard_pass"]
    assert full["reward"] > partial["reward"]


def test_choose_pair_uses_distinct_high_and_low_reward_candidates():
    candidates = [
        {"response": "best", "reward": 0.9, "hard_pass": True},
        {"response": "middle", "reward": 0.5, "hard_pass": False},
        {"response": "worst", "reward": 0.1, "hard_pass": False},
    ]
    chosen, rejected = choose_pair(candidates, min_reward_gap=0.2)
    assert chosen["response"] == "best"
    assert rejected["response"] == "worst"


def test_balanced_selection_is_task_balanced_and_source_unique():
    records = []
    tasks = (
        "continuation",
        "keyword_story",
        "sentence_count",
        "question_answering",
    )
    for task in tasks:
        for index in range(4):
            records.append(
                {
                    "id": f"{task}-{index}",
                    "task_type": task,
                    "source_group": f"{task}-source-{index}",
                }
            )
    selected = select_balanced_records(records, total=12, valid_fraction=0.25, seed=7)
    counts = {task: 0 for task in tasks}
    for record in selected:
        counts[record["task_type"]] += 1
    assert set(counts.values()) == {3}
    assert len({record["source_group"] for record in selected}) == len(selected)
    assert {record["alignment_split"] for record in selected} == {"train", "valid"}


def test_strict_pair_requires_hard_success_and_hard_failure():
    candidates = [
        {"response": "pass", "reward": 0.9, "hard_pass": True},
        {"response": "also pass", "reward": 0.8, "hard_pass": True},
        {"response": "fail", "reward": 0.2, "hard_pass": False},
    ]
    chosen, rejected = choose_strict_pair(candidates)
    assert chosen["response"] == "pass"
    assert chosen["hard_pass"]
    assert rejected["response"] == "fail"
    assert not rejected["hard_pass"]
    assert choose_strict_pair(candidates[:2]) is None


def test_strict_export_excludes_continuation_and_balances_tasks():
    records = []
    for task, count in {
        "question_answering": 4,
        "sentence_count": 5,
        "keyword_story": 6,
        "continuation": 8,
    }.items():
        for index in range(count):
            records.append(
                {
                    "id": f"{task}-{index}",
                    "source_group": f"source-{task}-{index}",
                    "task_type": task,
                    "prompt": "Instruction: test\nResponse:",
                    "candidates": [
                        {"response": "good", "reward": 1.0, "hard_pass": True},
                        {"response": "bad", "reward": 0.0, "hard_pass": False},
                    ],
                }
            )
    output, stats = build_strict_dpo_records(records, valid_fraction=0.25, seed=3)
    assert stats["selected_per_task"] == 4
    assert stats["train"]["tasks"] == {
        "question_answering": 3,
        "sentence_count": 3,
        "keyword_story": 3,
    }
    assert stats["valid"]["tasks"] == {
        "question_answering": 1,
        "sentence_count": 1,
        "keyword_story": 1,
    }
    assert not any(item["task_type"] == "continuation" for item in output["train"])
