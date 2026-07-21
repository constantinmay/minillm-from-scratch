import pytest

from demo_compare import build_prompt, parse_models


def test_demo_builds_current_plain_text_templates():
    assert build_prompt("continuation", "A story.") == (
        "Instruction: Continue the story.\nInput: A story.\nResponse:"
    )
    qa = build_prompt("qa", "Lily ran.", question="Who ran?")
    assert "Question: Who ran?" in qa
    assert qa.endswith("Response:")
    keyword = build_prompt("keywords", "A story.", keywords=["bird", "home"])
    assert '"bird"' in keyword and '"home"' in keyword
    sentence = build_prompt("sentence_count", "A story.", sentence_count=2)
    assert "exactly 2 sentences" in sentence


def test_demo_rejects_missing_task_arguments():
    with pytest.raises(ValueError):
        build_prompt("qa", "A story.")
    with pytest.raises(ValueError):
        build_prompt("keywords", "A story.")
    with pytest.raises(ValueError):
        build_prompt("sentence_count", "A story.", sentence_count=0)


def test_demo_parses_named_models():
    models = parse_models(["A=one.pt", "B=two.pt"])
    assert list(models.items()) == [("A", "one.pt"), ("B", "two.pt")]
