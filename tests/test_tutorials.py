"""Structural checks for the GitHub beginner tutorial."""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "docs" / "tutorial"

NOTEBOOKS = [
    "01_tokenizer_and_lm.ipynb",
    "02_transformer.ipynb",
    "03_pretraining.ipynb",
    "04_sft.ipynb",
    "05_alignment.ipynb",
    "06_evaluation.ipynb",
    "07_reproduce.ipynb",
]


def test_tutorial_notebooks_combine_explanation_equations_and_code():
    for name in NOTEBOOKS:
        path = TUTORIAL / "notebooks" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert "$$" in markdown, f"{name} needs at least one displayed equation"
        assert "```" in markdown, f"{name} needs formal code/command examples"
        assert len(code_cells) >= 3, f"{name} needs interleaved runnable examples"
        assert "## 可运行验证" not in markdown, f"{name} must not collect code at the end"
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert index > 0
                previous = notebook["cells"][index - 1]
                assert previous["cell_type"] == "markdown"
                explanation = "".join(previous["source"])
                assert explanation.startswith("### 动手验证")
                assert len(explanation.splitlines()) >= 3
                ast.parse("".join(cell["source"]), filename=str(path))


def test_no_duplicate_markdown_chapters_remain():
    assert not list(TUTORIAL.glob("[0-9][0-9]_*.md"))


def test_documentation_index_links_resolve_locally():
    for index in (ROOT / "docs" / "README.md", TUTORIAL / "README.md"):
        text = index.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            path = (index.parent / target.split("#", 1)[0]).resolve()
            assert path.exists(), f"broken link in {index}: {target}"
