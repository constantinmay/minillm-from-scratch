"""Structural checks for the GitHub beginner tutorial."""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "docs" / "tutorial"
LANGUAGES = {
    "notebooks": "### 动手验证",
    "notebooks_en": "### Hands-on check",
}

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
    for directory, bridge_heading in LANGUAGES.items():
        for name in NOTEBOOKS:
            path = TUTORIAL / directory / name
            notebook = json.loads(path.read_text(encoding="utf-8"))
            assert notebook["nbformat"] == 4
            assert notebook["cells"]
            markdown = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "markdown"
            )
            code_cells = [
                cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            ]
            assert "$$" in markdown, f"{path} needs a displayed equation"
            assert len(code_cells) >= 3, f"{path} needs interleaved runnable examples"
            assert "## 可运行验证" not in markdown
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    assert index > 0
                    previous = notebook["cells"][index - 1]
                    assert previous["cell_type"] == "markdown"
                    explanation = "".join(previous["source"])
                    assert explanation.startswith(bridge_heading)
                    assert len(explanation.splitlines()) >= 3
                    ast.parse("".join(cell["source"]), filename=str(path))


def test_bilingual_notebooks_share_identical_code_cells():
    for name in NOTEBOOKS:
        code_by_language = []
        for directory in LANGUAGES:
            notebook = json.loads(
                (TUTORIAL / directory / name).read_text(encoding="utf-8")
            )
            code_by_language.append(
                [
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                ]
            )
        assert code_by_language[0] == code_by_language[1], name


def test_no_duplicate_markdown_chapters_remain():
    assert not list(TUTORIAL.glob("[0-9][0-9]_*.md"))


def test_documentation_index_links_resolve_locally():
    indexes = (
        ROOT / "README.md",
        ROOT / "README_zh.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "README_en.md",
        TUTORIAL / "README.md",
        TUTORIAL / "README_en.md",
    )
    for index in indexes:
        text = index.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            path = (index.parent / target.split("#", 1)[0]).resolve()
            assert path.exists(), f"broken link in {index}: {target}"


def test_notebook_local_links_resolve():
    for directory in LANGUAGES:
        for name in NOTEBOOKS:
            notebook_path = TUTORIAL / directory / name
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            markdown = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "markdown"
            )
            for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown):
                if target.startswith(("http://", "https://", "#")):
                    continue
                path = (notebook_path.parent / target.split("#", 1)[0]).resolve()
                assert path.exists(), f"broken link in {notebook_path}: {target}"
