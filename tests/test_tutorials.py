"""Structural checks for the GitHub beginner tutorial."""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "docs" / "tutorial"

CHAPTERS = [
    "01_tokenizer_and_lm.md",
    "02_transformer.md",
    "03_pretraining.md",
    "04_sft.md",
    "05_alignment.md",
    "06_evaluation.md",
    "07_reproduce.md",
]

NOTEBOOKS = [
    "01_tokenizer_and_loss.ipynb",
    "02_transformer_forward.ipynb",
    "03_sft_and_dpo.ipynb",
    "04_evaluation_metrics.ipynb",
]


def test_all_tutorial_chapters_exist_and_have_equations_and_code():
    for name in CHAPTERS:
        text = (TUTORIAL / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} needs at least one displayed equation"
        assert "```" in text, f"{name} needs at least one code/example block"


def test_tutorial_notebooks_are_valid_json_and_code_compiles():
    for name in NOTEBOOKS:
        path = TUTORIAL / "notebooks" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=str(path))


def test_documentation_index_links_resolve_locally():
    for index in (ROOT / "docs" / "README.md", TUTORIAL / "README.md"):
        text = index.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            path = (index.parent / target.split("#", 1)[0]).resolve()
            assert path.exists(), f"broken link in {index}: {target}"
