"""Keep the beginner notebooks executable as the production code evolves."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = [
    ROOT / "docs" / "tutorial" / "notebooks",
    ROOT / "docs" / "tutorial" / "notebooks_en",
]
NOTEBOOK_PATHS = [
    path for directory in NOTEBOOK_DIRS for path in sorted(directory.glob("*.ipynb"))
]


@pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS)
def test_tutorial_notebook_code_cells(notebook_path, monkeypatch):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__tutorial_test__"}
    monkeypatch.chdir(ROOT)

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        code = compile(source, f"{notebook_path.name}:cell-{index}", "exec")
        exec(code, namespace)
