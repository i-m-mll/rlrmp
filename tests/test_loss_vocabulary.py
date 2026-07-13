"""Boundary and compatibility tests for loss-objective identifiers."""

from __future__ import annotations

import ast
from pathlib import Path

from rlrmp import loss
from rlrmp import loss_vocabulary


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_science_vocabulary_depends_on_leaf_loss_vocabulary() -> None:
    source = (REPO_ROOT / "src" / "rlrmp" / "train" / "science_vocabulary.py").read_text(
        encoding="utf-8"
    )
    imported_modules = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }

    assert "rlrmp.loss_vocabulary" in imported_modules
    assert "rlrmp.loss" not in imported_modules


def test_loss_preserves_public_objective_identifiers() -> None:
    for name in loss_vocabulary.__all__:
        assert getattr(loss, name) == getattr(loss_vocabulary, name)
