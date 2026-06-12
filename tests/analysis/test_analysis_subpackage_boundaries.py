"""Compatibility and layering checks for the analysis subpackage split."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


MATH_DIR = Path(__file__).parents[2] / "src" / "rlrmp" / "analysis" / "math"


@pytest.mark.parametrize(
    ("old_path", "new_path"),
    [
        ("rlrmp.analysis.rerun_metadata", "rlrmp.analysis.math.rerun_metadata"),
        ("rlrmp.analysis.bridge_contracts", "rlrmp.analysis.pipelines.bridge_contracts"),
    ],
)
def test_old_analysis_import_paths_alias_canonical_modules(old_path: str, new_path: str) -> None:
    """Representative old paths remain import-compatible after the split."""

    old_module = importlib.import_module(old_path)
    new_module = importlib.import_module(new_path)

    assert old_module is new_module


def test_analysis_math_layer_does_not_import_pipelines_layer() -> None:
    """Lower-level math modules must not depend on orchestration pipelines."""

    offenders: list[str] = []

    for path in sorted(MATH_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "rlrmp.analysis.pipelines" or alias.name.startswith(
                        "rlrmp.analysis.pipelines."
                    ):
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "rlrmp.analysis.pipelines" or module.startswith(
                    "rlrmp.analysis.pipelines."
                ):
                    offenders.append(f"{path.name}: from {module} import ...")
                elif module == "rlrmp.analysis" and any(
                    alias.name == "pipelines" for alias in node.names
                ):
                    offenders.append(f"{path.name}: from rlrmp.analysis import pipelines")
                elif node.level >= 2 and (
                    module == "pipelines" or module.startswith("pipelines.")
                ):
                    dots = "." * node.level
                    offenders.append(f"{path.name}: from {dots}{module} import ...")

    assert offenders == []
