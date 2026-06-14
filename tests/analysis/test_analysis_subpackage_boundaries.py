"""Layering checks for the analysis subpackage split."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

MATH_DIR = Path(__file__).parents[2] / "src" / "rlrmp" / "analysis" / "math"


def test_analysis_compatibility_shims_are_absent() -> None:
    """Representative old root paths are no longer provided as shim modules."""

    assert importlib.util.find_spec("rlrmp.analysis.rerun_metadata") is None
    assert importlib.util.find_spec("rlrmp.analysis.bridge_contracts") is None


def test_canonical_analysis_modules_are_importable() -> None:
    """Representative canonical paths remain importable after shim removal."""

    assert importlib.util.find_spec("rlrmp.analysis.math.rerun_metadata") is not None
    assert importlib.util.find_spec("rlrmp.analysis.pipelines.bridge_contracts") is not None


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
