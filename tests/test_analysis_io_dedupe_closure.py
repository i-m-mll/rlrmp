"""Behavior and structural guards for the analysis-I/O dedupe closure."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _function(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(node: ast.FunctionDef) -> set[str]:
    names = set()
    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
        if isinstance(call.func, ast.Name):
            names.add(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            names.add(call.func.attr)
    return names


def _loc(node: ast.FunctionDef) -> int:
    assert node.end_lineno is not None
    return node.end_lineno - node.lineno + 1


def test_retired_multi_cell_pipeline_cannot_reaccrete() -> None:
    for path in (
        "src/rlrmp/analysis/multi_cell_driver.py",
        "results/3702f54/scripts/analyse_pregomatrix.py",
        "results/b399efc/scripts/analyse_movement_ramp_matrix.py",
    ):
        assert not (REPO_ROOT / path).exists()


def test_path_and_ensemble_residuals_stay_canonical() -> None:
    for path, name in (
        (
            "results/3c5836c/scripts/materialize_frozen_finite_policy_audit.py",
            "_repo_rel",
        ),
    ):
        node = _function(path, name)
        assert _loc(node) <= 2
        assert "portable_repo_path" in _calls(node)

    declarative = ast.parse(
        (REPO_ROOT / "src/rlrmp/analysis/declarative_materialization.py").read_text(
            encoding="utf-8"
        )
    )
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "_repo_relative"
        for node in ast.walk(declarative)
    )
    assert not (
        REPO_ROOT / "tests/analysis/pipelines/test_objective_comparator.py"
    ).exists()
    canonical_objective_tests = (
        REPO_ROOT / "tests/analysis/test_objective_comparator.py"
    ).read_text(encoding="utf-8")
    assert "build_objective_comparator_sidecar_from_cached" in canonical_objective_tests
