"""Structural guards for governed rollout-recovery routing (issue 56aad38)."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MODULE = "rlrmp.eval.output_feedback_rollout_recovery"
ACTIVE_FULL_FIT_CALLERS = (
    "scripts/materialize_output_feedback_failure_decomposition.py",
    "scripts/materialize_output_feedback_sweep_certificates.py",
)
RESULT_SCRIPTS = (
    "results/1c014e5/scripts/materialize_output_feedback_optimizer_basin_diagnostic.py",
    "results/3becdec/scripts/materialize_output_feedback_observer_error_coverage.py",
)


def _tree(relative_path: str) -> ast.Module:
    path = REPO_ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)


def _imports_from(tree: ast.Module, module: str) -> set[str]:
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def _called_names(tree: ast.Module) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_active_full_fit_callers_use_governed_manifest_custody() -> None:
    for relative_path in ACTIVE_FULL_FIT_CALLERS:
        tree = _tree(relative_path)
        canonical_imports = _imports_from(tree, CANONICAL_MODULE)
        calls = _called_names(tree)

        assert "execute_governed_output_feedback_rollout_recovery" in canonical_imports
        assert "execute_governed_output_feedback_rollout_recovery" in calls
        assert "run_output_feedback_rollout_recovery" not in canonical_imports
        assert "run_output_feedback_rollout_recovery" not in calls


def test_result_scripts_import_computational_helpers_from_canonical_module() -> None:
    expected_helpers = {
        RESULT_SCRIPTS[0]: {"adamw_optimizer_whitened", "run_output_feedback_rollout_recovery"},
        RESULT_SCRIPTS[1]: {
            "observer_error_coverage_conditions",
            "run_output_feedback_rollout_recovery",
        },
    }
    for relative_path, helpers in expected_helpers.items():
        tree = _tree(relative_path)
        assert helpers <= _imports_from(tree, CANONICAL_MODULE)
        assert "LinearOptimizationConfig" in _imports_from(
            tree, "rlrmp.analysis.math.linear_round_trip"
        )


def test_rollout_recovery_facade_is_absent() -> None:
    facade = REPO_ROOT / "src/rlrmp/analysis/pipelines/output_feedback_rollout_recovery.py"
    assert not facade.exists()

    searched_paths = (
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
        REPO_ROOT / "tests",
        REPO_ROOT / "results/1c014e5/scripts",
        REPO_ROOT / "results/3becdec/scripts",
    )
    residuals = []
    retired_module = "rlrmp.analysis.pipelines." + "output_feedback_rollout_recovery"
    for root in searched_paths:
        for path in root.rglob("*.py"):
            if retired_module in path.read_text(encoding="utf-8"):
                residuals.append(path.relative_to(REPO_ROOT).as_posix())
    assert not residuals
