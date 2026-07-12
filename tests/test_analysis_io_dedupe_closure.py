"""Behavior and structural guards for the analysis-I/O dedupe closure."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rlrmp.analysis.math.trial_alignment import (
    align_trials,
    pooled_trial_mean_with_band,
    replicate_mean_curves,
)
from rlrmp.analysis.multi_cell_driver import (
    args_namespace,
    legacy_task_trainer_history_skeleton,
)
from rlrmp.viz.figures import build_forward_velocity_figure, build_hold_drift_figure


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


def _kinematics() -> dict[str, np.ndarray]:
    velocity = np.asarray(
        [
            [[0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.5, 2.5]],
            [[0.0, 2.0, 4.0, 6.0], [0.0, 1.0, 3.0, 5.0]],
        ]
    )
    return {
        "forward_vel_profile": velocity,
        "pos_forward_profile": velocity * 0.001,
        "go_idx": np.asarray([1, 2]),
    }


def test_multi_cell_figure_builders_preserve_reductions() -> None:
    cell_kms = {"cell": _kinematics()}
    common = {
        "labels": ("cell",),
        "display_names": {"cell": "Cell"},
        "colors": {"cell": "#2563eb"},
        "title": "title",
        "width": 900,
        "height_per_cell": 200,
        "vertical_spacing": 0.06,
    }

    replicate = build_forward_velocity_figure(
        cell_kms,
        trace_mode="replicate",
        **common,
    )
    aligned, center = align_trials(
        cell_kms["cell"]["forward_vel_profile"],
        cell_kms["cell"]["go_idx"],
    )
    expected_curves, window = replicate_mean_curves(aligned)
    expected_time = ((np.arange(aligned.shape[-1]) - center) * 0.01)[window]
    assert len(replicate.data) == 2
    np.testing.assert_allclose(replicate.data[0].x, expected_time)
    np.testing.assert_allclose(replicate.data[0].y, expected_curves[0])

    pooled = build_forward_velocity_figure(cell_kms, trace_mode="pooled", **common)
    mean, lower, upper, window = pooled_trial_mean_with_band(aligned, band="sd")
    expected_time = ((np.arange(aligned.shape[-1]) - center) * 0.01)[window]
    assert len(pooled.data) == 3
    np.testing.assert_allclose(pooled.data[0].y, upper)
    np.testing.assert_allclose(pooled.data[1].y, lower)
    np.testing.assert_allclose(pooled.data[2].x, expected_time)
    np.testing.assert_allclose(pooled.data[2].y, mean)

    hold = build_hold_drift_figure(
        cell_kms,
        trace_mode="replicate",
        pre_go_window_steps=1,
        **common,
    )
    aligned_position, center = align_trials(
        cell_kms["cell"]["pos_forward_profile"],
        cell_kms["cell"]["go_idx"],
    )
    curves, window = replicate_mean_curves(aligned_position)
    time = ((np.arange(aligned_position.shape[-1]) - center) * 0.01)[window]
    keep = (time >= -0.01) & (time <= 0.0)
    np.testing.assert_allclose(hold.data[0].x, time[keep])
    np.testing.assert_allclose(hold.data[0].y, curves[0, keep] * 1000.0)


def test_profiled_args_namespace_preserves_training_defaults() -> None:
    lit = args_namespace(
        profile="lit_replication",
        n_warmup_batches=12,
        n_replicates=5,
        overrides={"nn_output_jerk": 9.0},
    )
    assert lit.n_warmup_batches == 12
    assert lit.n_replicates == 5
    assert lit.nn_output_jerk == 9.0
    assert lit.effector_hold_pos == 1.0
    assert lit.nn_hidden_derivative == 0.001

def test_historical_training_history_reader_fails_closed_on_current_feedbax() -> None:
    if importlib.util.find_spec("feedbax.train") is not None:
        pytest.skip("producing-era Feedbax API is available")
    with pytest.raises(RuntimeError, match="producing Feedbax checkout"):
        legacy_task_trainer_history_skeleton(None)

    for path in (
        "results/3702f54/scripts/analyse_pregomatrix.py",
        "results/b399efc/scripts/analyse_movement_ramp_matrix.py",
    ):
        assert "from feedbax.train import" not in (REPO_ROOT / path).read_text(encoding="utf-8")


def test_multi_cell_members_are_thin_canonical_adapters() -> None:
    figure_members = tuple(
        (path, name, canonical)
        for path in (
            "results/b399efc/scripts/analyse_movement_ramp_matrix.py",
        )
        for name, canonical in (
            ("make_forward_velocity_profile_figure", "canonical_forward_velocity_figure"),
            ("make_hold_drift_figure", "canonical_hold_drift_figure"),
        )
    )
    for path, name, canonical in figure_members:
        node = _function(path, name)
        assert _loc(node) <= 30
        assert canonical in _calls(node)


def test_args_namespace_members_cannot_reaccrete() -> None:
    members = (
        ("results/b399efc/scripts/analyse_movement_ramp_matrix.py", "_make_args_namespace"),
        ("results/3702f54/scripts/analyse_pregomatrix.py", "_make_args_namespace"),
    )
    for path, name in members:
        node = _function(path, name)
        assert _loc(node) <= 12
        assert "args_namespace" in _calls(node)


def test_path_and_ensemble_residuals_stay_canonical() -> None:
    for path, name in (
        (
            "results/3becdec/scripts/materialize_output_feedback_observer_error_coverage.py",
            "_repo_relative",
        ),
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
