"""Tests for live training diagnostics monitoring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_monitor_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "monitor_training_diagnostics.py"
    spec = importlib.util.spec_from_file_location("monitor_training_diagnostics", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_monitor_ignores_unsampled_pgd_placeholder_nans(tmp_path: Path) -> None:
    monitor = _load_monitor_module()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    np.savez(
        output_dir / "training_diagnostics.npz",
        batch_index=np.array([0, 1]),
        train_loss__total=np.array([[3.0], [2.0]]),
        validation_loss__total=np.array([[4.0], [3.0]]),
        pgd_broad_epsilon_diagnostic_sampled=np.array([False, True]),
        pgd_broad_epsilon_inner_objective_improvement=np.array([[np.nan], [1.0]]),
        pgd_broad_epsilon_inner_objective_best=np.array([[np.nan], [2.0]]),
        pgd_broad_epsilon_inner_objective_final_endpoint=np.array([[np.nan], [2.0]]),
        pgd_broad_epsilon_inner_objective_final_endpoint_gap=np.array([[np.nan], [0.0]]),
        pgd_broad_epsilon_epsilon_norm_radius_ratio_mean=np.array([[np.nan], [0.75]]),
        pgd_broad_epsilon_boundary_fraction=np.array([[np.nan], [0.0]]),
    )

    summary = monitor.summarize_output_dir(output_dir)

    assert summary["ok"] is True
    assert summary["alerts"] == []
    assert summary["latest_batch_index"] == 1
    assert summary["latest"]["pgd_broad_epsilon_inner_objective_improvement"]["mean"] == 1.0
