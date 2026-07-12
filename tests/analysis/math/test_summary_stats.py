from __future__ import annotations

from pathlib import Path

import numpy as np

from rlrmp.analysis.math.summary_stats import summary_stats


def test_summary_stats_default_schema_matches_pipeline_manifests() -> None:
    summary = summary_stats([[1.0, 2.0], [3.0, 4.0]])

    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert summary["std"] == np.std([1.0, 2.0, 3.0, 4.0])
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
    assert summary["p50"] == 2.5
    assert summary["p95"] == np.quantile([1.0, 2.0, 3.0, 4.0], 0.95)


def test_summary_stats_preserves_worst_case_epsilon_schema() -> None:
    summary = summary_stats([1.0, 2.0, 3.0], count_key="n", quantiles=())

    assert summary == {
        "n": 3,
        "mean": 2.0,
        "std": np.std([1.0, 2.0, 3.0]),
        "min": 1.0,
        "max": 3.0,
    }


def test_summary_stats_preserves_evaluation_diagnostic_quantiles() -> None:
    summary = summary_stats([0.0, 10.0], quantiles=(0.05, 0.5, 0.95))

    assert summary["p05"] == 0.5
    assert summary["p50"] == 5.0
    assert summary["p95"] == 9.5


def test_summary_stats_empty_inputs_omit_quantiles() -> None:
    summary = summary_stats([], quantiles=(0.05, 0.5, 0.95))

    assert set(summary) == {"count", "mean", "std", "min", "max"}
    assert summary["count"] == 0
    assert np.isnan(summary["mean"])


def test_dup_0173_pipeline_modules_import_shared_helper() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    module_paths = [
        "src/rlrmp/analysis/pipelines/gru_worst_case_epsilon_audit.py",
        "src/rlrmp/eval/gru_diagnostics.py",
        "src/rlrmp/analysis/pipelines/gru_feedback_ablation.py",
        "src/rlrmp/analysis/pipelines/gru_perturbation_bank.py",
        "src/rlrmp/analysis/pipelines/objective_comparator.py",
    ]

    for relpath in module_paths:
        source = (repo_root / relpath).read_text()
        assert "from rlrmp.analysis.math.summary_stats import " in source
        assert "def _summary_stats(" not in source
