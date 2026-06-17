"""Tests for the post-run evaluation materialization benchmark."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlrmp.analysis.pipelines.gru_feedback_ablation import selected_feedback_ablation_bins_for_bank
from rlrmp.analysis.pipelines.gru_perturbation_bank import default_cs_perturbation_bank
import rlrmp.benchmarks.postrun_eval_materialization as benchmark
from rlrmp.benchmarks.postrun_eval_materialization import (
    build_parser,
    run_benchmark,
    subset_perturbation_bank,
)


def test_parser_defaults_to_local_subset_contract() -> None:
    args = build_parser().parse_args([])

    assert args.source_experiment == "020a65b"
    assert args.issue == "79d2d8b"
    assert args.n_rollout_trials == 1
    assert args.perturbation_evaluation_backend == "serial"
    assert args.worst_case_optimizer_backend == "serial"
    assert args.no_write_bulk_arrays is False


def test_parser_accepts_backend_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--worst-case-optimizer-backend",
            "serial",
        ]
    )

    assert args.perturbation_evaluation_backend == "serial"
    assert args.worst_case_optimizer_backend == "serial"


def test_run_benchmark_records_and_passes_backend_context(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}
    run = SimpleNamespace(run_id="run_a", label="run A", run_spec={"hps": {}})
    bank = {
        "bank_id": "bank",
        "perturbations": [
            {"perturbation_id": "row_a", "family": "process_epsilon_force_state_xy"}
        ],
    }

    monkeypatch.setattr(benchmark, "resolve_run_inputs", lambda **_: [run])
    monkeypatch.setattr(benchmark, "default_cs_perturbation_bank", lambda: bank)
    monkeypatch.setattr(benchmark, "selected_feedback_ablation_bins_for_bank", lambda _: {})
    monkeypatch.setattr(
        benchmark,
        "materialize_gru_standard_result",
        lambda **_: {"rows": [], "checkpoint_policy": "test"},
    )
    monkeypatch.setattr(benchmark, "write_gru_standard_result", lambda *_, **__: None)
    monkeypatch.setattr(
        benchmark,
        "materialize_gru_evaluation_diagnostics",
        lambda **_: {"runs": {"run_a": {"status_counts": {"ok": 1}}}},
    )
    monkeypatch.setattr(benchmark, "materialize_gru_pilot_figures", lambda **_: {"fig": {}})
    monkeypatch.setattr(
        benchmark,
        "materialize_gru_objective_comparator_sidecar",
        lambda **_: {"status": "materialized"},
    )
    monkeypatch.setattr(
        benchmark,
        "materialize_gru_map_error_decomposition",
        lambda **_: {"rows": []},
    )

    def fake_perturbation_bank(*_, **kwargs):
        calls["perturbation_evaluation_backend"] = kwargs["evaluation_backend"]
        return {"status_counts": {"evaluated": 1}, "perturbations": [{}]}

    def fake_worst_case(*_, **kwargs):
        calls["worst_case_optimizer_backend"] = kwargs["optimizer_backend"]
        return {"status": "evaluated"}

    monkeypatch.setattr(benchmark, "evaluate_run_perturbation_bank", fake_perturbation_bank)
    monkeypatch.setattr(benchmark, "evaluate_run_feedback_ablation", lambda *_, **__: {"rows": []})
    monkeypatch.setattr(benchmark, "audit_run_worst_case_epsilon", fake_worst_case)
    monkeypatch.setattr(benchmark, "_environment", lambda: {"jax_default_backend": "cpu"})

    payload = run_benchmark(
        perturbation_evaluation_backend="serial",
        worst_case_optimizer_backend="serial",
        output_path=tmp_path / "timing.json",
        scratch_dir=tmp_path / "scratch",
        repo_root=tmp_path,
    )

    assert payload["context"]["perturbation_evaluation_backend"] == "serial"
    assert payload["context"]["worst_case_optimizer_backend"] == "serial"
    assert calls == {
        "perturbation_evaluation_backend": "serial",
        "worst_case_optimizer_backend": "serial",
    }


@pytest.mark.parametrize("option", ["--perturbation-evaluation-backend", "--worst-case-optimizer-backend"])
def test_parser_rejects_unknown_backend(option: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([option, "unknown"])


def test_subset_bank_is_feedback_bin_compatible() -> None:
    bank = subset_perturbation_bank(default_cs_perturbation_bank(), max_rows=7)

    bins = selected_feedback_ablation_bins_for_bank(bank)
    row_ids = {row["perturbation_id"] for row in bank["perturbations"]}

    assert bank["bank_id"].startswith("benchmark_subset_of_")
    assert len(bank["perturbations"]) == 7
    assert bins["nominal"] is None
    assert any(value in row_ids for value in bins.values() if value is not None)
    assert all(value is None or value in row_ids for value in bins.values())
