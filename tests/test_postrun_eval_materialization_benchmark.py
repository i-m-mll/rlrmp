"""Tests for the post-run evaluation materialization benchmark."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from rlrmp.analysis.pipelines.gru_feedback_ablation import selected_feedback_ablation_bins_for_bank
from rlrmp.eval.perturbation_bank import default_cs_perturbation_bank
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
    assert args.warm_replay is False
    assert args.no_write_bulk_arrays is False


def test_parser_accepts_backend_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--worst-case-optimizer-backend",
            "serial",
            "--warm-replay",
        ]
    )

    assert args.perturbation_evaluation_backend == "serial"
    assert args.worst_case_optimizer_backend == "serial"
    assert args.warm_replay is True


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
    monkeypatch.setattr(
        benchmark,
        "build_validation_checkpoint_selection_manifest",
        lambda **_: SimpleNamespace(id="checkpoint-selection-fixture"),
    )
    monkeypatch.setattr(benchmark, "default_cs_perturbation_bank", lambda: bank)
    monkeypatch.setattr(benchmark, "selected_feedback_ablation_bins_for_bank", lambda _: {})
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

    def fake_feedback_pipeline(**kwargs):
        calls["feedback_scope"] = kwargs["scope"]
        return SimpleNamespace(payload={"status_counts": {}, "runs": {}})

    monkeypatch.setattr(benchmark, "evaluate_run_perturbation_bank", fake_perturbation_bank)
    monkeypatch.setattr(
        benchmark,
        "execute_feedback_ablation_pipeline",
        fake_feedback_pipeline,
    )
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
    assert payload["schema_version"] == "rlrmp.postrun_eval_materialization_benchmark.v2"
    assert payload["process_timing"]["process_startup_elapsed_s"]["status"] == "not_measured"
    assert payload["setup_timing"]["elapsed_s"] >= 0.0
    assert payload["context"]["warm_replay"]["enabled"] is False
    assert payload["context"]["warm_replay"]["reason_if_disabled"]
    assert payload["report_serialization_timing"]["elapsed_s"] >= 0.0
    assert calls == {
        "perturbation_evaluation_backend": "serial",
        "feedback_scope": "postrun_eval_materialization_benchmark",
        "worst_case_optimizer_backend": "serial",
    }
    assert payload["bundles"]
    for bundle in payload["bundles"]:
        assert bundle["cold_call_elapsed_s"] >= 0.0
        assert bundle["cold_ready_block_s"] >= 0.0
        assert bundle["summary_elapsed_s"] >= 0.0
        assert bundle["cold_ready_blocked_leaves"] == 0
        assert bundle["cold_ready_block_note"] == "no JAX leaves with block_until_ready"
        assert bundle["warm_replay_status"] == "disabled"
        assert bundle["warm_call_elapsed_s"] is None
        assert bundle["xla_compile_execution_split"]["status"] == "not_measured"
        assert "not pure" in bundle["cold_call_definition"]


def test_time_bundle_blocks_jax_leaves_before_summarizing() -> None:
    seen_ready = {"value": False}

    def summarize(result):
        seen_ready["value"] = result["array"].is_ready()
        return {"shape": tuple(result["array"].shape)}

    result = benchmark._time_bundle(
        "jax_bundle",
        lambda: {"array": jnp.arange(4)},
        summarize=summarize,
    )

    assert result.status == "ok"
    assert result.cold_ready_blocked_leaves == 1
    assert result.cold_ready_block_note == "blocked JAX leaves with block_until_ready"
    assert result.cold_call_elapsed_s >= 0.0
    assert result.cold_ready_block_s >= 0.0
    assert result.summary_elapsed_s >= 0.0
    assert result.summary == {"shape": (4,)}
    assert seen_ready["value"] is True
    assert result.warm_replay_status == "disabled"


def test_time_bundle_explains_when_no_jax_leaves_are_present() -> None:
    result = benchmark._time_bundle("plain_bundle", lambda: {"rows": 3})

    assert result.status == "ok"
    assert result.summary == {"rows": 3}
    assert result.cold_ready_blocked_leaves == 0
    assert result.cold_ready_block_s == 0.0
    assert result.cold_ready_block_note == "no JAX leaves with block_until_ready"
    assert result.output_write_mode == "not_applicable"


def test_time_bundle_measures_separated_output_write_without_changing_summary(tmp_path) -> None:
    output_path = tmp_path / "cold" / "summary.txt"

    def write_output(result):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(result["rows"]), encoding="utf-8")

    result = benchmark._time_bundle(
        "writer_bundle",
        lambda: {"rows": 3},
        summarize=lambda value: {"rows": value["rows"]},
        output_writer=write_output,
        output_write_note="test output write",
    )

    assert result.status == "ok"
    assert result.summary == {"rows": 3}
    assert output_path.read_text(encoding="utf-8") == "3"
    assert result.output_write_mode == "separate_measured"
    assert result.output_write_elapsed_s is not None
    assert result.output_write_elapsed_s >= 0.0


def test_time_bundle_warm_replay_uses_separate_writer(tmp_path) -> None:
    cold_output = tmp_path / "cold.txt"
    warm_output = tmp_path / "warm.txt"

    def cold_call():
        return {"value": jnp.arange(2)}

    def warm_call():
        return {"value": jnp.arange(3)}

    def write_cold(result):
        cold_output.write_text(str(tuple(result["value"].shape)), encoding="utf-8")

    def write_warm(result):
        warm_output.write_text(str(tuple(result["value"].shape)), encoding="utf-8")

    result = benchmark._time_bundle(
        "warm_bundle",
        cold_call,
        summarize=lambda value: {"shape": tuple(value["value"].shape)},
        output_writer=write_cold,
        warm_replay=True,
        warm_fn=warm_call,
        warm_output_writer=write_warm,
    )

    assert result.status == "ok"
    assert result.summary == {"shape": (2,)}
    assert cold_output.read_text(encoding="utf-8") == "(2,)"
    assert warm_output.read_text(encoding="utf-8") == "(3,)"
    assert result.warm_replay_status == "ok"
    assert result.warm_call_elapsed_s is not None
    assert result.warm_call_elapsed_s >= 0.0
    assert result.warm_ready_blocked_leaves == 1


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
