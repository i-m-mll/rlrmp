"""Tests for the post-run evaluation materialization benchmark."""

from __future__ import annotations

from rlrmp.analysis.pipelines.gru_feedback_ablation import selected_feedback_ablation_bins_for_bank
from rlrmp.analysis.pipelines.gru_perturbation_bank import default_cs_perturbation_bank
from rlrmp.benchmarks.postrun_eval_materialization import build_parser, subset_perturbation_bank


def test_parser_defaults_to_local_subset_contract() -> None:
    args = build_parser().parse_args([])

    assert args.source_experiment == "020a65b"
    assert args.issue == "79d2d8b"
    assert args.n_rollout_trials == 1
    assert args.no_write_bulk_arrays is False


def test_subset_bank_is_feedback_bin_compatible() -> None:
    bank = subset_perturbation_bank(default_cs_perturbation_bank(), max_rows=7)

    bins = selected_feedback_ablation_bins_for_bank(bank)
    row_ids = {row["perturbation_id"] for row in bank["perturbations"]}

    assert bank["bank_id"].startswith("benchmark_subset_of_")
    assert len(bank["perturbations"]) == 7
    assert bins["nominal"] is None
    assert any(value in row_ids for value in bins.values() if value is not None)
    assert all(value is None or value in row_ids for value in bins.values())
