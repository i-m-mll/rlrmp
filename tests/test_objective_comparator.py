"""Tests for objective-comparator sidecars."""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np

import rlrmp.analysis.objective_comparator as objective_comparator
from rlrmp.analysis.objective_comparator import (
    SCHEMA_VERSION,
    ExtLQGCostDecomposition,
    SharedRolloutBank,
    build_objective_comparator_sidecar,
    extlqg_x0_only_sanity_check,
    load_run_objective_metadata,
    materialize_gru_objective_comparator_sidecar,
    render_objective_comparator_markdown,
    shared_full_qrf_cost_summary,
    write_objective_comparator_sidecar,
)


def _checkpoint_selection() -> dict[str, object]:
    return {
        "schema_version": "rlrmp.validation_selected_gru_checkpoints.v1",
        "selection_policy": "validation-selected test policy",
        "runs": {
            "run_b": [
                {
                    "replicate": 0,
                    "scoring_validation_objective": 44.0,
                    "best_logged_validation_objective": 43.0,
                }
            ],
            "run_a": [
                {
                    "replicate": 0,
                    "scoring_validation_objective": 10.0,
                    "best_logged_validation_objective": 9.0,
                },
                {
                    "replicate": 1,
                    "scoring_validation_objective": 14.0,
                    "best_logged_validation_objective": 11.0,
                },
            ],
        },
    }


def _full_qrf_run_metadata() -> dict[str, object]:
    return {
        "status": "available",
        "loss_objective": "full_analytical_qrf",
        "objective_profile": "full_analytical_qrf",
        "full_qrf_lens": {
            "status": "available",
            "active_terms": ["control_r", "state_running_q", "terminal_q_f"],
        },
    }


def test_extlqg_decomposition_reports_component_sum_and_declared_total() -> None:
    decomposition = ExtLQGCostDecomposition(
        deterministic_initial_state=4.0,
        initial_covariance_trace=3.0,
        accumulated_noise_scalar=2.0,
        total_expected_cost=9.5,
        provenance="unit-test",
    )

    payload = decomposition.to_json()

    assert payload["component_sum"] == 9.0
    assert payload["total_expected_cost"] == 9.5
    assert payload["component_sum_delta"] == 0.5
    assert payload["comparable_scalar"] == 4.0
    assert payload["comparable_scalar_lens"] == "extlqg_deterministic_initial_state_full_qrf"


def test_build_objective_comparator_sidecar_uses_deterministic_comparator_lens() -> None:
    shared_rollout = {
        "status": "available",
        "lens": "shared_rollout_full_qrf",
        "interpretation": "stress_test_only",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "bank": {
            "bank_id": "unit-bank",
            "seed": 7,
            "n_trials": 2,
        },
        "noise_comparability": {
            "limitation": "unit limitation",
        },
        "runs": {
            "run_a": {
                "status": "available",
                "gru_vs_extlqg": {"terms": {"total": {"ratio_to_extlqg": 1.25}}},
            }
        },
        "standard_split_bank_comparator": {
            "status": "available",
            "lens": "standard_split_rollout_bank_full_qrf",
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
            "lenses": {
                "deterministic_nominal": {"status": "available"},
                "x0_only": {"status": "available"},
                "epsilon_only": {"status": "available"},
                "x0_plus_epsilon": {
                    "status": "available",
                    "interpretation": "stress_test_only",
                },
            },
            "extlqg_x0_only_sanity_check": {
                "status": "pass",
                "expected_cost_wording_allowed": True,
            },
            "runs": {
                "run_a": {
                    "status": "available",
                    "lenses": {
                        "x0_only": {
                            "status": "available",
                            "gru_vs_extlqg": {
                                "terms": {"total": {"ratio_to_extlqg": 1.1}}
                            },
                        },
                    },
                },
            },
        },
    }
    sidecar = build_objective_comparator_sidecar(
        issue="abc1234",
        source_manifest="source.json",
        checkpoint_selection=_checkpoint_selection(),
        extlqg=ExtLQGCostDecomposition(
            deterministic_initial_state=12.0,
            initial_covariance_trace=30.0,
            accumulated_noise_scalar=2.0,
            total_expected_cost=44.0,
            provenance="unit-test",
        ),
        scope="unit scope",
        generated_by="unit",
        run_metadata_by_id={
            "run_a": _full_qrf_run_metadata(),
            "run_b": _full_qrf_run_metadata(),
        },
        shared_rollout_comparator=shared_rollout,
    )

    assert sidecar["schema_version"] == SCHEMA_VERSION
    assert sidecar["extlqg_decomposition"]["total_expected_cost"] == 44.0
    assert sidecar["same_noise_bank_monte_carlo"]["status"] == "available_with_limitations"
    assert sidecar["same_noise_bank_monte_carlo"]["lens"] == "shared_rollout_full_qrf"
    assert sidecar["per_term_realized_scoring"]["status"] == "not_implemented"
    assert (
        sidecar["objective_lenses"]["extlqg_covariance_inclusive_expected_cost"]["noise_bank"]
        == "analytical_covariance_expectation_not_realized_validation_bank"
    )

    first_row = sidecar["rows"][0]
    assert first_row["run_id"] == "run_a"
    assert first_row["comparability"]["status"] == "comparable_deterministic_full_qrf"
    assert first_row["gru_mean_selected_validation_full_qrf"] == 12.0
    assert first_row["selected_to_extlqg_deterministic_ratio"] == 1.0
    assert first_row["selected_to_extlqg_total_ratio_not_apples_to_apples"] == 12.0 / 44.0
    assert first_row["extlqg_comparable_lens"] == "extlqg_deterministic_initial_state_full_qrf"
    assert first_row["per_term_realized_scoring"]["status"] == "not_implemented"
    assert sidecar["shared_rollout_comparator"]["status"] == "available"
    assert sidecar["shared_rollout_comparator"]["interpretation"] == "stress_test_only"
    assert sidecar["standard_split_bank_comparator"]["status"] == "available"
    assert (
        sidecar["standard_split_bank_comparator"]["lenses"]["x0_plus_epsilon"][
            "interpretation"
        ]
        == "stress_test_only"
    )
    assert (
        sidecar["standard_split_bank_comparator"]["extlqg_x0_only_sanity_check"][
            "expected_cost_wording_allowed"
        ]
        is True
    )
    assert first_row["shared_rollout_comparator"]["status"] == "available"
    assert first_row["standard_split_bank_comparator"]["status"] == "available"
    assert sidecar["rows"][1]["shared_rollout_comparator"]["status"] == "not_available"


def test_shared_rollout_bank_serializes_declared_shared_channels() -> None:
    bank = SharedRolloutBank(
        bank_id="unit-bank",
        seed=123,
        initial_states=np.zeros((3, 48), dtype=np.float64),
        process_epsilon=np.zeros((3, 60, 8), dtype=np.float64),
        initial_covariance=0.01,
    )

    payload = bank.to_json()

    assert payload["n_trials"] == 3
    assert payload["initial_state"]["status"] == "shared"
    assert payload["process_load_epsilon"]["status"] == "shared"
    assert payload["sensory_noise"]["status"] == "not_shared"
    assert payload["command_or_motor_noise"]["status"] == "not_shared"


def test_shared_full_qrf_cost_summary_decomposes_zero_rollout() -> None:
    states = np.zeros((2, 60, 48), dtype=np.float64)
    commands = np.zeros((2, 60, 2), dtype=np.float64)
    initial_states = np.zeros((2, 48), dtype=np.float64)

    summary = shared_full_qrf_cost_summary(
        states=states,
        commands=commands,
        initial_states=initial_states,
    )

    total = summary["total"]["mean"]
    term_sum = sum(
        summary[key]["mean"]
        for key in (
            "running_state",
            "terminal_state",
            "command_control",
            "force_filter_state",
            "disturbance_integrator_state",
        )
    )
    assert summary["status"] == "available"
    assert total == term_sum
    assert summary["command_control"]["mean"] == 0.0
    assert summary["total"]["shape"] == [2]


def test_extlqg_x0_only_sanity_check_reports_pass_and_warning() -> None:
    extlqg = ExtLQGCostDecomposition(
        deterministic_initial_state=10.0,
        initial_covariance_trace=2.0,
        accumulated_noise_scalar=7.0,
        provenance="unit-test",
    )
    passing = extlqg_x0_only_sanity_check(
        x0_only_cost={"total": {"mean": 12.1}},
        extlqg=extlqg,
        relative_tolerance=0.02,
    )
    warning = extlqg_x0_only_sanity_check(
        x0_only_cost={"total": {"mean": 15.0}},
        extlqg=extlqg,
        relative_tolerance=0.02,
    )

    assert passing["status"] == "pass"
    assert passing["expected_deterministic_plus_initial_covariance_trace"] == 12.0
    assert passing["expected_cost_wording_allowed"] is True
    assert warning["status"] == "warning"
    assert warning["expected_cost_wording_allowed"] is False


def test_build_objective_comparator_sidecar_marks_partial_rows_not_comparable() -> None:
    sidecar = build_objective_comparator_sidecar(
        issue="abc1234",
        source_manifest="source.json",
        checkpoint_selection=_checkpoint_selection(),
        extlqg=ExtLQGCostDecomposition(
            deterministic_initial_state=12.0,
            initial_covariance_trace=30.0,
            accumulated_noise_scalar=2.0,
            total_expected_cost=44.0,
            provenance="unit-test",
        ),
        scope="unit scope",
        generated_by="unit",
        run_metadata_by_id={
            "run_a": {
                "status": "available",
                "loss_objective": "partial_net_output_force_filter",
                "objective_profile": "partial_net_output_force_filter",
            },
            "run_b": _full_qrf_run_metadata(),
        },
    )

    first_row = sidecar["rows"][0]

    assert first_row["comparability"]["status"] == "not_comparable"
    assert first_row["selected_to_extlqg_deterministic_ratio"] is None
    assert first_row["selected_to_extlqg_total_ratio_not_apples_to_apples"] is None
    assert "must not be inferred" in first_row["comparability"]["reason"]


def test_write_objective_comparator_sidecar_serializes_json_and_markdown(tmp_path) -> None:
    sidecar = build_objective_comparator_sidecar(
        issue="abc1234",
        source_manifest="source.json",
        checkpoint_selection=_checkpoint_selection(),
        extlqg=ExtLQGCostDecomposition(
            deterministic_initial_state=12.0,
            initial_covariance_trace=30.0,
            accumulated_noise_scalar=2.0,
            total_expected_cost=44.0,
            provenance="unit-test",
        ),
        scope="unit scope",
        generated_by="unit",
        run_metadata_by_id={
            "run_a": _full_qrf_run_metadata(),
            "run_b": _full_qrf_run_metadata(),
        },
    )
    json_path = tmp_path / "sidecar.json"
    markdown_path = tmp_path / "sidecar.md"

    write_objective_comparator_sidecar(
        sidecar,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert reloaded["schema_version"] == SCHEMA_VERSION
    assert render_objective_comparator_markdown(sidecar) == markdown
    assert "Scope: unit scope." in markdown
    assert "not directly comparable to GRU validation values" in markdown
    assert "selected/total" in markdown
    assert "same-noise-bank Monte Carlo" in markdown
    assert "Per-term realized scoring" in markdown


def test_load_run_objective_metadata_extracts_full_qrf_contract(tmp_path: Path) -> None:
    run_spec_path = tmp_path / "run.json"
    run_spec_path.write_text(
        json.dumps(
            {
                "loss_objective": "full_analytical_qrf",
                "loss_summary": {
                    "objective_profile": "full_analytical_qrf",
                    "active_cs_terms": {
                        "state_running_q": {},
                        "terminal_q_f": {},
                        "control_r": {},
                    },
                    "force_filter_state_cost": "included_via_Q_entries_4_5_each_delay_block",
                    "disturbance_integrator_state_cost": (
                        "included_via_Q_entries_6_7_each_delay_block"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = load_run_objective_metadata(run_spec_path)

    assert metadata["status"] == "available"
    assert metadata["loss_objective"] == "full_analytical_qrf"
    assert metadata["full_qrf_lens"]["status"] == "available"
    assert metadata["full_qrf_lens"]["active_terms"] == [
        "control_r",
        "state_running_q",
        "terminal_q_f",
    ]


def test_materialize_gru_objective_comparator_sidecar_uses_validation_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "results" / "abc1234" / "notes" / "standard_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    output_json = tmp_path / "results" / "abc1234" / "notes" / "objective.json"
    output_md = tmp_path / "results" / "abc1234" / "notes" / "objective.md"

    monkeypatch.setattr(
        objective_comparator,
        "compute_default_extlqg_cost_decomposition",
        lambda: ExtLQGCostDecomposition(
            deterministic_initial_state=12.0,
            initial_covariance_trace=30.0,
            accumulated_noise_scalar=2.0,
            total_expected_cost=44.0,
            provenance="unit-test",
        ),
    )

    result = materialize_gru_objective_comparator_sidecar(
        experiment="abc1234",
        run_ids=("run_a", "run_b"),
        checkpoint_policy="validation_selected_per_replicate",
        use_validation_selected_checkpoints=True,
        checkpoint_manifest=_checkpoint_selection(),
        checkpoint_manifest_path=None,
        standard_manifest_path=manifest_path,
        output_path=output_json,
        note_path=output_md,
        repo_root=tmp_path,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert result["status"] == "materialized"
    assert result["n_rows"] == 2
    assert payload["source_manifest"] == "results/abc1234/notes/standard_manifest.json"
    assert payload["rows"][0]["comparability"]["status"] == "not_comparable"
    assert payload["rows"][0]["selected_to_extlqg_deterministic_ratio"] is None
    assert output_md.exists()
