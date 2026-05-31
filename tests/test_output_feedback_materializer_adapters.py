"""Tests for reusable output-feedback certificate materializer adapters."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import materialize_output_feedback_failure_decomposition as failure  # noqa: E402
import materialize_output_feedback_sweep_certificates as certificates  # noqa: E402
import materialize_output_feedback_time_constrained as time_constrained  # noqa: E402
from rlrmp.analysis.bridge_contracts import BridgeCertificateComponent  # noqa: E402


def _fit(label: str) -> dict[str, object]:
    return {
        "label": label,
        "objective_ratio_to_reference": 1.2,
        "objective_final": 12.0,
        "objective_reference": 10.0,
        "gradient_norm_final": 0.4,
        "projected_gradient_norm_final": 0.3,
        "clean_action_mismatch_ratio": 0.1,
        "under_epsilon_action_mismatch_ratio": 0.2,
        "exact_l2_cost_ratio_to_lqr": 1.1,
        "exact_l2_cost_ratio_to_hinf": 1.3,
        "gamma_penalized_lambda_over_gamma_squared": 1.5,
        "gamma_penalized_feasible": True,
        "n_iterations": 7,
        "optimizer_success": True,
        "optimizer_status": "ok",
        "clean_rollout": {
            "peak_forward_velocity": 1.0,
            "terminal_position_error_m": 0.01,
            "control_effort": 2.0,
        },
    }


def test_standard_adapter_uses_arbitrary_array_prefix_and_preserves_schema(monkeypatch, tmp_path):
    calls = []

    def fake_components(**kwargs):
        calls.append(kwargs)
        return (
            BridgeCertificateComponent.available(
                certificates.STATE_WEIGHTED_ACTION_MISMATCH,
                mismatch_ratio_mean=0.1,
            ),
        )

    monkeypatch.setattr(certificates, "_full_standard_components", fake_components)
    arrays = {"smooth_cell_K": np.ones((2, 1, 2))}

    rows = certificates.deterministic_standard_rows_from_manifest_entries(
        entries=[
            {
                "fit": _fit("cell_a"),
                "array_prefix": "smooth_cell",
                "run_parts": ("spline_time_basis", "cell_a"),
                "parameters": {"basis": "smooth"},
                "metrics": {"basis_degree": 3},
            }
        ],
        arrays=arrays,
        reference=object(),
        output_config=object(),
        issue_id="87edaae",
        source_manifest=tmp_path / "manifest.json",
        default_family="smooth time-basis",
        default_training_distribution="mixed",
    )

    assert [row["spec"]["run_id"] for row in rows] == [
        "spline_time_basis__cell_a__nominal_clean",
        "spline_time_basis__cell_a__riccati_epsilon_response",
    ]
    assert {row["spec"]["issue_id"] for row in rows} == {"87edaae"}
    assert rows[0]["spec"]["parameters"]["basis"] == "smooth"
    assert rows[0]["metrics"]["basis_degree"] == 3
    assert rows[0]["status"] == "full_standard_certificate"
    assert calls[0]["array_prefix"] == "smooth_cell_clean"
    assert calls[1]["array_prefix"] == "smooth_cell_under_eps"
    assert np.array_equal(calls[0]["candidate_gain"], arrays["smooth_cell_K"])


def test_failure_adapter_joins_standard_rows_by_generated_run_id(monkeypatch):
    monkeypatch.setattr(
        failure,
        "materialize_reference",
        lambda: SimpleNamespace(schedule=SimpleNamespace(R=np.eye(1))),
    )
    standard_rows = []
    for lens in ("nominal_clean", "riccati_epsilon_response"):
        standard_rows.append(
            {
                "spec": {
                    "run_id": f"spline_time_basis__cell_a__{lens}",
                    "parameters": {"distribution_family": "smooth time-basis"},
                },
                "status": "full_standard_certificate",
                "certificate_components": [
                    {
                        "name": certificates.STATE_WEIGHTED_ACTION_MISMATCH,
                        "summary": {"mismatch_ratio_mean": 0.1},
                    },
                    {
                        "name": certificates.BELLMAN_HESSIAN_RESIDUAL,
                        "summary": {"residual_ratio_mean": 0.2},
                    },
                    {
                        "name": certificates.CLOSED_LOOP_TRANSITION_MISMATCH,
                        "summary": {"mismatch_ratio_mean": 0.3},
                    },
                    {
                        "name": certificates.VALUE_POLICY_GAP,
                        "summary": {"gap_ratio_mean": 0.4},
                    },
                ],
            }
        )
    arrays = {
        "lqr_reference_K": np.zeros((2, 1, 2)),
        "smooth_cell_K": np.ones((2, 1, 2)),
        "smooth_cell_clean_x_hat": np.asarray([[1.0, 0.0], [0.5, 0.0], [0.25, 0.0]]),
        "smooth_cell_under_eps_x_hat": np.asarray([[0.0, 1.0], [0.0, 0.5], [0.0, 0.25]]),
    }

    rows = failure.failure_rows_from_manifest_entries(
        entries=[
            {
                "fit": _fit("cell_a"),
                "array_prefix": "smooth_cell",
                "run_parts": ("spline_time_basis", "cell_a"),
                "row_parameters": {"basis": "smooth"},
            }
        ],
        arrays=arrays,
        standard_rows={"standard_certificate": {"rows": standard_rows}},
        default_source_group="spline_time_basis",
    )

    assert [row["run_id"] for row in rows] == [
        "spline_time_basis__cell_a__nominal_clean",
        "spline_time_basis__cell_a__riccati_epsilon_response",
    ]
    assert rows[0]["source_group"] == "spline_time_basis"
    assert rows[0]["row_parameters"] == {"basis": "smooth"}
    assert rows[0]["source_standard_status"] == "full_standard_certificate"
    assert rows[0]["certificate"]["state_weighted_action_mismatch"] == 0.1
    assert rows[0]["gain_error_decomposition"]["time_steps"] == 2


def test_time_constrained_row_entries_group_r12_coverage_metadata():
    summary = {
        "diagnostics": {"basis_family": "cardinal_cubic_b_spline"},
        "fits": [
            {
                **_fit("spline_r12__state_eigenspectrum_state"),
                "condition": {
                    "rank": 12,
                    "optimizer": "adamw_then_lbfgsb",
                    "learning_rate": 0.01,
                    "state_eigenspectrum_coverage": {
                        "objective": "state",
                        "n_modes": 4,
                        "scale": 3.0,
                        "weight": 0.1,
                        "reference": "lqr",
                    },
                },
                "initialization": "scratch",
            },
            {
                **_fit("spline_r12__observer_error_state"),
                "condition": {
                    "rank": 12,
                    "optimizer": "adamw_then_lbfgsb",
                    "learning_rate": 0.01,
                    "observer_error_coverage": {
                        "objective": "state",
                        "n_modes": 1,
                        "scale": 0.3,
                        "weight": 0.1,
                        "reference": "lqr",
                    },
                },
                "initialization": "scratch",
            },
            {
                **_fit("spline_r12__trajectory_eigenspectrum"),
                "condition": {
                    "rank": 12,
                    "optimizer": "adamw_then_lbfgsb",
                    "learning_rate": 0.01,
                    "eigenspectrum_coverage": {
                        "objective": "trajectory",
                        "n_modes": 1,
                        "scale": 1.0,
                        "weight": 0.1,
                    },
                },
                "initialization": "scratch",
            },
        ],
    }

    entries = time_constrained._row_entries(summary)

    assert [entry["fit"]["label"] for entry in entries] == [
        "spline_r12__state_eigenspectrum_state",
        "spline_r12__observer_error_state",
    ]
    state_entry, observer_entry = entries
    assert state_entry["training_distribution"] == "state_eigenspectrum_state"
    assert state_entry["source_group"] == "state_eigenspectrum"
    assert state_entry["parameters"] == {
        "rank": 12,
        "basis_family": "cardinal_cubic_b_spline",
        "initialization": "scratch",
        "coverage_family": "state_eigenspectrum",
        "coverage_objective": "state",
        "coverage_modes": 4,
        "coverage_scale": 3.0,
        "coverage_weight": 0.1,
        "coverage_reference": "lqr",
    }
    assert state_entry["run_parts"][:4] == (
        "smooth_spline_time_basis",
        "rank_12",
        "state_eigenspectrum",
        "state",
    )
    assert observer_entry["training_distribution"] == "observer_error_state"
    assert observer_entry["parameters"]["coverage_family"] == "observer_error"
    assert observer_entry["parameters"]["coverage_modes"] == 1


def test_time_constrained_row_entries_group_r20_coverage_metadata():
    summary = {
        "diagnostics": {"basis_family": "cardinal_cubic_b_spline"},
        "fits": [
            {
                **_fit("spline_r20__state_eigenspectrum_state"),
                "condition": {
                    "rank": 20,
                    "optimizer": "adamw_then_lbfgsb",
                    "learning_rate": 0.01,
                    "eigenspectrum_coverage": {
                        "objective": "state",
                        "n_modes": 4,
                        "scale": 1.0,
                        "weight": 0.1,
                        "reference": "lqr_exact_budget_l2",
                    },
                },
                "initialization": "scratch",
            },
            {
                **_fit("spline_r20__observer_error_state"),
                "condition": {
                    "rank": 20,
                    "optimizer": "adamw_then_lbfgsb",
                    "learning_rate": 0.01,
                    "observer_error_coverage": {
                        "objective": "state",
                        "n_modes": 1,
                        "scale": 0.3,
                        "weight": 0.1,
                        "reference": "lqr_exact_budget_l2",
                    },
                },
                "initialization": "scratch",
            },
        ],
    }

    entries = time_constrained._row_entries(summary)

    assert [entry["fit"]["label"] for entry in entries] == [
        "spline_r20__state_eigenspectrum_state",
        "spline_r20__observer_error_state",
    ]
    assert entries[0]["parameters"] == {
        "rank": 20,
        "basis_family": "cardinal_cubic_b_spline",
        "initialization": "scratch",
        "coverage_family": "state_eigenspectrum",
        "coverage_objective": "state",
        "coverage_modes": 4,
        "coverage_scale": 1.0,
        "coverage_weight": 0.1,
        "coverage_reference": "lqr_exact_budget_l2",
    }
    assert entries[0]["run_parts"][:4] == (
        "smooth_spline_time_basis",
        "rank_20",
        "state_eigenspectrum",
        "state",
    )
    assert "r=20 state-coverage follow-up" in entries[0]["notes"]
    assert entries[1]["training_distribution"] == "observer_error_state"
    assert entries[1]["source_group"] == "observer_error"


def test_time_constrained_r20_coverage_conditions_are_focused():
    conditions = time_constrained._r20_coverage_conditions(
        include=True,
        rank=20,
        learning_rate=0.01,
        adamw_steps=5000,
        polish_maxiter=1000,
        state_eigenspectrum_modes=(4,),
        state_eigenspectrum_scales=(1.0, 3.0),
        observer_error_modes=(1,),
        observer_error_scales=(0.3,),
        weight=0.1,
    )

    assert len(conditions) == 3
    assert {
        (condition.eigenspectrum_coverage.n_modes, condition.eigenspectrum_coverage.scale)
        for condition in conditions
        if condition.eigenspectrum_coverage is not None
    } == {(4, 1.0), (4, 3.0)}
    assert {
        (condition.observer_error_coverage.n_modes, condition.observer_error_coverage.scale)
        for condition in conditions
        if condition.observer_error_coverage is not None
    } == {(1, 0.3)}
    assert all(condition.rank == 20 for condition in conditions)


def test_time_constrained_main_routes_r20_coverage_to_distinct_outputs(monkeypatch):
    calls = {}

    def fake_parse_args():
        return SimpleNamespace(
            ranks=",".join(str(rank) for rank in time_constrained.SPLINE_RANKS),
            fit_ranks="",
            adamw_lrs="0.003,0.01",
            lbfgsb_maxiter=2000,
            adamw_steps=5000,
            polish_maxiter=1000,
            note_output=time_constrained.NOTE_PATH,
            manifest_output=time_constrained.MANIFEST_PATH,
            artifact_output=time_constrained.ARTIFACT_PATH,
            include_r12_coverage=False,
            r12_coverage_rank=12,
            r12_state_eigenspectrum_modes="1,4",
            r12_state_eigenspectrum_scales="0.3,1,3",
            r12_observer_error_modes="1",
            r12_observer_error_scales="0.3,1",
            r12_coverage_weight=0.1,
            include_r20_coverage=True,
            r20_coverage_rank=20,
            r20_state_eigenspectrum_modes="4",
            r20_state_eigenspectrum_scales="1,3",
            r20_observer_error_modes="1",
            r20_observer_error_scales="0.3",
            r20_coverage_weight=0.1,
        )

    def fake_materialize(**kwargs):
        calls["materialize"] = kwargs
        return {"fits": [], "projections": []}, {}

    def fake_write_result(summary, *, arrays, note_path, manifest_path, artifact_path):
        calls["write"] = {
            "summary": summary,
            "arrays": arrays,
            "note_path": note_path,
            "manifest_path": manifest_path,
            "artifact_path": artifact_path,
        }

    monkeypatch.setattr(time_constrained, "parse_args", fake_parse_args)
    monkeypatch.setattr(time_constrained, "materialize", fake_materialize)
    monkeypatch.setattr(time_constrained, "write_result", fake_write_result)

    time_constrained.main()

    assert calls["materialize"]["ranks"] == (20,)
    assert calls["materialize"]["fit_ranks"] == (20,)
    assert calls["materialize"]["include_r20_coverage"] is True
    assert calls["materialize"]["r20_state_eigenspectrum_modes"] == (4,)
    assert calls["materialize"]["r20_state_eigenspectrum_scales"] == (1.0, 3.0)
    assert calls["materialize"]["r20_observer_error_scales"] == (0.3,)
    assert calls["write"]["note_path"] == time_constrained.R20_COVERAGE_NOTE_PATH
    assert calls["write"]["manifest_path"] == time_constrained.R20_COVERAGE_MANIFEST_PATH
    assert calls["write"]["artifact_path"] == time_constrained.R20_COVERAGE_ARTIFACT_PATH
