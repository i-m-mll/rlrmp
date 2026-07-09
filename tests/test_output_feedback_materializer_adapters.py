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
from rlrmp.analysis.pipelines.bridge_contracts import BridgeCertificateComponent  # noqa: E402


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
