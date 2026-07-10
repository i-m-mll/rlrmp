"""Tests for C&S GRU feedback-ablation diagnostics."""

from __future__ import annotations

import json

import jax.numpy as jnp
import jax.random as jr
from jax import Array as JaxArray
import numpy as np
from feedbax import TaskTrialSpec
from feedbax.runtime.graph import Wire
from feedbax.runtime.state import CartesianState

from rlrmp.analysis.pipelines.gru_feedback_ablation import (
    FEEDBACK_AUDIT_SELECTION_ROLE,
    SCHEMA_VERSION,
    _per_replicate_command_penalty_metrics,
    _per_replicate_cost_delta_values,
    build_observation_ablation_spec,
    build_observation_tape,
    default_ablation_modes,
    feedback_checkpoint_selection_audit,
    insert_observation_ablation,
    interpret_run_feedback_ablation,
    render_feedback_ablation_markdown,
    selected_feedback_ablation_bins,
    selected_feedback_ablation_bins_for_bank,
    summarize_normalized_feedback_use,
)
from rlrmp.analysis.pipelines.gru_perturbation_bank import default_cs_perturbation_bank
from rlrmp.model.cs_lss_gru import build_cs_lss_gru_graph


def test_standard_modes_and_bins_are_json_serializable() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "modes": list(default_ablation_modes()),
        "bins": selected_feedback_ablation_bins(),
    }

    decoded = json.loads(json.dumps(payload))

    assert decoded["schema_version"] == SCHEMA_VERSION
    assert decoded["modes"] == [
        "normal",
        "frozen_nominal_observation_tape",
        "zeroed_perturbation_observation_deviation",
        "shuffled_observation_history",
        "lagged_observation_history",
        "position_only_observation",
        "velocity_only_observation",
    ]
    assert set(decoded["bins"]) == {
        "nominal",
        "initial_state",
        "process_epsilon",
        "sensory_feedback",
        "delayed_observation",
    }


def test_selected_feedback_ablation_bins_exist_in_current_bank() -> None:
    bank = default_cs_perturbation_bank()
    perturbations = {str(row["perturbation_id"]) for row in bank["perturbations"]}

    for perturbation_id in selected_feedback_ablation_bins_for_bank(bank).values():
        if perturbation_id is not None:
            assert perturbation_id in perturbations


def test_observation_tape_modes_transform_expected_axes() -> None:
    feedback = np.arange(2 * 3 * 4 * 4, dtype=np.float64).reshape(2, 3, 4, 4)
    nominal = np.full_like(feedback, -1.0)

    frozen = build_observation_tape(
        "frozen_nominal_observation_tape",
        bin_feedback=feedback,
        nominal_feedback=nominal,
    )
    zeroed = build_observation_tape(
        "zeroed_perturbation_observation_deviation",
        bin_feedback=feedback,
        nominal_feedback=nominal,
    )
    shuffled = build_observation_tape(
        "shuffled_observation_history",
        bin_feedback=feedback,
        nominal_feedback=nominal,
    )
    lagged = build_observation_tape(
        "lagged_observation_history",
        bin_feedback=feedback,
        nominal_feedback=nominal,
    )

    np.testing.assert_allclose(frozen, nominal)
    np.testing.assert_allclose(zeroed, nominal)
    np.testing.assert_allclose(shuffled[:, 0], feedback[:, -1])
    np.testing.assert_allclose(lagged[:, :, 0], feedback[:, :, 0])
    np.testing.assert_allclose(lagged[:, :, 1:], feedback[:, :, :-1])
    assert (
        build_observation_tape(
            "position_only_observation",
            bin_feedback=feedback,
            nominal_feedback=nominal,
        )
        is None
    )


def test_observation_tape_preserves_jax_array_boundary() -> None:
    feedback = jnp.arange(2 * 3 * 4 * 4, dtype=jnp.float64).reshape(2, 3, 4, 4)
    nominal = jnp.full_like(feedback, -1.0)

    frozen = build_observation_tape(
        "frozen_nominal_observation_tape",
        bin_feedback=feedback,
        nominal_feedback=nominal,
    )
    shuffled = build_observation_tape(
        "shuffled_observation_history",
        bin_feedback=feedback,
        nominal_feedback=nominal,
    )
    lagged = build_observation_tape(
        "lagged_observation_history",
        bin_feedback=feedback,
        nominal_feedback=nominal,
    )

    assert isinstance(frozen, JaxArray)
    assert isinstance(shuffled, JaxArray)
    assert isinstance(lagged, JaxArray)
    np.testing.assert_allclose(np.asarray(frozen), np.asarray(nominal))
    np.testing.assert_allclose(np.asarray(shuffled[:, 0]), np.asarray(feedback[:, -1]))
    np.testing.assert_allclose(np.asarray(lagged[:, :, 0]), np.asarray(feedback[:, :, 0]))
    np.testing.assert_allclose(np.asarray(lagged[:, :, 1:]), np.asarray(feedback[:, :, :-1]))


def test_feedback_ablation_inserts_override_and_mask_nodes() -> None:
    graph = build_cs_lss_gru_graph(hidden_size=3, key=jr.PRNGKey(0))
    override = build_observation_ablation_spec(
        "frozen_nominal_observation_tape",
        bin_id="initial_state",
    )
    masked = build_observation_ablation_spec("position_only_observation", bin_id="nominal")

    override_graph = insert_observation_ablation(graph, override)
    masked_graph = insert_observation_ablation(graph, masked)

    assert Wire("sensory", "output", "net", "feedback") not in override_graph.wires
    assert Wire("sensory", "output", override.label, "signal") in override_graph.wires
    assert Wire(override.label, "feedback", "net", "feedback") in override_graph.wires
    assert override.input_key in override_graph.input_ports
    assert override_graph.input_bindings[override.input_key] == (override.label, "replacement")
    assert Wire("sensory", "output", "net", "feedback") not in masked_graph.wires
    assert masked.input_key is None
    assert masked.label in masked_graph.nodes


def test_interpretation_labels_feedback_and_motor_tape_cases() -> None:
    sensitive_rows = [
        {
            "status": "evaluated",
            "bin": "initial_state",
            "mode": "lagged_observation_history",
            "metrics": {"delta_action_norm": {"mean": 0.2}},
        }
    ]
    tape_rows = [
        {
            "status": "evaluated",
            "bin": "initial_state",
            "mode": "lagged_observation_history",
            "metrics": {"delta_action_norm": {"mean": 0.0}},
        }
    ]

    assert interpret_run_feedback_ablation(sensitive_rows)["label"] == "feedback_sensitive"
    assert interpret_run_feedback_ablation(tape_rows)["label"] == "motor_tape_like"
    assert interpret_run_feedback_ablation([])["label"] == "inconclusive"


def test_normalized_feedback_use_reports_ratios_and_denominator_warnings() -> None:
    rows = [
        {
            "status": "evaluated",
            "bin": "initial_state",
            "mode": "normal",
            "metrics": {
                "rollout_full_qrf": {
                    "status": "available",
                    "delta_cost": {"total": {"mean": 2.0}},
                }
            },
        },
        {
            "status": "evaluated",
            "bin": "initial_state",
            "mode": "frozen_nominal_observation_tape",
            "metrics": {
                "baseline_action_norm": {"mean": 2.0},
                "delta_action_norm": {"mean": 0.5},
                "rollout_full_qrf": {
                    "status": "available",
                    "delta_cost": {"total": {"mean": 3.0}},
                    "perturbed_cost": {"total": {"mean": 12.0}},
                },
            },
        },
        {
            "status": "evaluated",
            "bin": "sensory_feedback",
            "mode": "lagged_observation_history",
            "metrics": {
                "baseline_action_norm": {"mean": 0.0},
                "delta_action_norm": {"mean": 1.0},
            },
        },
    ]

    summary = summarize_normalized_feedback_use(rows)

    assert summary["status"] == "available"
    assert summary["ablation_dependence_index"]["value"] == 0.25
    assert summary["perturbation_rescue_index"]["value"] == 0.25
    assert summary["correction_index_vs_open_loop"]["status"] == "not_available"
    assert any("open-loop data not supplied" in warning for warning in summary["warnings"])
    assert any("denominator unavailable or near zero" in warning for warning in summary["warnings"])


def test_feedback_checkpoint_selection_audit_selects_candidate_without_primary_leakage() -> None:
    manifest = {
        "checkpoint_policy": "validation_selected_per_replicate",
        "runs": {
            "run_a": {
                "feedback_checkpoint_rescore": {
                    "status": "materialized",
                    "feedback_selected_checkpoints": [
                        {
                            "replicate": 0,
                            "feedback_selected_checkpoint_batches": 6500,
                            "validation_selected_checkpoint_batches": 11500,
                            "feedback_minus_validation_batches": -5000,
                        }
                    ],
                }
            }
        },
    }

    audit = feedback_checkpoint_selection_audit(manifest)

    assert audit["status"] == "materialized"
    assert audit["candidate_granularity"] == "checkpoint_batch_per_replicate"
    assert audit["selection_use"] == FEEDBACK_AUDIT_SELECTION_ROLE
    assert (
        audit["runs"]["run_a"]["feedback_selected_checkpoints"][0][
            "feedback_minus_validation_batches"
        ]
        == -5000
    )


def test_feedback_checkpoint_selection_audit_has_legacy_run_fallback() -> None:
    manifest = {
        "checkpoint_policy": "validation_selected_per_replicate",
        "runs": {
            "run_a": {
                "label": "A",
                "checkpoint_selection": [{"replicate": 0, "batch": 500}],
                "normalized_feedback_use": {
                    "score": 0.2,
                    "score_components": ["ablation_dependence_index"],
                },
            },
            "run_b": {
                "label": "B",
                "checkpoint_selection": [{"replicate": 0, "batch": 1000}],
                "normalized_feedback_use": {
                    "score": 0.7,
                    "score_components": ["perturbation_rescue_index"],
                },
            },
        },
    }

    audit = feedback_checkpoint_selection_audit(manifest)

    assert audit["status"] == "available"
    assert audit["selection_use"] == FEEDBACK_AUDIT_SELECTION_ROLE
    assert audit["primary_checkpoint_policy"] == "validation_selected_per_replicate"
    assert audit["selected_candidate"]["run_id"] == "run_b"
    assert audit["candidate_granularity"] == "run_legacy_fallback"


def test_per_replicate_cost_delta_values_reduces_trials_only() -> None:
    base_cost = {
        "status": "available",
        "total": {"values": [[10.0, 12.0], [20.0, 22.0]]},
    }
    perturbed_cost = {
        "status": "available",
        "total": {"values": [[11.0, 15.0], [30.0, 36.0]]},
    }

    assert _per_replicate_cost_delta_values(
        base_cost,
        perturbed_cost,
        n_replicates=2,
    ) == [2.0, 12.0]


def test_per_replicate_command_penalty_metrics_accepts_jax_arrays() -> None:
    baseline = jnp.asarray(
        [
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
            [[[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]]],
        ],
        dtype=jnp.float64,
    )
    perturbed = baseline * 2.0

    metrics = _per_replicate_command_penalty_metrics(
        baseline_command=baseline,
        perturbed_command=perturbed,
        n_replicates=2,
    )

    assert [row["status"] for row in metrics] == ["available", "available"]
    np.testing.assert_allclose(metrics[0]["command_energy_ratio"], 4.0)
    np.testing.assert_allclose(metrics[1]["command_smoothness_ratio"], 4.0)
    assert metrics[0]["command_oscillation_ratio"] == metrics[1]["command_oscillation_ratio"]


def test_markdown_renders_not_available_rows() -> None:
    manifest = {
        "issue": "57ab156",
        "source_experiment": "aacb9ed",
        "scope": "test",
        "checkpoint_policy": "validation_selected_per_replicate",
        "runs": {
            "run": {
                "interpretation": {"label": "inconclusive", "reason": "test"},
                "ablations": [
                    {
                        "bin": "delayed_observation",
                        "mode": "frozen_nominal_observation_tape",
                        "status": "not_available",
                        "reason": "missing adapter",
                    }
                ],
            }
        },
    }

    markdown = render_feedback_ablation_markdown(manifest)

    assert "GRU Feedback Ablation Diagnostic" in markdown
    assert "delayed_observation" in markdown
    assert "not_available" in markdown


def test_trial_spec_imports_remain_available_for_payload_shape_contract() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.effector": CartesianState(pos=jnp.zeros((2, 2)))},
        targets={},
        inputs={"effector_target": CartesianState(pos=jnp.zeros((2, 5, 2)))},
    )

    assert trial_specs.inputs["effector_target"].pos.shape == (2, 5, 2)
