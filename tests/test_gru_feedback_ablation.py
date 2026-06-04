"""Tests for C&S GRU feedback-ablation diagnostics."""

from __future__ import annotations

import json

import jax.random as jr
import jax.numpy as jnp
import numpy as np
from feedbax.graph import Wire
from feedbax.state import CartesianState
from feedbax.task import TaskTrialSpec

from rlrmp.analysis.gru_feedback_ablation import (
    SCHEMA_VERSION,
    build_observation_ablation_spec,
    build_observation_tape,
    default_ablation_modes,
    insert_observation_ablation,
    interpret_run_feedback_ablation,
    render_feedback_ablation_markdown,
    selected_feedback_ablation_bins,
)
from rlrmp.cs_lss_gru import build_cs_lss_gru_graph


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
    assert build_observation_tape(
        "position_only_observation",
        bin_feedback=feedback,
        nominal_feedback=nominal,
    ) is None


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
