"""Tests for the C&S GRU perturbation-response bank."""

from __future__ import annotations

import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from feedbax import TaskTrialSpec, TrialTimeline
from feedbax.runtime.graph import Wire
from feedbax.objectives.loss import TargetSpec
from feedbax.runtime.state import CartesianState

import rlrmp.analysis.pipelines.gru_perturbation_bank as perturbation_bank
from rlrmp.analysis.math.cs_game_card import build_canonical_game, build_no_integrator_game
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import RolloutEvaluation
from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    GRAPH_ADAPTER_INPUT_PREFIX,
    SCHEMA_VERSION,
    apply_perturbation_to_trial_specs,
    default_cs_perturbation_bank,
    delta_full_qrf_cost_summary,
    evaluate_extlqg_perturbation_comparator,
    evaluate_robust_output_feedback_perturbation_comparator,
    extlqg_comparator_status,
    full_qrf_cost_summary,
    render_perturbation_response_markdown,
    robust_output_feedback_comparator_status,
    score_full_qrf_rollout_cost,
    summarize_perturbation_bank,
    summarize_perturbation_response,
)
from rlrmp.model.cs_lss_gru import build_cs_lss_gru_graph
from rlrmp.train.cs_perturbation_training import (
    GRAPH_ADAPTER_SPECS as TRAINING_GRAPH_ADAPTER_SPECS,
    add_zero_graph_channel_inputs,
    graph_adapter_specs,
    install_perturbation_training_graph_adapters,
)


def _delayed_trial_specs(
    go_steps: np.ndarray,
    *,
    n_steps: int = 60,
    include_epsilon: bool = False,
) -> TaskTrialSpec:
    go_steps = np.asarray(go_steps, dtype=np.int32)
    batch_size = int(go_steps.shape[0])
    inputs: dict[str, object] = {
        "effector_target": CartesianState(pos=np.zeros((batch_size, n_steps, 2))),
    }
    if include_epsilon:
        inputs["epsilon"] = np.zeros((batch_size, n_steps, 8), dtype=np.float64)
    epoch_bounds = np.stack(
        [
            np.zeros_like(go_steps),
            go_steps,
            np.full_like(go_steps, n_steps),
        ],
        axis=-1,
    )
    return TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((batch_size, 8), dtype=np.float64)},
        targets={},
        inputs=inputs,
        timeline=TrialTimeline(n_steps=n_steps, epoch_bounds=epoch_bounds),
    )


def test_default_bank_is_json_serializable_with_required_channels() -> None:
    bank = default_cs_perturbation_bank()

    encoded = json.dumps(bank)
    decoded = json.loads(encoded)

    assert decoded["schema_version"] == SCHEMA_VERSION
    assert decoded["bank_id"] == "cs_standard_perturbation_response_v3"
    assert decoded["initial_position_information_contracts"]["cases"][
        "D_current_state_immediately_visible"
    ]["status"] == "implemented"
    assert decoded["initial_position_information_contracts"]["cases"][
        "A_target_changed_hand_start_nominal"
    ]["status"] == "not_available"
    assert decoded["target_relative_alignment"]["missing_inputs_status"] == "not_available"
    assert decoded["calibration_metadata_hooks"]["coordinating_issue"] == "1ad3c16"
    channels = {row["channel"] for row in decoded["perturbations"]}
    assert channels == {
        "initial_state",
        "command_input",
        "process_epsilon",
        "sensory_feedback",
        "target_stream",
    }
    assert decoded["graphspec_alignment"]["named_channels"] == [
        "initial_state",
        "command_input",
        "process_epsilon",
        "sensory_feedback",
        "target_stream",
    ]
    assert "plant_force" in decoded["legacy_migration"]
    assert not any(row["channel"] == "plant_force" for row in decoded["perturbations"])
    process_families = {
        row["family"] for row in decoded["perturbations"] if row["channel"] == "process_epsilon"
    }
    assert process_families == {
        "process_epsilon_position_xy",
        "process_epsilon_velocity_xy",
        "process_epsilon_force_state_xy",
        "process_epsilon_integrator_xy",
    }
    force_y_rows = [
        row
        for row in decoded["perturbations"]
        if row.get("epsilon_component") == "force_state_y"
    ]
    assert force_y_rows
    assert {row["epsilon_index"] for row in force_y_rows} == {5}
    initial_position_rows = [
        row for row in decoded["perturbations"] if row["family"] == "initial_position_offset"
    ]
    assert {row["initial_position_case"] for row in initial_position_rows} == {
        "D_current_state_immediately_visible"
    }
    lateral_rows = [
        row
        for row in decoded["perturbations"]
        if row["family"] == "target_aligned_lateral_command_load_pulse"
    ]
    assert len(lateral_rows) == 6
    assert {row["axis"] for row in lateral_rows} == {"y"}
    assert {row["timing_bin"] for row in lateral_rows} == {"early", "mid", "late"}
    assert {row["semantic_family"] for row in lateral_rows} == {
        "human_protocol_like_lateral_mechanical_load"
    }
    assert all(
        row["channel_provenance"]["target_relative_axis_role"] == "tangential"
        for row in lateral_rows
    )
    assert len(decoded["perturbations"]) == 111


def test_default_bank_emits_timing_bin_specific_rows() -> None:
    bank = default_cs_perturbation_bank()
    rows = bank["perturbations"]

    command_rows = [row for row in rows if row["family"] == "command_input_pulse"]
    assert len(command_rows) == 12
    assert {
        (row["timing_bin"], row["timing"]["start_time_index"], row["timing"]["duration_steps"])
        for row in command_rows
    } == {("early", 5, 5), ("mid", 15, 5), ("late", 35, 5)}

    process_rows = [row for row in rows if row["channel"] == "process_epsilon"]
    assert len(process_rows) == 48
    assert {
        (row["family"], row["timing_bin"])
        for row in process_rows
        if row["epsilon_component"] == "force_state_y"
    } == {
        ("process_epsilon_force_state_xy", "early"),
        ("process_epsilon_force_state_xy", "mid"),
        ("process_epsilon_force_state_xy", "late"),
    }

    sensory_rows = [row for row in rows if row["family"] == "sensory_feedback_offset"]
    delayed_rows = [row for row in rows if row["family"] == "delayed_observation_offset"]
    assert len(sensory_rows) == 36
    assert delayed_rows == []
    assert not any(row["channel"] == "delayed_observation" for row in rows)
    assert {
        (row["timing_bin"], row["timing"]["start_time_index"], row["timing"]["duration_steps"])
        for row in sensory_rows
    } == {("early_visible", 10, 5), ("mid_visible", 20, 5), ("late_visible", 40, 5)}
    assert {
        (
            row["channel_provenance"]["feedback_quantity"],
            row["channel_provenance"]["target_relative_axis_role"],
            row["sign"],
        )
        for row in sensory_rows
    } == {
        ("position", "radial", -1),
        ("position", "radial", 1),
        ("position", "tangential", -1),
        ("position", "tangential", 1),
        ("velocity", "radial", -1),
        ("velocity", "radial", 1),
        ("velocity", "tangential", -1),
        ("velocity", "tangential", 1),
        ("force_filter", "radial", -1),
        ("force_filter", "radial", 1),
        ("force_filter", "tangential", -1),
        ("force_filter", "tangential", 1),
    }
    force_filter_rows = [
        row
        for row in sensory_rows
        if row["channel_provenance"]["feedback_quantity"] == "force_filter"
    ]
    assert len(force_filter_rows) == 12
    assert {row["units"] for row in force_filter_rows} == {"N"}
    assert {row["channel_provenance"]["feedback_payload_index"] for row in force_filter_rows} == {
        4,
        5,
    }
    assert all(
        row["channel_provenance"]["force_filter_feedback_only"] is True
        for row in force_filter_rows
    )

    initial_rows = [row for row in rows if row["channel"] == "initial_state"]
    assert {row["timing_bin"] for row in initial_rows} == {"initial_condition"}
    assert {row["timing"]["time_index"] for row in initial_rows} == {0}
    assert bank["timing_bin_conventions"]["plant_side"][0]["start_time_index"] == 5
    assert bank["timing_bin_conventions"]["controller_visible"][0]["start_time_index"] == 10


def _feedback_scale_manifest() -> dict[str, object]:
    return {
        "schema_version": "rlrmp.gru_evaluation_diagnostics.v1",
        "runs": {
            "run_a": {
                "controller_feedback_scales": {
                    "status": "available",
                    "schema_version": "rlrmp.controller_feedback_scales.v1",
                    "run_id": "run_a",
                    "checkpoint_policy": "validation_selected_per_replicate",
                    "feedback_basis": "target_relative_delayed_feedback_plus_force_filter",
                    "feedback_dim": 6,
                    "statistic": "p95_norm",
                    "components": {
                        "position": {
                            "units": "m",
                            "reference_scale": 0.12,
                            "reference_scale_statistic": "p95_norm",
                        },
                        "velocity": {
                            "units": "m/s",
                            "reference_scale": 2.0,
                            "reference_scale_statistic": "p95_norm",
                        },
                        "force_filter": {
                            "units": "N",
                            "reference_scale": 40.0,
                            "reference_scale_statistic": "p95_norm",
                            "feedback_basis_indices": [4, 5],
                            "gru_input_indices": [4, 5],
                        },
                    },
                }
            }
        },
    }


def test_calibrated_bank_requires_feedback_scale_manifest_for_force_filter_rows() -> None:
    with pytest.raises(ValueError, match="force/filter feedback rows require"):
        default_cs_perturbation_bank(mode="calibrated", calibration_level="small")


def test_calibrated_bank_includes_force_filter_feedback_rows() -> None:
    bank = default_cs_perturbation_bank(
        mode="calibrated",
        calibration_level="small",
        feedback_scale_manifest=_feedback_scale_manifest(),
    )
    rows = bank["perturbations"]

    force_filter_rows = [
        row for row in rows if row.get("feedback_quantity") == "force_filter"
    ]
    assert len(force_filter_rows) == 12
    assert {row["channel"] for row in force_filter_rows} == {"sensory_feedback"}
    assert not any(row["channel"] == "delayed_observation" for row in rows)
    assert not any(row["family"] == "delayed_observation_offset" for row in rows)
    assert {row["units"] for row in force_filter_rows} == {"N"}
    assert {row["feedback_payload_index"] for row in force_filter_rows} == {4, 5}
    assert all(row["force_filter_feedback_only"] is True for row in force_filter_rows)
    assert all(
        row["calibration_role"] == "reach_relative_calibrated_native_units"
        for row in force_filter_rows
    )
    assert all(
        row["reference_force_filter_scale_N"] == 40.0
        for row in force_filter_rows
    )
    assert {
        row["amplitude"] / row["level_fraction_of_reach"]
        for row in force_filter_rows
    } == {40.0}
    assert all(
        row["controller_feedback_scale"]["aggregation"]
        == "mean_reference_scale_across_manifest_runs"
        for row in force_filter_rows
    )


def test_initial_position_adapter_offsets_cartesian_state_without_mutating_source() -> None:
    trial_specs = TaskTrialSpec(
        inits={
            "mechanics.effector": CartesianState(
                pos=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
                vel=np.asarray([[0.0, 0.0], [0.0, 0.0]]),
                force=np.asarray([[0.0, 0.0], [0.0, 0.0]]),
            )
        },
        targets={},
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 3, 2)))},
    )
    perturbation = {
        "channel": "initial_state",
        "family": "initial_position_offset",
        "amplitude": 0.01,
        "axis": "x",
        "sign": 1,
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    np.testing.assert_allclose(
        result.trial_specs.inits["mechanics.effector"].pos,
        np.asarray([[0.01, 0.0], [1.01, 1.0]]),
    )
    np.testing.assert_allclose(
        trial_specs.inits["mechanics.effector"].pos,
        np.asarray([[0.0, 0.0], [1.0, 1.0]]),
    )


def test_initial_velocity_adapter_offsets_vector_state() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 3, 2)))},
    )
    perturbation = {
        "channel": "initial_state",
        "family": "initial_velocity_offset",
        "amplitude": 0.05,
        "axis": "y",
        "sign": -1,
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    np.testing.assert_allclose(result.trial_specs.inits["mechanics.vector"][:, 3], -0.05)
    np.testing.assert_allclose(trial_specs.inits["mechanics.vector"], 0.0)


def test_command_input_pulse_adapter_sets_external_graph_input_payload() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 10, 2)))},
    )
    perturbation = {
        "perturbation_id": "command_input_pulse__t3_y_neg",
        "channel": "command_input",
        "family": "command_input_pulse",
        "amplitude": 2.0,
        "axis": "y",
        "sign": -1,
        "timing": {"start_time_index": 3, "duration_steps": 2},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    input_key = f"{GRAPH_ADAPTER_INPUT_PREFIX}.command_input.command_input_pulse__t3_y_neg"
    payload = result.trial_specs.inputs[input_key]
    np.testing.assert_allclose(payload[:, 3:5, 1], -2.0)
    np.testing.assert_allclose(payload[:, :3, :], 0.0)
    assert result.adapter_provenance["external_load_force"] is False
    assert result.adapter_provenance["insertion_point"] == "efferent.output -> mechanics.force"
    assert result.adapter_provenance["feedbax_additive_channel_adapter"]["label"] == "command_input"
    assert result.adapter_provenance["controller_input_mutated"] is False


def test_movement_indexed_graph_adapter_shifts_per_trial_delayed_go_cues() -> None:
    go_steps = np.asarray([10, 20], dtype=np.int32)
    trial_specs = _delayed_trial_specs(go_steps)
    perturbation = {
        "perturbation_id": "command_input_pulse__movement_early_x_pos",
        "channel": "command_input",
        "family": "command_input_pulse",
        "amplitude": 2.0,
        "axis": "x",
        "sign": 1,
        "timing": {"epoch": "movement_indexed", "start_time_index": 5, "duration_steps": 2},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    input_key = (
        f"{GRAPH_ADAPTER_INPUT_PREFIX}.command_input."
        "command_input_pulse__movement_early_x_pos"
    )
    payload = np.asarray(result.trial_specs.inputs[input_key])
    np.testing.assert_allclose(payload[0, 15:17, 0], 2.0)
    np.testing.assert_allclose(payload[1, 25:27, 0], 2.0)
    assert not np.any(payload[0, : go_steps[0], :])
    assert not np.any(payload[1, : go_steps[1], :])
    assert result.adapter_provenance["movement_start_aligned"] is True
    assert result.adapter_provenance["absolute_start_time_indices"] == [15, 25]


def test_movement_indexed_graph_adapter_preserves_immediate_start_without_timeline() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 10, 2)))},
    )
    perturbation = {
        "perturbation_id": "command_input_pulse__movement_early_y_neg",
        "channel": "command_input",
        "family": "command_input_pulse",
        "amplitude": 1.0,
        "axis": "y",
        "sign": -1,
        "timing": {"epoch": "movement_indexed", "start_time_index": 3, "duration_steps": 2},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    payload = result.trial_specs.inputs[
        f"{GRAPH_ADAPTER_INPUT_PREFIX}.command_input."
        "command_input_pulse__movement_early_y_neg"
    ]
    np.testing.assert_allclose(payload[:, 3:5, 1], -1.0)
    np.testing.assert_allclose(payload[:, :3, :], 0.0)
    assert result.adapter_provenance["movement_start_aligned"] is True
    assert result.adapter_provenance["movement_start_indices"] == [0, 0]


def test_command_input_graph_adapter_inserts_external_node_on_force_edge() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 10, 2)))},
    )
    graph = build_cs_lss_gru_graph(hidden_size=3, key=jr.PRNGKey(0))
    perturbation = {
        "perturbation_id": "command_input_pulse__t3_x_pos",
        "channel": "command_input",
        "family": "command_input_pulse",
        "amplitude": 1.0,
        "axis": "x",
        "sign": 1,
        "timing": {"start_time_index": 3, "duration_steps": 2},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation, model=graph)

    assert result.status == "evaluated"
    assert result.model is not None
    adapter_label = result.adapter_provenance["adapter_node"]
    assert adapter_label in result.model.nodes
    assert Wire("efferent", "output", "mechanics", "force") not in result.model.wires
    assert Wire("efferent", "output", adapter_label, "a") in result.model.wires
    assert Wire(adapter_label, "output", "mechanics", "force") in result.model.wires
    assert result.adapter_provenance["input_key"] in result.model.input_ports
    assert result.model.input_bindings[result.adapter_provenance["input_key"]] == (
        adapter_label,
        "b",
    )


def test_training_graph_channel_adapters_use_feedbax_sum_specs() -> None:
    graph = build_cs_lss_gru_graph(hidden_size=3, key=jr.PRNGKey(0))
    graph = install_perturbation_training_graph_adapters(graph)
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={
            "mechanics.effector.pos": TargetSpec(
                value=np.zeros((2, 10, 2), dtype=np.float32),
            )
        },
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 10, 2)))},
        timeline=TrialTimeline(n_steps=10),
    )
    trial_specs = add_zero_graph_channel_inputs(trial_specs)

    for spec in TRAINING_GRAPH_ADAPTER_SPECS.values():
        adapter_node = f"{spec.label}_additive"
        target = spec.target
        assert spec.target.kind == "edge"
        assert adapter_node in graph.nodes
        assert Wire(target.source_node, target.source_port, adapter_node, "a") in graph.wires
        assert Wire(adapter_node, "output", target.target_node, target.target_port) in graph.wires
        assert graph.input_bindings[spec.input_key] == (adapter_node, "b")
        assert spec.input_key in trial_specs.inputs
        assert trial_specs.inputs[spec.input_key].shape[-1] == spec.payload_shape[-1]


def test_process_epsilon_pulse_adapter_offsets_epsilon_input() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={
            "effector_target": CartesianState(pos=np.zeros((2, 10, 2))),
            "epsilon": np.zeros((2, 10, 8), dtype=np.float64),
        },
    )
    perturbation = {
        "channel": "process_epsilon",
        "family": "process_epsilon_pulse",
        "amplitude": 0.25,
        "axis": "x",
        "sign": 1,
        "timing": {"start_time_index": 3, "duration_steps": 2},
    }

    graph = build_cs_lss_gru_graph(
        hidden_size=3,
        bind_epsilon_input=True,
        key=jr.PRNGKey(0),
    )
    result = apply_perturbation_to_trial_specs(trial_specs, perturbation, model=graph)

    assert result.status == "evaluated"
    input_key = result.adapter_provenance["input_key"]
    np.testing.assert_allclose(result.trial_specs.inputs[input_key][:, 3:5, 0], 0.25)
    np.testing.assert_allclose(result.trial_specs.inputs[input_key][:, :3, :], 0.0)
    np.testing.assert_allclose(trial_specs.inputs["epsilon"], 0.0)
    assert result.adapter_provenance["process_channel"] == "LinearStateSpace.B_w"
    assert result.model.input_bindings["epsilon"][1] == "a"
    assert result.model.input_bindings[input_key][1] == "b"


def test_movement_indexed_process_epsilon_adapter_shifts_per_trial_delayed_go_cues() -> None:
    go_steps = np.asarray([10, 20], dtype=np.int32)
    trial_specs = _delayed_trial_specs(go_steps, include_epsilon=True)
    perturbation = {
        "perturbation_id": "process_epsilon_pulse__force_state_x__early_t5_pos",
        "channel": "process_epsilon",
        "family": "process_epsilon_force_state_xy",
        "epsilon_component": "force_state_x",
        "epsilon_index": 4,
        "amplitude": 0.25,
        "axis": "x",
        "sign": 1,
        "timing": {"epoch": "movement_indexed", "start_time_index": 5, "duration_steps": 2},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    payload = np.asarray(result.trial_specs.inputs[result.adapter_provenance["input_key"]])
    np.testing.assert_allclose(payload[0, 15:17, 4], 0.25)
    np.testing.assert_allclose(payload[1, 25:27, 4], 0.25)
    assert not np.any(payload[0, : go_steps[0], :])
    assert not np.any(payload[1, : go_steps[1], :])
    np.testing.assert_allclose(result.trial_specs.inputs["epsilon"], 0.0)
    assert result.adapter_provenance["absolute_start_time_indices"] == [15, 25]


def test_process_epsilon_adapter_uses_explicit_force_state_epsilon_index() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={
            "effector_target": CartesianState(pos=np.zeros((2, 10, 2))),
            "epsilon": np.zeros((2, 10, 8), dtype=np.float64),
        },
    )
    perturbation = {
        "channel": "process_epsilon",
        "family": "process_epsilon_force_state_xy",
        "epsilon_component": "force_state_y",
        "epsilon_index": 5,
        "amplitude": 0.25,
        "axis": "y",
        "sign": -1,
        "timing": {"start_time_index": 3, "duration_steps": 2},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    input_key = result.adapter_provenance["input_key"]
    np.testing.assert_allclose(result.trial_specs.inputs[input_key][:, 3:5, 5], -0.25)
    np.testing.assert_allclose(result.trial_specs.inputs[input_key][:, 3:5, 3], 0.0)
    np.testing.assert_allclose(result.trial_specs.inputs["epsilon"], 0.0)
    assert result.adapter_provenance["epsilon_component"] == "force_state_y"
    assert result.adapter_provenance["epsilon_index"] == 5


def test_process_epsilon_adapter_blocks_without_epsilon_input() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 10, 2)))},
    )
    perturbation = {
        "channel": "process_epsilon",
        "family": "process_epsilon_pulse",
        "amplitude": 0.25,
        "axis": "x",
        "sign": 1,
        "timing": {"start_time_index": 3, "duration_steps": 2},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "blocked"
    assert "mechanics.epsilon / B_w" in result.reason


def test_delayed_initial_state_rows_use_movement_onset_epsilon_impulses() -> None:
    go_steps = np.asarray([10, 20], dtype=np.int32)
    trial_specs = _delayed_trial_specs(go_steps, include_epsilon=True)
    perturbation = {
        "channel": "initial_state",
        "family": "initial_velocity_offset",
        "amplitude": 0.05,
        "axis": "y",
        "sign": -1,
        "timing": {"epoch": "initial_condition", "time_index": 0},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    np.testing.assert_allclose(result.trial_specs.inits["mechanics.vector"], 0.0)
    epsilon_delta = np.asarray(result.trial_specs.inputs["epsilon"] - trial_specs.inputs["epsilon"])
    np.testing.assert_allclose(epsilon_delta[0, go_steps[0], 3], -0.05)
    np.testing.assert_allclose(epsilon_delta[1, go_steps[1], 3], -0.05)
    assert not np.any(epsilon_delta[0, : go_steps[0], :])
    assert not np.any(epsilon_delta[1, : go_steps[1], :])
    assert result.adapter_provenance["adapter"] == "trial_specs.inputs.epsilon"
    assert result.adapter_provenance["movement_start_aligned"] is True


def test_immediate_initial_state_rows_remain_trial_start_init_offsets() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={
            "effector_target": CartesianState(pos=np.zeros((2, 10, 2))),
            "epsilon": np.zeros((2, 10, 8), dtype=np.float64),
        },
    )
    perturbation = {
        "channel": "initial_state",
        "family": "initial_position_offset",
        "amplitude": 0.01,
        "axis": "x",
        "sign": 1,
        "timing": {"epoch": "initial_condition", "time_index": 0},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    np.testing.assert_allclose(result.trial_specs.inits["mechanics.vector"][:, 0], 0.01)
    np.testing.assert_allclose(result.trial_specs.inputs["epsilon"], 0.0)
    assert result.adapter_provenance["adapter"] == "trial_specs.inits.*[pos_vel_vector]"


def test_sensory_adapter_uses_external_graph_channel_payload() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 10, 2)))},
    )
    perturbation = {
        "perturbation_id": "sensory_feedback_offset__x_pos",
        "channel": "sensory_feedback",
        "family": "sensory_feedback_offset",
        "amplitude": 0.01,
        "axis": "x",
        "sign": 1,
        "timing": {"start_time_index": 0, "duration_steps": 10},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    input_key = f"{GRAPH_ADAPTER_INPUT_PREFIX}.sensory_feedback.sensory_feedback_offset__x_pos"
    payload = result.trial_specs.inputs[input_key]
    assert payload.shape == (2, 10, 4)
    np.testing.assert_allclose(payload[:, :, 0], 0.01)
    assert result.adapter_provenance["effective_payload_dim"] == 4
    assert result.adapter_provenance["active_calibrated_components"] == 4
    assert result.adapter_provenance["payload_shape_source"] == "adapter_spec"
    assert result.adapter_provenance["insertion_point"] == "sensory.output -> net.feedback"
    assert result.adapter_provenance["controller_input_mutated"] is False


def test_sensory_adapter_applies_force_filter_feedback_row_to_payload_index() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=4,
        key=jr.PRNGKey(0),
        target_relative_feedback=True,
        force_filter_feedback=True,
    )
    graph = install_perturbation_training_graph_adapters(graph, force_filter_feedback=True)
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={
            "mechanics.effector.pos": TargetSpec(
                value=np.zeros((2, 10, 2), dtype=np.float32),
            )
        },
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 10, 2)))},
        timeline=TrialTimeline(n_steps=10),
    )
    trial_specs = add_zero_graph_channel_inputs(trial_specs, force_filter_feedback=True)
    specs = graph_adapter_specs(force_filter_feedback=True)
    perturbation = {
        "perturbation_id": "sensory_feedback_offset__force_filter__x_pos",
        "channel": "sensory_feedback",
        "family": "sensory_feedback_offset",
        "amplitude": 0.25,
        "units": "N",
        "axis": "x",
        "sign": 1,
        "timing": {"start_time_index": 0, "duration_steps": 10},
        "channel_provenance": {
            "feedback_quantity": "force_filter",
            "feedback_payload_index": 4,
            "force_filter_feedback_only": True,
        },
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation, model=graph)

    assert result.status == "evaluated"
    payload = result.trial_specs.inputs[specs["sensory_feedback"].input_key]
    assert payload.shape == (2, 10, 6)
    np.testing.assert_allclose(payload[:, :, :4], 0.0)
    np.testing.assert_allclose(payload[:, :, 4], 0.25)
    np.testing.assert_allclose(payload[:, :, 5], 0.0)
    assert result.adapter_provenance["graph_adapter_reused"] is True
    assert result.adapter_provenance["declared_payload_dim"] == 4
    assert result.adapter_provenance["effective_payload_dim"] == 6
    assert result.adapter_provenance["active_calibrated_components"] == 6
    assert result.adapter_provenance["payload_shape_source"] == "existing_trial_input"


def test_delayed_observation_adapter_applies_force_filter_feedback_row_to_payload_index() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=4,
        key=jr.PRNGKey(0),
        target_relative_feedback=True,
        force_filter_feedback=True,
    )
    graph = install_perturbation_training_graph_adapters(graph, force_filter_feedback=True)
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={
            "mechanics.effector.pos": TargetSpec(
                value=np.zeros((2, 10, 2), dtype=np.float32),
            )
        },
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 10, 2)))},
        timeline=TrialTimeline(n_steps=10),
    )
    trial_specs = add_zero_graph_channel_inputs(trial_specs, force_filter_feedback=True)
    specs = graph_adapter_specs(force_filter_feedback=True)
    perturbation = {
        "perturbation_id": "delayed_observation_offset__force_filter__y_neg",
        "channel": "delayed_observation",
        "family": "delayed_observation_offset",
        "amplitude": 0.5,
        "units": "N",
        "axis": "y",
        "sign": -1,
        "timing": {"start_time_index": 0, "duration_steps": 10},
        "channel_provenance": {
            "feedback_quantity": "force_filter",
            "feedback_payload_index": 5,
            "force_filter_feedback_only": True,
        },
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation, model=graph)

    assert result.status == "evaluated"
    payload = result.trial_specs.inputs[specs["delayed_observation"].input_key]
    assert payload.shape == (2, 10, 6)
    np.testing.assert_allclose(payload[:, :, :5], 0.0)
    np.testing.assert_allclose(payload[:, :, 5], -0.5)
    assert result.adapter_provenance["graph_adapter_reused"] is True
    assert result.adapter_provenance["declared_payload_dim"] == 4
    assert result.adapter_provenance["effective_payload_dim"] == 6
    assert result.adapter_provenance["active_calibrated_components"] == 6
    assert result.adapter_provenance["payload_shape_source"] == "existing_trial_input"


def test_extlqg_6d_context_skips_8d_only_process_epsilon_rows() -> None:
    context = perturbation_bank._build_extlqg_comparator_context(physical_dim=6)

    assert context["physical_dim"] == 6
    assert context["plant"].n == 36
    assert context["plant"].m_w == 6
    assert context["config"].n_phys == 6
    assert getattr(context["base_evaluation"], "mechanics_vector").shape[-1] == 36

    result = evaluate_extlqg_perturbation_comparator(
        {
            "perturbation_id": "process_epsilon_pulse__integrator_x_pos",
            "channel": "process_epsilon",
            "family": "process_epsilon_pulse",
            "amplitude": 0.01,
            "axis": "x",
            "sign": 1,
            "timing": {"start_time_index": 0, "duration_steps": 1},
            "epsilon_index": 6,
        },
        context=context,
        gru_metrics={},
    )

    assert result["status"] == "not_applicable"
    assert "epsilon_index 6" in result["reason"]
    assert "6 process disturbance dimensions" in result["reason"]


def test_extlqg_observation_offset_flips_target_relative_position_and_velocity() -> None:
    position_perturbation = {
        "perturbation_id": "sensory_feedback_offset__position_small__early_t10_x_pos",
        "channel": "sensory_feedback",
        "family": "sensory_feedback_offset",
        "amplitude": 0.25,
        "axis": "x",
        "sign": 1,
        "timing": {"start_time_index": 1, "duration_steps": 2},
        "channel_provenance": {
            "feedback_quantity": "position",
            "feedback_payload_index": 0,
        },
    }
    velocity_perturbation = {
        **position_perturbation,
        "perturbation_id": "sensory_feedback_offset__velocity_small__early_t10_vx_neg",
        "amplitude": 0.5,
        "axis": "vx",
        "sign": -1,
        "channel_provenance": {
            "feedback_quantity": "velocity",
            "feedback_payload_index": 2,
        },
    }

    position_offset = perturbation_bank._extlqg_observation_offset(
        position_perturbation,
        horizon=5,
        observation_dim=6,
    )
    velocity_offset = perturbation_bank._extlqg_observation_offset(
        velocity_perturbation,
        horizon=5,
        observation_dim=6,
    )

    np.testing.assert_allclose(np.asarray(position_offset)[1:3, 0], -0.25)
    np.testing.assert_allclose(np.asarray(position_offset)[:1], 0.0)
    np.testing.assert_allclose(np.asarray(position_offset)[3:], 0.0)
    np.testing.assert_allclose(np.asarray(velocity_offset)[1:3, 2], 0.5)
    np.testing.assert_allclose(np.asarray(velocity_offset)[:, :2], 0.0)
    np.testing.assert_allclose(np.asarray(velocity_offset)[:, 3:], 0.0)


def test_extlqg_observation_offset_uses_force_filter_feedback_payload_index() -> None:
    perturbation = {
        "perturbation_id": "sensory_feedback_offset__force_filter__late_t40_y_pos",
        "channel": "sensory_feedback",
        "family": "sensory_feedback_offset",
        "amplitude": 0.5,
        "axis": "y",
        "sign": 1,
        "timing": {"start_time_index": 0, "duration_steps": 2},
        "channel_provenance": {
            "feedback_quantity": "force_filter",
            "feedback_payload_index": 5,
            "force_filter_feedback_only": True,
        },
    }

    offset = perturbation_bank._extlqg_observation_offset(
        perturbation,
        horizon=4,
        observation_dim=6,
    )

    np.testing.assert_allclose(np.asarray(offset)[:2, 5], 0.5)
    np.testing.assert_allclose(np.asarray(offset)[:2, :5], 0.0)
    np.testing.assert_allclose(np.asarray(offset)[2:], 0.0)


def test_movement_start_indices_use_zero_for_single_movement_epoch() -> None:
    trial_specs = TaskTrialSpec(
        inits={},
        targets={},
        inputs={},
        timeline=TrialTimeline(
            n_steps=60,
            epoch_bounds=np.asarray([[0, 60], [0, 60]], dtype=np.int32),
            epoch_names=("movement",),
        ),
    )

    starts = perturbation_bank._movement_start_indices(trial_specs, batch_size=2)

    np.testing.assert_array_equal(starts, np.asarray([0, 0], dtype=np.int64))
    assert (
        perturbation_bank._movement_start_source(trial_specs)
        == "trial_specs.timeline.epoch_bounds[..., 0]"
    )


def test_delayed_observation_adapter_uses_clean_pre_noise_graph_channel() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 10, 2)))},
    )
    perturbation = {
        "perturbation_id": "delayed_observation_offset__x_pos",
        "channel": "delayed_observation",
        "family": "delayed_observation_offset",
        "amplitude": 0.01,
        "axis": "x",
        "sign": 1,
        "timing": {"start_time_index": 0, "duration_steps": 10},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    payload = result.trial_specs.inputs[
        f"{GRAPH_ADAPTER_INPUT_PREFIX}.delayed_observation.delayed_observation_offset__x_pos"
    ]
    assert payload.shape == (2, 10, 4)
    np.testing.assert_allclose(payload[:, :, 0], 0.01)
    assert result.adapter_provenance["insertion_point"] == "feedback.feedback -> sensory.input"
    metadata = result.adapter_provenance["feedbax_additive_channel_adapter"]["metadata"]
    assert "before sensory.input noise" in metadata["graphspec_mapping"]
    assert "pre_noise_delayed_measurement" in result.adapter_provenance[
        "metadata"
    ][
        "graphspec_mapping"
    ]
    assert "compatibility alias" in metadata["graphspec_mapping"]


def test_full_qrf_cost_scorer_reports_control_and_delta_breakdown() -> None:
    _plant, schedule = build_canonical_game()
    states = np.zeros((1, schedule.T, schedule.Q.shape[-1]), dtype=np.float64)
    initial = np.zeros((1, schedule.Q.shape[-1]), dtype=np.float64)
    base_commands = np.zeros((1, schedule.T, schedule.R.shape[-1]), dtype=np.float64)
    perturbed_commands = np.ones_like(base_commands)

    base = score_full_qrf_rollout_cost(
        states=states,
        commands=base_commands,
        initial_states=initial,
        target_pos=np.zeros((2,), dtype=np.float64),
    )
    perturbed = score_full_qrf_rollout_cost(
        states=states,
        commands=perturbed_commands,
        initial_states=initial,
        target_pos=np.zeros((2,), dtype=np.float64),
    )
    delta = delta_full_qrf_cost_summary(
        {
            "status": "available",
            "lens": base["lens"],
            "basis": base["basis"],
            "total": {"values": base["total"].tolist()},
            "stage_state": {"values": base["stage_state"].tolist()},
            "control": {"values": base["control"].tolist()},
            "terminal": {"values": base["terminal"].tolist()},
        },
        {
            "status": "available",
            "lens": perturbed["lens"],
            "basis": perturbed["basis"],
            "total": {"values": perturbed["total"].tolist()},
            "stage_state": {"values": perturbed["stage_state"].tolist()},
            "control": {"values": perturbed["control"].tolist()},
            "terminal": {"values": perturbed["terminal"].tolist()},
        },
    )

    assert base["status"] == "available"
    np.testing.assert_allclose(base["total"], 0.0)
    np.testing.assert_allclose(perturbed["stage_state"], 0.0)
    np.testing.assert_allclose(perturbed["terminal"], 0.0)
    np.testing.assert_allclose(perturbed["control"], 2.0 * schedule.T)
    assert delta["status"] == "available"
    assert delta["delta_cost"]["control"]["mean"] == 2.0 * schedule.T


def test_full_qrf_cost_scorer_keeps_internal_arrays_device_backed() -> None:
    _plant, schedule = build_canonical_game()
    states = jnp.zeros((1, 1, schedule.T, schedule.Q.shape[-1]), dtype=jnp.float64)
    commands = jnp.ones((1, 1, schedule.T, schedule.R.shape[-1]), dtype=jnp.float64)
    initial = jnp.zeros((1, schedule.Q.shape[-1]), dtype=jnp.float64)

    scored = score_full_qrf_rollout_cost(
        states=states,
        commands=commands,
        initial_states=initial,
        target_pos=jnp.zeros((2,), dtype=jnp.float64),
    )
    evaluation = RolloutEvaluation(
        position=jnp.zeros((1, 1, schedule.T, 2), dtype=jnp.float64),
        velocity=jnp.zeros((1, 1, schedule.T, 2), dtype=jnp.float64),
        command=commands,
        hidden=jnp.zeros((1, 1, schedule.T, 1), dtype=jnp.float64),
        gru_input=jnp.zeros((1, 1, schedule.T, 1), dtype=jnp.float64),
        initial_position=jnp.zeros((1, 2), dtype=jnp.float64),
        initial_velocity=jnp.zeros((1, 2), dtype=jnp.float64),
        target_position=jnp.zeros((1, schedule.T, 2), dtype=jnp.float64),
        dt=0.01,
    )
    object.__setattr__(evaluation, "mechanics_vector", states)
    summary = full_qrf_cost_summary(
        evaluation,
        TaskTrialSpec(
            inits={"mechanics.vector": initial},
            inputs={},
            targets={},
        ),
    )

    assert hasattr(scored["total"], "block_until_ready")
    assert hasattr(scored["timewise_control"], "block_until_ready")
    assert summary["status"] == "available"
    assert summary["control"]["values"] == np.asarray(scored["control"]).tolist()
    np.testing.assert_allclose(summary["control"]["mean"], 2.0 * schedule.T)


def test_full_qrf_cost_scorer_supports_no_integrator_6d_delayed_basis() -> None:
    _plant, schedule = build_no_integrator_game()
    states = np.zeros((1, schedule.T, schedule.Q.shape[-1]), dtype=np.float64)
    initial = np.zeros((1, schedule.Q.shape[-1]), dtype=np.float64)
    commands = np.ones((1, schedule.T, schedule.R.shape[-1]), dtype=np.float64)

    scored = score_full_qrf_rollout_cost(
        states=states,
        commands=commands,
        initial_states=initial,
        target_pos=np.zeros((2,), dtype=np.float64),
    )

    assert scored["status"] == "available"
    assert scored["basis"]["physical_state_dim"] == 6
    assert scored["basis"]["schedule_source"].endswith("build_no_integrator_game")
    np.testing.assert_allclose(scored["stage_state"], 0.0)
    np.testing.assert_allclose(scored["terminal"], 0.0)
    np.testing.assert_allclose(scored["control"], 2.0 * schedule.T)


def test_full_qrf_cost_summary_slices_delayed_movement_window() -> None:
    _plant, schedule = build_canonical_game()
    states = np.zeros((1, 1, 90, schedule.Q.shape[-1]), dtype=np.float64)
    commands = np.zeros((1, 1, 90, schedule.R.shape[-1]), dtype=np.float64)
    initial = np.zeros((1, schedule.Q.shape[-1]), dtype=np.float64)
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": initial},
        inputs={},
        targets={},
        timeline=TrialTimeline(
            epoch_bounds=np.asarray([[0, 30, 90]], dtype=np.int32),
            epoch_names=("prep", "movement"),
            event_steps=np.asarray([[30]], dtype=np.int32),
        ),
    )
    evaluation = RolloutEvaluation(
        position=np.zeros((1, 1, 90, 2), dtype=np.float64),
        velocity=np.zeros((1, 1, 90, 2), dtype=np.float64),
        command=commands,
        hidden=np.zeros((1, 1, 90, 1), dtype=np.float64),
        gru_input=np.zeros((1, 1, 90, 1), dtype=np.float64),
        initial_position=np.zeros((1, 2), dtype=np.float64),
        initial_velocity=np.zeros((1, 2), dtype=np.float64),
        target_position=np.zeros((1, 90, 2), dtype=np.float64),
        dt=0.01,
    )
    object.__setattr__(evaluation, "mechanics_vector", states)

    summary = full_qrf_cost_summary(evaluation, trial_specs)

    assert summary["status"] == "available"
    assert summary["basis"]["time_window"]["basis"] == "timeline_epoch_bounds_movement_window"
    assert summary["basis"]["time_window"]["start"] == 30
    assert summary["basis"]["time_window"]["stop"] == 90
    assert summary["timewise_control"]["count"] == 60


def test_perturbation_response_reports_controller_io_metrics() -> None:
    zeros_pos = np.zeros((1, 1, 3, 2), dtype=np.float64)
    target = np.zeros((1, 3, 2), dtype=np.float64)
    base = RolloutEvaluation(
        position=zeros_pos,
        velocity=zeros_pos,
        command=np.zeros((1, 1, 3, 2), dtype=np.float64),
        hidden=np.zeros((1, 1, 3, 4), dtype=np.float64),
        gru_input=np.zeros((1, 1, 3, 3), dtype=np.float64),
        initial_position=np.zeros((1, 2), dtype=np.float64),
        initial_velocity=np.zeros((1, 2), dtype=np.float64),
        target_position=target,
        dt=0.01,
    )
    perturbed = RolloutEvaluation(
        position=zeros_pos,
        velocity=zeros_pos,
        command=np.ones((1, 1, 3, 2), dtype=np.float64),
        hidden=np.zeros((1, 1, 3, 4), dtype=np.float64),
        gru_input=np.ones((1, 1, 3, 3), dtype=np.float64),
        initial_position=np.zeros((1, 2), dtype=np.float64),
        initial_velocity=np.zeros((1, 2), dtype=np.float64),
        target_position=target,
        dt=0.01,
    )

    metrics = summarize_perturbation_response(base, perturbed)

    io = metrics["controller_io_response"]
    assert io["status"] == "available"
    assert io["input_key"] == "states.net.input"
    assert io["output_key"] == "states.net.output"
    np.testing.assert_allclose(io["delta_input_norm"]["mean"], np.sqrt(3.0))
    np.testing.assert_allclose(io["action_per_input_gain"]["mean"], np.sqrt(2.0) / np.sqrt(3.0))


def test_perturbation_response_reports_v3_shape_and_alignment_metrics() -> None:
    target = np.broadcast_to(
        np.asarray([10.0, 0.0], dtype=np.float64),
        (1, 4, 2),
    )
    base = RolloutEvaluation(
        position=np.zeros((1, 1, 4, 2), dtype=np.float64),
        velocity=np.zeros((1, 1, 4, 2), dtype=np.float64),
        command=np.zeros((1, 1, 4, 2), dtype=np.float64),
        hidden=np.zeros((1, 1, 4, 4), dtype=np.float64),
        gru_input=np.zeros((1, 1, 4, 3), dtype=np.float64),
        initial_position=np.zeros((1, 2), dtype=np.float64),
        initial_velocity=np.zeros((1, 2), dtype=np.float64),
        target_position=target,
        dt=0.5,
    )
    perturbed_position = np.zeros((1, 1, 4, 2), dtype=np.float64)
    perturbed_position[0, 0, :, 0] = [0.0, 1.0, 3.0, 0.0]
    perturbed_action = np.zeros((1, 1, 4, 2), dtype=np.float64)
    perturbed_action[0, 0, :, 1] = [0.0, 2.0, 4.0, 0.0]
    perturbed = RolloutEvaluation(
        position=perturbed_position,
        velocity=np.ones((1, 1, 4, 2), dtype=np.float64),
        command=perturbed_action,
        hidden=np.zeros((1, 1, 4, 4), dtype=np.float64),
        gru_input=np.ones((1, 1, 4, 3), dtype=np.float64),
        initial_position=np.zeros((1, 2), dtype=np.float64),
        initial_velocity=np.zeros((1, 2), dtype=np.float64),
        target_position=target,
        dt=0.5,
    )

    metrics = summarize_perturbation_response(base, perturbed)

    assert metrics["delta_position_response_m"]["status"] == "available"
    assert metrics["delta_position_response_m"]["max"]["mean"] == 3.0
    assert metrics["delta_position_response_m"]["auc"]["mean"] == 2.0
    assert metrics["delta_action_response"]["max"]["mean"] == 4.0
    assert metrics["delta_action_response"]["auc"]["mean"] == 3.0
    assert metrics["response_shape"]["peak_time_s"]["mean"] == 1.0
    assert metrics["response_shape"]["recovery_time_s"]["mean"] == 1.5
    assert metrics["response_shape"]["n_unrecovered"] == 0
    alignment = metrics["target_relative_alignment"]
    assert alignment["status"] == "available"
    assert alignment["delta_position"]["abs_radial_component"]["max"] == 3.0
    assert alignment["delta_position"]["abs_tangential_component"]["max"] == 0.0
    assert alignment["delta_action"]["abs_radial_component"]["max"] == 0.0
    assert alignment["delta_action"]["abs_tangential_component"]["max"] == 4.0


def test_perturbation_bank_summary_reports_ratio_of_means_and_signed_pairs() -> None:
    rows = [
        {
            "perturbation_id": "initial_position_offset__x_pos",
            "channel": "initial_state",
            "family": "initial_position_offset",
            "axis": "x",
            "sign": 1,
            "amplitude": 0.5,
            "timing": {"time_index": 0},
            "status": "evaluated",
            "metrics": {
                "delta_action_norm": {"mean": 6.0},
                "delta_endpoint_error_m": {"mean": 4.0},
                "controller_io_response": {
                    "status": "available",
                    "delta_input_norm": {"mean": 3.0},
                    "action_per_input_gain": {"mean": 2.0},
                },
                "extra_full_qrf_cost": {
                    "delta_cost": {
                        "total": {"mean": 8.0},
                    },
                },
            },
            "extlqg_comparator": {
                "status": "available",
                "reference_response_metrics": {
                    "delta_action_norm": {"mean": 2.0},
                    "delta_endpoint_error_m": {"mean": 1.0},
                    "controller_io_response": {
                        "status": "not_available",
                    },
                    "extra_full_qrf_cost": {
                        "delta_cost": {
                            "total": {"mean": 0.5},
                        },
                    },
                },
            },
        },
        {
            "perturbation_id": "initial_position_offset__x_neg",
            "channel": "initial_state",
            "family": "initial_position_offset",
            "axis": "x",
            "sign": -1,
            "amplitude": 0.5,
            "timing": {"time_index": 0},
            "status": "evaluated",
            "metrics": {
                "delta_action_norm": {"mean": 2.0},
                "delta_endpoint_error_m": {"mean": 2.0},
                "controller_io_response": {
                    "status": "available",
                    "delta_input_norm": {"mean": 1.0},
                    "action_per_input_gain": {"mean": 1.0},
                },
                "extra_full_qrf_cost": {
                    "delta_cost": {
                        "total": {"mean": 4.0},
                    },
                },
            },
            "extlqg_comparator": {
                "status": "available",
                "reference_response_metrics": {
                    "delta_action_norm": {"mean": 2.0},
                    "delta_endpoint_error_m": {"mean": 1.0},
                    "extra_full_qrf_cost": {
                        "delta_cost": {
                            "total": {"mean": 0.25},
                        },
                    },
                },
            },
        },
    ]

    summary = summarize_perturbation_bank(rows)

    assert summary["status"] == "available"
    ratio = summary["ratio_of_means"]["initial_state/initial_position_offset"]["metrics"]
    assert ratio["delta_action_norm"]["ratio_of_means"] == 2.0
    assert ratio["extra_full_qrf_cost.delta_cost.total"]["inflated_ratio"] is True
    assert ratio["extra_full_qrf_cost.delta_cost.total"]["raw_numerator_values"] == [8.0, 4.0]
    assert ratio["extra_full_qrf_cost.delta_cost.total"]["raw_denominator_values"] == [0.5, 0.25]
    signed = summary["signed_pair_response"]
    assert signed["status"] == "available"
    assert signed["n_pairs"] == 1
    endpoint = signed["pairs"][0]["metrics"]["delta_endpoint_error_m"]
    assert endpoint["odd_response"] == 1.0
    assert endpoint["even_nonlinear_residual"] == 3.0
    assert endpoint["curvature_like_symmetric_response"] == 12.0
    assert summary["controller_io_response"]["status"] == "available"
    assert summary["controller_io_response"]["delta_input_norm"]["mean"] == 2.0


def test_perturbation_bank_summary_reports_class_bins_and_na_ratios() -> None:
    rows = [
        {
            "perturbation_id": "initial_position_offset__x_pos",
            "channel": "initial_state",
            "family": "initial_position_offset",
            "axis": "x",
            "sign": 1,
            "amplitude": 0.5,
            "timing": {"time_index": 0},
            "status": "evaluated",
            "metrics": {
                "delta_action_norm": {"mean": 6.0},
                "delta_position_trajectory_norm_m": {"mean": 0.4},
                "delta_velocity_trajectory_norm_m_s": {"mean": 0.3},
                "delta_state_trajectory_norm": {"mean": 0.5},
                "delta_position_response_m": {
                    "max": {"mean": 0.7},
                    "auc": {"mean": 0.8},
                },
                "delta_state_response": {
                    "max": {"mean": 0.9},
                    "auc": {"mean": 1.0},
                },
                "delta_action_response": {
                    "max": {"mean": 6.5},
                    "auc": {"mean": 6.75},
                },
                "response_shape": {
                    "peak_time_s": {"mean": 0.2},
                    "recovery_time_s": {"mean": 0.6},
                },
                "target_relative_alignment": {
                    "delta_position": {
                        "abs_radial_component": {"mean": 0.35},
                        "abs_tangential_component": {"mean": 0.05},
                    },
                },
                "delta_endpoint_error_m": {"mean": 0.2},
                "delta_terminal_speed_m_s": {"mean": 0.1},
                "extra_full_qrf_cost": {
                    "delta_cost": {
                        "total": {"mean": 9.0},
                    },
                },
            },
            "extlqg_comparator": {
                "status": "available",
                "reference_response_metrics": {
                    "extra_full_qrf_cost": {
                        "delta_cost": {
                            "total": {"mean": 3.0},
                        },
                    },
                },
            },
        },
        {
            "perturbation_id": "command_input_pulse__t3_x_pos",
            "channel": "command_input",
            "family": "command_input_pulse",
            "axis": "x",
            "sign": 1,
            "amplitude": 0.25,
            "timing_bin": "early",
            "timing": {"start_time_index": 3, "duration_steps": 2},
            "status": "evaluated",
            "metrics": {
                "delta_action_norm": {"mean": 1.5},
                "delta_position_trajectory_norm_m": {"mean": 0.2},
                "delta_velocity_trajectory_norm_m_s": {"mean": 0.1},
                "delta_endpoint_error_m": {"mean": 0.05},
                "delta_terminal_speed_m_s": {"mean": 0.025},
                "extra_full_qrf_cost": {
                    "delta_cost": {
                        "total": {"mean": 4.0},
                    },
                },
            },
            "extlqg_comparator": extlqg_comparator_status(
                {"channel": "command_input", "family": "command_input_pulse"},
                status="not_applicable",
            ),
        },
        {
            "perturbation_id": "target_stream_jump__x_pos",
            "channel": "target_stream",
            "family": "target_stream_jump",
            "status": "not_applicable",
            "reason": "fixed-target checkpoints do not expose a target stream",
            "extlqg_comparator": extlqg_comparator_status(
                {"channel": "target_stream", "family": "target_stream_jump"},
                status="not_applicable",
            ),
        },
    ]

    summary = summarize_perturbation_bank(rows)
    class_summary = summary["class_summary"]["groups"]

    initial = class_summary["initial_state/initial_position_offset"]
    assert initial["n_rows"] == 1
    assert initial["status_counts"] == {"evaluated": 1}
    assert initial["metrics"]["delta_action_norm"]["mean"] == 6.0
    assert initial["metrics"]["delta_position_response_m.max"]["mean"] == 0.7
    assert initial["metrics"]["delta_position_response_m.auc"]["mean"] == 0.8
    assert initial["metrics"]["delta_state_response.max"]["mean"] == 0.9
    assert initial["metrics"]["delta_action_response.max"]["mean"] == 6.5
    assert initial["metrics"]["response_shape.peak_time_s"]["mean"] == 0.2
    assert initial["metrics"][
        "target_relative_alignment.delta_position.abs_radial_component"
    ]["mean"] == 0.35
    assert initial["metrics"]["extra_full_qrf_delta_cost_total"]["mean"] == 9.0
    assert initial["gru_extlqg_delta_cost_ratio"]["ratio_of_means"] == 3.0

    command = class_summary["command_input/command_input_pulse"]
    assert command["gru_extlqg_delta_cost_ratio"]["status"] == "not_available"
    assert "no meaningful extLQG" in command["gru_extlqg_delta_cost_ratio"]["reason"]
    assert command["extlqg_not_applicable_reasons"]
    timing_cells = summary["timing_cell_summary"]["groups"]
    assert timing_cells["command_input/command_input_pulse/early"]["timing_bin"] == "early"
    assert timing_cells["command_input/command_input_pulse/early"]["n_rows"] == 1
    assert (
        summary["ratio_of_means_by_timing"]["command_input/command_input_pulse/early"][
            "metrics"
        ]["delta_action_norm"]["status"]
        == "not_available"
    )

    target = class_summary["target_stream/target_stream_jump"]
    assert target["status_counts"] == {"not_applicable": 1}
    assert target["metrics"]["delta_action_norm"]["status"] == "not_available"
    assert target["not_applicable_reasons"] == {
        "fixed-target checkpoints do not expose a target stream": 1
    }


def test_perturbation_markdown_renders_class_binned_summary() -> None:
    rows = [
        {
            "perturbation_id": "command_input_pulse__t3_x_pos",
            "channel": "command_input",
            "family": "command_input_pulse",
            "axis": "x",
            "sign": 1,
            "amplitude": 0.25,
            "timing": {"start_time_index": 3, "duration_steps": 2},
            "status": "evaluated",
            "metrics": {
                "delta_action_norm": {"mean": 1.5},
                "delta_position_trajectory_norm_m": {"mean": 0.2},
                "delta_velocity_trajectory_norm_m_s": {"mean": 0.1},
                "delta_endpoint_error_m": {"mean": 0.05},
                "delta_terminal_speed_m_s": {"mean": 0.025},
                "extra_full_qrf_cost": {
                    "delta_cost": {
                        "total": {"mean": 4.0},
                    },
                },
            },
            "extlqg_comparator": extlqg_comparator_status(
                {"channel": "command_input", "family": "command_input_pulse"},
                status="not_applicable",
            ),
        },
        {
            "perturbation_id": "target_stream_jump__x_pos",
            "channel": "target_stream",
            "family": "target_stream_jump",
            "status": "not_applicable",
            "reason": "fixed-target checkpoints do not expose a target stream",
            "extlqg_comparator": extlqg_comparator_status(
                {"channel": "target_stream", "family": "target_stream_jump"},
                status="not_applicable",
            ),
        },
    ]
    manifest = {
        "issue": "3992394",
        "source_experiment": "aacb9ed",
        "semantics_correction": "",
        "bank": {
            "perturbations": [
                {"channel": row["channel"], "family": row["family"]} for row in rows
            ],
        },
        "runs": {
            "synthetic": {
                "status_counts": {"evaluated": 1, "not_applicable": 1},
                "n_rollout_trials_per_replicate": 1,
                "robust_response_summary": summarize_perturbation_bank(rows),
            },
        },
        "extlqg_comparator": {"status": "available", "reason": "synthetic"},
        "full_qrf_cost": {"status": "available", "reason": "synthetic"},
    }

    markdown = render_perturbation_response_markdown(manifest)

    assert "#### Class-Binned Summary" in markdown
    assert "`command_input/command_input_pulse`" in markdown
    assert "no meaningful extLQG" in markdown
    assert "fixed-target checkpoints do not expose a target stream" in markdown


def test_extlqg_comparator_status_defers_target_stream_for_fixed_target_rows() -> None:
    perturbation = {
        "channel": "target_stream",
        "family": "target_stream_jump",
        "perturbation_id": "target_stream_jump__x_pos",
    }

    status = extlqg_comparator_status(perturbation, status="not_applicable")

    assert status["status"] == "not_applicable"
    assert "fixed-target checkpoints" in status["reason"]
    assert status["selection_role"] == "audit_only_not_used_for_checkpoint_selection"


def test_extlqg_comparator_evaluates_sensory_and_delayed_observation_offsets(
    monkeypatch,
) -> None:
    base = _minimal_rollout_evaluation(command_value=0.0)
    perturbed = _minimal_rollout_evaluation(command_value=1.0)
    initial_state = np.zeros((48,), dtype=np.float64)
    calls = []

    def fake_simulate_extlqg_perturbed(perturbation, *, context):
        calls.append(perturbation["channel"])
        adapter = {
            "adapter": f"fake_{perturbation['channel']}",
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
        }
        return perturbed, initial_state, adapter

    def fake_extlqg_cost_summary(evaluation, initial_state):
        del evaluation, initial_state
        cost = {"values": [0.0]}
        return {
            "status": "available",
            "total": cost,
            "stage_state": cost,
            "control": cost,
            "terminal": cost,
        }

    monkeypatch.setattr(
        perturbation_bank,
        "_simulate_extlqg_perturbed",
        fake_simulate_extlqg_perturbed,
    )
    monkeypatch.setattr(perturbation_bank, "_extlqg_cost_summary", fake_extlqg_cost_summary)
    context = {
        "base_evaluation": base,
        "base_initial_state": initial_state,
        "parity_status": "test",
        "n_iterations": 0,
    }

    for channel, family, expected_adapter in (
        ("sensory_feedback", "sensory_feedback_offset", "fake_sensory_feedback"),
        ("delayed_observation", "delayed_observation_offset", "fake_delayed_observation"),
    ):
        comparator = evaluate_extlqg_perturbation_comparator(
            {
                "channel": channel,
                "family": family,
                "axis": "x",
                "amplitude": 0.01,
                "sign": 1,
                "timing": {"start_time_index": 0, "duration_steps": 2},
            },
            context=context,
            gru_metrics={"delta_action_norm": {"mean": 1.0}},
        )

        assert comparator["status"] == "available"
        assert comparator["analytical_adapter"]["adapter"] == expected_adapter
        assert comparator["analytical_adapter"]["controller_internal_state_mutated"] is False

    assert calls == ["sensory_feedback", "delayed_observation"]


def test_extlqg_comparator_requires_context_for_command_input() -> None:
    comparator = evaluate_extlqg_perturbation_comparator(
        {
            "channel": "command_input",
            "family": "command_input_pulse",
            "axis": "x",
            "amplitude": 1.0,
            "sign": 1,
            "timing": {"start_time_index": 0, "duration_steps": 1},
        },
        context={},
        gru_metrics={},
    )

    assert comparator["status"] == "blocked"
    assert "requires extLQG comparator context" in comparator["reason"]


def test_robust_output_feedback_comparator_reports_available_and_not_applicable(
    monkeypatch,
) -> None:
    base = _minimal_rollout_evaluation(command_value=0.0)
    perturbed = _minimal_rollout_evaluation(command_value=2.0)
    initial_state = np.zeros((48,), dtype=np.float64)

    def fake_simulate_robust_perturbed(perturbation, *, context):
        del context
        return (
            perturbed,
            initial_state,
            {
                "adapter": f"robust_fake_{perturbation['channel']}",
                "controller_input_mutated": False,
                "controller_internal_state_mutated": False,
            },
        )

    def fake_extlqg_cost_summary(evaluation, initial_state):
        del evaluation, initial_state
        cost = {"values": [0.0]}
        return {
            "status": "available",
            "total": cost,
            "stage_state": cost,
            "control": cost,
            "terminal": cost,
        }

    monkeypatch.setattr(
        perturbation_bank,
        "_simulate_robust_output_feedback_perturbed",
        fake_simulate_robust_perturbed,
    )
    monkeypatch.setattr(perturbation_bank, "_extlqg_cost_summary", fake_extlqg_cost_summary)
    context = {
        "base_evaluation": base,
        "base_initial_state": initial_state,
        "plant": object(),
        "schedule": object(),
        "config": object(),
        "solution": object(),
        "gains": object(),
        "gamma_factor": 1.4,
        "gamma": 2.0,
    }

    comparator = evaluate_robust_output_feedback_perturbation_comparator(
        {
            "channel": "command_input",
            "family": "target_aligned_lateral_command_load_pulse",
            "axis": "y",
            "amplitude": 1.0,
            "sign": 1,
            "timing": {"start_time_index": 0, "duration_steps": 1},
        },
        context=context,
        gru_metrics={"delta_action_norm": {"mean": 1.0}},
    )

    assert comparator["status"] == "available"
    assert comparator["gamma_factor"] == 1.4
    assert comparator["analytical_adapter"]["adapter"] == "robust_fake_command_input"
    assert "gru_vs_robust_analytical" in comparator

    sensory_status = robust_output_feedback_comparator_status(
        {"channel": "sensory_feedback", "family": "sensory_feedback_offset"},
        status="not_applicable",
    )
    assert sensory_status["status"] == "not_applicable"
    assert "measurement-offset ports" in sensory_status["reason"]


def test_perturbation_bulk_writer_materializes_jax_arrays_and_schema_keys(tmp_path) -> None:
    base = _minimal_rollout_evaluation(command_value=0.0)
    perturbed = _minimal_rollout_evaluation(command_value=1.0)
    base = RolloutEvaluation(
        position=jnp.asarray(base.position, dtype=jnp.float64),
        velocity=jnp.asarray(base.velocity, dtype=jnp.float64),
        command=jnp.asarray(base.command, dtype=jnp.float64),
        hidden=jnp.asarray(base.hidden, dtype=jnp.float64),
        gru_input=jnp.asarray(base.gru_input, dtype=jnp.float64),
        initial_position=jnp.asarray(base.initial_position, dtype=jnp.float64),
        initial_velocity=jnp.asarray(base.initial_velocity, dtype=jnp.float64),
        target_position=jnp.asarray(base.target_position, dtype=jnp.float64),
        dt=base.dt,
    )
    perturbed = RolloutEvaluation(
        position=jnp.asarray(perturbed.position, dtype=jnp.float64),
        velocity=jnp.asarray(perturbed.velocity, dtype=jnp.float64),
        command=jnp.asarray(perturbed.command, dtype=jnp.float64),
        hidden=jnp.asarray(perturbed.hidden, dtype=jnp.float64),
        gru_input=jnp.asarray(perturbed.gru_input, dtype=jnp.float64),
        initial_position=jnp.asarray(perturbed.initial_position, dtype=jnp.float64),
        initial_velocity=jnp.asarray(perturbed.initial_velocity, dtype=jnp.float64),
        target_position=jnp.asarray(perturbed.target_position, dtype=jnp.float64),
        dt=perturbed.dt,
    )

    path = perturbation_bank._write_perturbation_bulk_arrays(
        base,
        perturbed,
        bulk_dir=tmp_path,
        perturbation_id="row_a",
    )

    with np.load(path) as archive:
        assert set(archive.files) == {
            "delta_action",
            "delta_gru_input",
            "delta_position",
            "delta_velocity",
            "base_position",
            "perturbed_position",
            "base_velocity",
            "perturbed_velocity",
            "base_action",
            "perturbed_action",
            "base_gru_input",
            "perturbed_gru_input",
        }
        assert archive["delta_action"].dtype == np.float64
        np.testing.assert_allclose(archive["delta_action"], 1.0)


def test_summarize_perturbation_response_accepts_jax_backed_rollout_with_parity() -> None:
    base = _minimal_rollout_evaluation(command_value=0.0)
    perturbed = _minimal_rollout_evaluation(command_value=1.0)
    jax_base = RolloutEvaluation(
        **{
            field: jnp.asarray(getattr(base, field), dtype=jnp.float64)
            for field in (
                "position",
                "velocity",
                "command",
                "hidden",
                "gru_input",
                "initial_position",
                "initial_velocity",
                "target_position",
            )
        },
        dt=base.dt,
    )
    jax_perturbed = RolloutEvaluation(
        **{
            field: jnp.asarray(getattr(perturbed, field), dtype=jnp.float64)
            for field in (
                "position",
                "velocity",
                "command",
                "hidden",
                "gru_input",
                "initial_position",
                "initial_velocity",
                "target_position",
            )
        },
        dt=perturbed.dt,
    )

    assert summarize_perturbation_response(jax_base, jax_perturbed) == (
        summarize_perturbation_response(base, perturbed)
    )


def _minimal_rollout_evaluation(*, command_value: float) -> RolloutEvaluation:
    position = np.zeros((1, 1, 2, 2), dtype=np.float64)
    velocity = np.zeros_like(position)
    command = np.full((1, 1, 2, 2), command_value, dtype=np.float64)
    return RolloutEvaluation(
        position=position,
        velocity=velocity,
        command=command,
        hidden=np.zeros((1, 1, 2, 0), dtype=np.float64),
        gru_input=np.zeros((1, 1, 2, 0), dtype=np.float64),
        initial_position=np.zeros((1, 2), dtype=np.float64),
        initial_velocity=np.zeros((1, 2), dtype=np.float64),
        target_position=np.zeros((1, 2, 2), dtype=np.float64),
        dt=0.01,
    )
