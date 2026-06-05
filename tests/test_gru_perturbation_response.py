"""Tests for the C&S GRU perturbation-response bank."""

from __future__ import annotations

import json

import jax.random as jr
import numpy as np
from feedbax.graph import Wire
from feedbax.state import CartesianState
from feedbax.task import TaskTrialSpec

from rlrmp.analysis.gru_perturbation_bank import (
    GRAPH_ADAPTER_INPUT_PREFIX,
    SCHEMA_VERSION,
    apply_perturbation_to_trial_specs,
    default_cs_perturbation_bank,
    delta_full_qrf_cost_summary,
    evaluate_extlqg_perturbation_comparator,
    extlqg_comparator_status,
    render_perturbation_response_markdown,
    score_full_qrf_rollout_cost,
    summarize_perturbation_bank,
    summarize_perturbation_response,
)
import rlrmp.analysis.gru_perturbation_bank as perturbation_bank
from rlrmp.analysis.gru_evaluation_diagnostics import RolloutEvaluation
from rlrmp.analysis.cs_game_card import build_canonical_game
from rlrmp.cs_lss_gru import build_cs_lss_gru_graph


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
        "delayed_observation",
        "target_stream",
    }
    assert decoded["graphspec_alignment"]["named_channels"] == [
        "initial_state",
        "command_input",
        "process_epsilon",
        "sensory_feedback",
        "delayed_observation",
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
    assert len(decoded["perturbations"]) == 75


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
    assert {
        (row["timing_bin"], row["timing"]["start_time_index"], row["timing"]["duration_steps"])
        for row in sensory_rows
    } == {("early_visible", 10, 5), ("mid_visible", 20, 5), ("late_visible", 40, 5)}
    assert {
        (row["timing_bin"], row["timing"]["start_time_index"], row["timing"]["duration_steps"])
        for row in delayed_rows
    } == {("early_visible", 10, 5), ("mid_visible", 20, 5), ("late_visible", 40, 5)}
    assert {row["channel"] for row in delayed_rows} == {"delayed_observation"}
    assert {row["semantic_family"] for row in delayed_rows} == {
        "pre_noise_delayed_measurement_offset"
    }
    assert all(
        row["channel_provenance"]["not_literal_extra_delay"] is True for row in delayed_rows
    )

    initial_rows = [row for row in rows if row["channel"] == "initial_state"]
    assert {row["timing_bin"] for row in initial_rows} == {"initial_condition"}
    assert {row["timing"]["time_index"] for row in initial_rows} == {0}
    assert bank["timing_bin_conventions"]["plant_side"][0]["start_time_index"] == 5
    assert bank["timing_bin_conventions"]["controller_visible"][0]["start_time_index"] == 10


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
    input_key = f"{GRAPH_ADAPTER_INPUT_PREFIX}:command_input_pulse__t3_y_neg"
    payload = result.trial_specs.inputs[input_key]
    np.testing.assert_allclose(payload[:, 3:5, 1], -2.0)
    np.testing.assert_allclose(payload[:, :3, :], 0.0)
    assert result.adapter_provenance["external_load_force"] is False
    assert result.adapter_provenance["insertion_point"] == "efferent.output -> mechanics.force"
    assert result.adapter_provenance["temporary_pre_graphspec"] is True
    assert result.adapter_provenance["controller_input_mutated"] is False


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
    adapter_label = result.adapter_provenance["label"]
    assert adapter_label in result.model.nodes
    assert Wire("efferent", "output", "mechanics", "force") not in result.model.wires
    assert Wire("efferent", "output", adapter_label, "signal") in result.model.wires
    assert Wire(adapter_label, "signal", "mechanics", "force") in result.model.wires
    assert result.adapter_provenance["input_key"] in result.model.input_ports
    assert result.model.input_bindings[result.adapter_provenance["input_key"]] == (
        adapter_label,
        "offset",
    )


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

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    np.testing.assert_allclose(result.trial_specs.inputs["epsilon"][:, 3:5, 0], 0.25)
    np.testing.assert_allclose(result.trial_specs.inputs["epsilon"][:, :3, :], 0.0)
    np.testing.assert_allclose(trial_specs.inputs["epsilon"], 0.0)
    assert result.adapter_provenance["process_channel"] == "LinearStateSpace.B_w"


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
    np.testing.assert_allclose(result.trial_specs.inputs["epsilon"][:, 3:5, 5], -0.25)
    np.testing.assert_allclose(result.trial_specs.inputs["epsilon"][:, 3:5, 3], 0.0)
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
    input_key = f"{GRAPH_ADAPTER_INPUT_PREFIX}:sensory_feedback_offset__x_pos"
    payload = result.trial_specs.inputs[input_key]
    assert payload.shape == (2, 10, 4)
    np.testing.assert_allclose(payload[:, :, 0], 0.01)
    assert result.adapter_provenance["insertion_point"] == "sensory.output -> net.feedback"
    assert result.adapter_provenance["controller_input_mutated"] is False


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
        f"{GRAPH_ADAPTER_INPUT_PREFIX}:delayed_observation_offset__x_pos"
    ]
    assert payload.shape == (2, 10, 4)
    np.testing.assert_allclose(payload[:, :, 0], 0.01)
    assert result.adapter_provenance["insertion_point"] == "feedback.feedback -> sensory.input"
    assert "before sensory.input noise" in result.adapter_provenance["future_graphspec_mapping"]
    assert "pre_noise_delayed_measurement" in result.adapter_provenance[
        "future_graphspec_mapping"
    ]
    assert "compatibility alias" in result.adapter_provenance["future_graphspec_mapping"]


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


def test_extlqg_comparator_keeps_command_input_not_applicable() -> None:
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

    assert comparator["status"] == "not_applicable"
    assert "command-port intervention" in comparator["reason"]


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
