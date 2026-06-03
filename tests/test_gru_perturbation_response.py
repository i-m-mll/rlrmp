"""Tests for the C&S GRU perturbation-response bank."""

from __future__ import annotations

import json

import numpy as np
from feedbax.intervene import FixedFieldParams
from feedbax.state import CartesianState
from feedbax.task import TaskTrialSpec

from rlrmp.analysis.gru_perturbation_bank import (
    SCHEMA_VERSION,
    apply_perturbation_to_trial_specs,
    default_cs_perturbation_bank,
    delta_full_qrf_cost_summary,
    extlqg_comparator_status,
    score_full_qrf_rollout_cost,
)
from rlrmp.analysis.cs_game_card import build_canonical_game
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL


def test_default_bank_is_json_serializable_with_required_channels() -> None:
    bank = default_cs_perturbation_bank()

    encoded = json.dumps(bank)
    decoded = json.loads(encoded)

    assert decoded["schema_version"] == SCHEMA_VERSION
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


def test_command_input_pulse_adapter_sets_named_intervention_params() -> None:
    trial_specs = TaskTrialSpec(
        inits={"mechanics.vector": np.zeros((2, 8), dtype=np.float64)},
        targets={},
        inputs={"effector_target": CartesianState(pos=np.zeros((2, 10, 2)))},
        intervene={
            PLANT_INTERVENOR_LABEL: FixedFieldParams(
                scale=np.zeros((2,), dtype=np.float32),
                field=np.zeros((2,), dtype=np.float32),
                active=False,
            )
        },
    )
    perturbation = {
        "channel": "command_input",
        "family": "command_input_pulse",
        "amplitude": 2.0,
        "axis": "y",
        "sign": -1,
        "timing": {"start_time_index": 3, "duration_steps": 2},
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "evaluated"
    params = result.trial_specs.intervene[PLANT_INTERVENOR_LABEL]
    np.testing.assert_allclose(params.field.value[:, 3:5, 1], -2.0)
    np.testing.assert_allclose(params.field.value[:, :3, :], 0.0)
    np.testing.assert_allclose(params.scale, 1.0)
    assert params.active.value[:, 3:5].all()
    assert result.adapter_provenance["external_load_force"] is False
    assert "efferent.output -> mechanics.force" in (
        result.adapter_provenance["future_graphspec_insertion_point"]
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


def test_sensory_adapter_is_explicitly_not_implemented() -> None:
    trial_specs = TaskTrialSpec(inits={}, targets={}, inputs={})
    perturbation = {
        "channel": "sensory_feedback",
        "family": "sensory_feedback_offset",
        "amplitude": 0.01,
        "axis": "x",
        "sign": 1,
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "not_implemented"
    assert "controller-input hacks" in result.reason


def test_delayed_observation_reason_names_clean_pre_noise_channel() -> None:
    trial_specs = TaskTrialSpec(inits={}, targets={}, inputs={})
    perturbation = {
        "channel": "delayed_observation",
        "family": "delayed_observation_offset",
        "amplitude": 0.01,
        "axis": "x",
        "sign": 1,
    }

    result = apply_perturbation_to_trial_specs(trial_specs, perturbation)

    assert result.status == "not_implemented"
    assert "DelayedPositionVelocityFeedback" in result.reason
    assert "before sensory noise" in result.reason


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
