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
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL


def test_default_bank_is_json_serializable_with_required_channels() -> None:
    bank = default_cs_perturbation_bank()

    encoded = json.dumps(bank)
    decoded = json.loads(encoded)

    assert decoded["schema_version"] == SCHEMA_VERSION
    channels = {row["channel"] for row in decoded["perturbations"]}
    assert channels == {
        "initial_state",
        "plant_force",
        "sensory_feedback",
        "delayed_observation",
        "target_stream",
    }
    assert decoded["graphspec_alignment"]["named_channels"] == [
        "initial_state",
        "plant_force",
        "sensory_feedback",
        "delayed_observation",
        "target_stream",
    ]


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


def test_plant_force_pulse_adapter_sets_named_intervention_params() -> None:
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
        "channel": "plant_force",
        "family": "plant_force_pulse",
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
