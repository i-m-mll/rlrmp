"""Smoke tests for analytical bridge controller substrates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlrmp.analysis.bridge_contracts import BridgeRolloutBatch
from rlrmp.analysis.bridge_controllers import (
    LinearRecurrentController,
    TimeConstrainedGainParameterization,
    hidden_growth_diagnostics,
    recurrent_spectral_radius,
    rollout_linear_recurrent_controller,
)


@dataclass(frozen=True)
class TinyPlant:
    """Minimal plant fixture matching the bridge controller protocol."""

    A: np.ndarray
    B: np.ndarray
    Bw: np.ndarray


def test_time_constrained_gain_piecewise_projection_reconstructs_exactly() -> None:
    parameterization = TimeConstrainedGainParameterization.piecewise_constant(
        segment_ids=np.array([0, 0, 1, 1]),
        action_dim=2,
        input_dim=3,
    )
    theta = np.arange(12, dtype=np.float64).reshape((2, 2, 3))
    gains = parameterization.gains_from_theta(theta)

    projection = parameterization.project_gains(gains)

    assert gains.shape == (4, 2, 3)
    assert projection.theta.shape == (2, 2, 3)
    assert projection.rank == 2
    np.testing.assert_allclose(projection.theta, theta)
    np.testing.assert_allclose(projection.reconstructed_gains, gains)
    assert projection.residual_norm < 1e-12
    assert projection.relative_residual < 1e-12


def test_time_constrained_gain_constant_projection_uses_temporal_mean() -> None:
    parameterization = TimeConstrainedGainParameterization.constant(
        horizon=3,
        action_dim=1,
        input_dim=2,
    )
    gains = np.array(
        [
            [[1.0, 3.0]],
            [[2.0, 5.0]],
            [[6.0, 7.0]],
        ]
    )

    projection = parameterization.project_gains(gains)

    expected_mean = np.mean(gains, axis=0, keepdims=True)
    np.testing.assert_allclose(projection.theta, expected_mean)
    np.testing.assert_allclose(
        projection.reconstructed_gains,
        np.broadcast_to(expected_mean, gains.shape),
    )
    assert projection.residual_norm > 0.0


def test_linear_recurrent_controller_rollout_updates_hidden_and_actions() -> None:
    plant = TinyPlant(
        A=np.eye(2),
        B=np.array([[1.0], [0.0]]),
        Bw=np.zeros((2, 1)),
    )
    controller = LinearRecurrentController(
        recurrent_weights=np.array([[0.5]]),
        observation_weights=np.array([[1.0, 0.0]]),
        readout_weights=np.array([[2.0]]),
    )

    batch = rollout_linear_recurrent_controller(
        controller,
        plant,
        np.array([1.0, 0.0]),
        horizon=2,
    )

    assert isinstance(batch, BridgeRolloutBatch)
    assert batch.plant_states.shape == (1, 3, 2)
    assert batch.observations is not None
    assert batch.observations.shape == (1, 2, 2)
    assert batch.hidden_states is not None
    assert batch.hidden_states.shape == (1, 3, 1)
    np.testing.assert_allclose(batch.actions[0, :, 0], np.array([0.0, 2.0]))
    np.testing.assert_allclose(batch.hidden_states[0, :, 0], np.array([0.0, 1.0, 1.5]))
    np.testing.assert_allclose(batch.plant_states[0, :, 0], np.array([1.0, 1.0, 3.0]))


def test_linear_recurrent_rollout_accepts_batches_and_bridge_array_specs() -> None:
    plant = TinyPlant(
        A=np.eye(2),
        B=np.array([[1.0], [0.0]]),
        Bw=np.array([[0.0], [1.0]]),
    )
    controller = LinearRecurrentController(
        recurrent_weights=np.array([[0.25]]),
        observation_weights=np.array([[1.0]]),
        readout_weights=np.array([[1.0]]),
        feedthrough_weights=np.array([[0.5]]),
    )

    batch = rollout_linear_recurrent_controller(
        controller,
        plant,
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        horizon=3,
        observation_matrix=np.array([[1.0, 0.0]]),
        disturbances=np.zeros((3, 1)),
    )

    specs = {spec.name: spec for spec in batch.array_specs()}
    diagnostics = batch.metadata["diagnostics"]

    assert batch.batch_size == 2
    assert batch.horizon == 3
    assert specs["plant_states"].shape == (2, 4, 2)
    assert specs["actions"].shape == (2, 3, 1)
    assert specs["observations"].shape == (2, 3, 1)
    assert specs["hidden_states"].shape == (2, 4, 1)
    assert diagnostics["recurrent_spectral_radius"] == 0.25
    assert diagnostics["hidden_max_norm"] > 0.0


def test_recurrence_stability_and_hidden_growth_diagnostics() -> None:
    hidden_states = np.array(
        [
            [[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.5], [0.0, 0.25]],
        ]
    )

    assert recurrent_spectral_radius(np.diag([0.5, 1.2])) == 1.2
    diagnostics = hidden_growth_diagnostics(hidden_states)
    assert diagnostics["hidden_initial_mean_norm"] == 1.0
    assert diagnostics["hidden_final_mean_norm"] == 2.125
    assert diagnostics["hidden_max_norm"] == 4.0
    assert diagnostics["hidden_max_to_initial"] == 4.0
