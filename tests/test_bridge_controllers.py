"""Smoke tests for analytical bridge controller substrates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlrmp.analysis.bridge_contracts import BridgeRolloutBatch
from rlrmp.analysis.bridge_controllers import (
    LinearRecurrentController,
    PhaseModulatedLinearRecurrentController,
    TimeConstrainedGainParameterization,
    clamped_bspline_time_basis,
    hidden_growth_diagnostics,
    project_matrix_sequence_to_basis,
    recurrent_spectral_radius,
    rollout_linear_recurrent_controller,
    rollout_phase_modulated_linear_recurrent_controller,
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


def test_time_constrained_gain_cubic_bspline_is_partition_of_unity() -> None:
    parameterization = TimeConstrainedGainParameterization.cubic_bspline(
        horizon=24,
        n_basis=6,
        action_dim=2,
        input_dim=3,
    )

    assert parameterization.basis.shape == (24, 6)
    assert np.all(parameterization.basis >= 0.0)
    np.testing.assert_allclose(np.sum(parameterization.basis, axis=1), np.ones(24))


def test_time_constrained_gain_cubic_bspline_is_smooth_and_local() -> None:
    parameterization = TimeConstrainedGainParameterization.cubic_bspline(
        horizon=24,
        n_basis=6,
        action_dim=1,
        input_dim=1,
    )

    nonzero_counts = np.count_nonzero(parameterization.basis > 1e-14, axis=1)
    adjacent_deltas = np.abs(np.diff(parameterization.basis, axis=0))

    assert np.max(nonzero_counts) <= 4
    assert np.max(adjacent_deltas) < 0.2


def test_time_constrained_gain_cubic_bspline_projection_reconstructs_representable_gains() -> None:
    parameterization = TimeConstrainedGainParameterization.cubic_bspline(
        horizon=24,
        n_basis=6,
        action_dim=2,
        input_dim=2,
    )
    theta = np.linspace(-1.0, 1.0, 24, dtype=np.float64).reshape((6, 2, 2))
    gains = parameterization.gains_from_theta(theta)

    projection = parameterization.project_gains(gains)

    assert projection.rank == 6
    np.testing.assert_allclose(projection.reconstructed_gains, gains, atol=1e-12)
    assert projection.residual_norm < 1e-12
    assert projection.relative_residual < 1e-12


def test_time_constrained_gain_cubic_bspline_r60_reconstructs_unconstrained_gains() -> None:
    rng = np.random.default_rng(87)
    parameterization = TimeConstrainedGainParameterization.cubic_bspline(
        horizon=60,
        n_basis=60,
        action_dim=2,
        input_dim=2,
    )
    gains = rng.normal(size=(60, 2, 2))

    projection = parameterization.project_gains(gains)

    assert projection.rank == 60
    assert projection.singular_values[-1] > 0.0
    np.testing.assert_allclose(projection.reconstructed_gains, gains, atol=1e-11)
    assert projection.residual_norm < 1e-10


def test_clamped_bspline_basis_is_partition_of_unity_and_endpoint_clamped() -> None:
    basis = clamped_bspline_time_basis(horizon=11, n_basis=5)

    assert basis.shape == (11, 5)
    assert np.all(basis >= 0.0)
    np.testing.assert_allclose(np.sum(basis, axis=1), np.ones(11))
    np.testing.assert_allclose(basis[0], np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(basis[-1], np.array([0.0, 0.0, 0.0, 0.0, 1.0]))


def test_project_matrix_sequence_to_basis_reconstructs_representable_sequence() -> None:
    basis = clamped_bspline_time_basis(horizon=9, n_basis=3)
    theta = np.arange(12, dtype=np.float64).reshape((3, 2, 2))
    sequence = np.einsum("tb,bij->tij", basis, theta)

    projection = project_matrix_sequence_to_basis(sequence, basis)

    assert projection.rank == 3
    np.testing.assert_allclose(projection.reconstructed, sequence, atol=1e-12)
    assert projection.relative_residual < 1e-12


def test_phase_modulated_recurrent_controller_changes_matrices_over_time() -> None:
    basis = clamped_bspline_time_basis(horizon=3, n_basis=2, degree=1)
    controller = PhaseModulatedLinearRecurrentController(
        basis=basis,
        recurrent_coefficients=np.array([[[0.0]], [[1.0]]]),
        observation_coefficients=np.array([[[1.0]], [[2.0]]]),
        previous_action_coefficients=np.zeros((2, 1, 1)),
        hidden_bias_coefficients=np.zeros((2, 1)),
        readout_coefficients=np.array([[[1.0]], [[3.0]]]),
        feedthrough_coefficients=np.zeros((2, 1, 1)),
        action_bias_coefficients=np.zeros((2, 1)),
    )

    first = controller.matrices_at(0)
    last = controller.matrices_at(2)

    np.testing.assert_allclose(first["A_h"], np.array([[0.0]]))
    np.testing.assert_allclose(last["A_h"], np.array([[1.0]]))
    assert first["C_h"][0, 0] != last["C_h"][0, 0]


def test_phase_modulated_recurrent_rollout_uses_time_varying_readout() -> None:
    plant = TinyPlant(
        A=np.eye(1),
        B=np.array([[1.0]]),
        Bw=np.zeros((1, 1)),
    )
    basis = clamped_bspline_time_basis(horizon=2, n_basis=2, degree=1)
    controller = PhaseModulatedLinearRecurrentController(
        basis=basis,
        recurrent_coefficients=np.ones((2, 1, 1)),
        observation_coefficients=np.ones((2, 1, 1)),
        previous_action_coefficients=np.zeros((2, 1, 1)),
        hidden_bias_coefficients=np.zeros((2, 1)),
        readout_coefficients=np.array([[[1.0]], [[2.0]]]),
        feedthrough_coefficients=np.zeros((2, 1, 1)),
        action_bias_coefficients=np.zeros((2, 1)),
        initial_hidden=np.array([1.0]),
    )

    batch = rollout_phase_modulated_linear_recurrent_controller(
        controller,
        plant,
        np.array([0.0]),
        observation_matrix=np.array([[1.0]]),
    )

    np.testing.assert_allclose(batch.actions[0, :, 0], np.array([1.0, 2.0]))
    assert batch.metadata["diagnostics"]["phase_modulates_matrices"] is True


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


def test_linear_recurrent_controller_accepts_action_phase_and_bias_terms() -> None:
    controller = LinearRecurrentController(
        recurrent_weights=np.array([[0.5]]),
        observation_weights=np.array([[2.0]]),
        previous_action_weights=np.array([[3.0]]),
        phase_weights=np.array([[4.0, 5.0]]),
        hidden_bias=np.array([6.0]),
        readout_weights=np.array([[7.0]]),
        feedthrough_weights=np.array([[8.0]]),
        readout_phase_weights=np.array([[9.0, 10.0]]),
        action_bias=np.array([11.0]),
    )

    hidden = np.array([[1.0]])
    observation = np.array([[2.0]])
    previous_action = np.array([[3.0]])
    phase = np.array([[4.0, 5.0]])

    next_hidden = controller.next_hidden(hidden, observation, previous_action, phase)
    action = controller.action(hidden, observation, phase)

    np.testing.assert_allclose(next_hidden, np.array([[0.5 + 4.0 + 9.0 + 16.0 + 25.0 + 6.0]]))
    np.testing.assert_allclose(action, np.array([[7.0 + 16.0 + 36.0 + 50.0 + 11.0]]))
    diagnostics = controller.stability_diagnostics()
    assert diagnostics["phase_dim"] == 2
    assert diagnostics["previous_action_weight_norm"] == 3.0


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
