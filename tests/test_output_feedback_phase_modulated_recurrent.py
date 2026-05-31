"""Tests for d6d25d6 phase-modulated recurrent bridge rows."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

import rlrmp.analysis.output_feedback_phase_modulated_recurrent as pm
from rlrmp.analysis.output_feedback import OutputFeedbackConfig


@dataclass(frozen=True)
class TinyPlant:
    """Minimal delayed-observation plant fixture."""

    A: np.ndarray
    B: np.ndarray
    Bw: np.ndarray
    n: int
    m_w: int


def _tiny_plant() -> TinyPlant:
    return TinyPlant(
        A=np.array([[1.0, 0.1], [0.0, 0.9]]),
        B=np.array([[0.0], [1.0]]),
        Bw=np.zeros((2, 1)),
        n=2,
        m_w=1,
    )


def test_oracle_recurrent_reference_realizes_time_varying_kalman_form(monkeypatch) -> None:
    plant = _tiny_plant()
    gains = np.array([[[0.2, 0.0]], [[0.1, 0.1]], [[0.0, 0.2]]])
    kalman = np.array([[[0.1], [0.2]], [[0.2], [0.3]], [[0.3], [0.4]]])
    monkeypatch.setattr(
        pm, "delayed_observation_matrix", lambda plant, config: np.array([[1.0, 0.0]])
    )
    monkeypatch.setattr(pm, "kalman_estimator_gains", lambda plant, K, config: kalman)

    reference = pm.build_oracle_recurrent_reference(
        plant=plant,
        gains=gains,
        output_config=OutputFeedbackConfig(n_phys=1, delay_steps=1),
        initial_hidden=np.array([1.0, 0.0]),
    )

    assert reference.recurrent_matrices.shape == (3, 2, 2)
    assert reference.observation_matrices.shape == (3, 2, 1)
    np.testing.assert_allclose(reference.readout_matrices, -gains)
    expected_a0 = plant.A - plant.B @ gains[0] - kalman[0] @ np.array([[1.0, 0.0]])
    np.testing.assert_allclose(reference.recurrent_matrices[0], expected_a0)


def test_project_oracle_reference_returns_phase_modulated_controller(monkeypatch) -> None:
    plant = _tiny_plant()
    gains = np.array([[[0.2, 0.0]], [[0.1, 0.1]], [[0.0, 0.2]]])
    monkeypatch.setattr(
        pm, "delayed_observation_matrix", lambda plant, config: np.array([[1.0, 0.0]])
    )
    monkeypatch.setattr(pm, "kalman_estimator_gains", lambda plant, K, config: np.zeros((3, 2, 1)))
    reference = pm.build_oracle_recurrent_reference(
        plant=plant,
        gains=gains,
        output_config=OutputFeedbackConfig(n_phys=1, delay_steps=1),
        initial_hidden=np.array([1.0, 0.0]),
    )
    basis = pm.phase_basis(horizon=3, rank=3)

    controller, projections = pm.project_oracle_reference(reference, basis=basis)

    assert controller.horizon == 3
    assert controller.hidden_dim == 2
    assert set(projections) == {"A_h", "B_y", "B_u", "b_h", "C_h", "D_y", "c"}
    np.testing.assert_allclose(
        controller.matrix_sequence(controller.readout_coefficients),
        reference.readout_matrices,
        atol=1e-10,
    )


def test_materialize_with_fake_reference_reports_available_io_map(monkeypatch) -> None:
    plant = _tiny_plant()
    schedule = SimpleNamespace(
        Q=np.broadcast_to(np.eye(2), (3, 2, 2)),
        R=np.broadcast_to(np.eye(1), (3, 1, 1)),
        Q_f=np.eye(2),
    )
    gains = np.array([[[0.2, 0.0]], [[0.1, 0.1]], [[0.0, 0.2]]])
    reference = SimpleNamespace(
        plant=plant,
        schedule=schedule,
        lqr_solution=SimpleNamespace(K=gains),
    )
    reference_clean = SimpleNamespace(
        x=np.array([[1.0, 0.0], [1.0, -0.2], [0.98, -0.3], [0.95, -0.25]]),
        x_hat=np.array([[1.0, 0.0], [1.0, -0.2], [0.98, -0.3], [0.95, -0.25]]),
        u=np.array([[-0.2], [-0.08], [0.06]]),
    )
    monkeypatch.setattr(pm, "materialize_reference", lambda gamma_factors: reference)
    monkeypatch.setattr(
        pm, "make_cs_output_feedback_initial_state", lambda p, c: np.array([1.0, 0.0])
    )
    monkeypatch.setattr(pm, "rollout_with_kalman_estimator", lambda p, k, x: reference_clean)
    monkeypatch.setattr(
        pm,
        "output_feedback_cost",
        lambda schedule, rollout: SimpleNamespace(total_without_disturbance_penalty=1.0),
    )
    monkeypatch.setattr(
        pm, "delayed_observation_matrix", lambda plant, config: np.array([[1.0, 0.0]])
    )
    monkeypatch.setattr(pm, "kalman_estimator_gains", lambda plant, K, config: np.zeros((3, 2, 1)))
    monkeypatch.setattr(pm, "process_covariance", lambda plant, config: np.zeros((2, 2)))
    condition = pm.PhaseModulatedCondition(
        label="pm_linrec_r3_projected_oracle_nominal_replay",
        row_family="projected_oracle_replay",
        rank=3,
        training_distribution="nominal",
        evaluation_lens="nominal_clean",
    )

    summary, arrays = pm.materialize(include_reward=False, conditions=(condition,))

    assert summary["issue"] == pm.ISSUE_ID
    assert (
        summary["rows"][0]["metrics"]["io_map_certificate"]["status"]
        == "standard_components_available"
    )
    assert summary["rows"][0]["metrics"]["verdict"] == "projected_oracle_replay_diagnostic"
    assert "clamped_bspline_r3_basis" in arrays
    assert any(key.endswith("__hidden_states") for key in arrays)


def test_default_conditions_relabel_state_coverage_and_projection_rows() -> None:
    conditions = pm.default_conditions(include_reward=False)
    labels_and_lenses = [
        (condition.label, condition.evaluation_lens, condition.row_family)
        for condition in conditions
    ]

    assert not any("imitation" in label for label, _lens, _family in labels_and_lenses)
    assert not any("exact_process_eigen" in lens for _label, lens, _family in labels_and_lenses)
    assert {
        condition.rank
        for condition in conditions
        if condition.row_family == "projected_oracle_replay"
    } == {12, 20, 30, 60}
    assert any(
        condition.row_family == "exact_oracle_sanity"
        and condition.evaluation_lens == "process_measurement_io"
        for condition in conditions
    )
    assert any(
        condition.row_family == "projected_oracle_state_coverage_eval"
        and condition.evaluation_lens == "state_coverage_eigen_m4_s0.3"
        for condition in conditions
    )


def test_exact_oracle_process_measurement_row_is_response_map_sanity(monkeypatch) -> None:
    plant = _tiny_plant()
    plant = TinyPlant(
        A=plant.A,
        B=plant.B,
        Bw=np.array([[0.0], [1.0]]),
        n=plant.n,
        m_w=plant.m_w,
    )
    schedule = SimpleNamespace(
        Q=np.broadcast_to(np.eye(2), (3, 2, 2)),
        R=np.broadcast_to(np.eye(1), (3, 1, 1)),
        Q_f=np.eye(2),
    )
    gains = np.array([[[0.2, 0.0]], [[0.1, 0.1]], [[0.0, 0.2]]])
    reference = SimpleNamespace(
        plant=plant,
        schedule=schedule,
        lqr_solution=SimpleNamespace(K=gains),
    )
    reference_clean = SimpleNamespace(
        x=np.array([[1.0, 0.0], [1.0, -0.2], [0.98, -0.3], [0.95, -0.25]]),
        x_hat=np.array([[1.0, 0.0], [1.0, -0.2], [0.98, -0.3], [0.95, -0.25]]),
        u=np.array([[-0.2], [-0.08], [0.06]]),
    )
    monkeypatch.setattr(pm, "materialize_reference", lambda gamma_factors: reference)
    monkeypatch.setattr(
        pm, "make_cs_output_feedback_initial_state", lambda p, c: np.array([1.0, 0.0])
    )
    monkeypatch.setattr(pm, "rollout_with_kalman_estimator", lambda p, k, x: reference_clean)
    monkeypatch.setattr(
        pm,
        "output_feedback_cost",
        lambda schedule, rollout: SimpleNamespace(total_without_disturbance_penalty=1.0),
    )
    monkeypatch.setattr(
        pm, "delayed_observation_matrix", lambda plant, config: np.array([[1.0, 0.0]])
    )
    monkeypatch.setattr(pm, "kalman_estimator_gains", lambda plant, K, config: np.zeros((3, 2, 1)))
    monkeypatch.setattr(pm, "process_covariance", lambda plant, config: np.zeros((2, 2)))
    condition = pm.PhaseModulatedCondition(
        label="pm_linrec_exact_oracle_process_measurement_io",
        row_family="exact_oracle_sanity",
        rank=3,
        training_distribution="process_measurement_io_probe",
        evaluation_lens="process_measurement_io",
        disturbance_scale=0.02,
        measurement_scale=0.02,
    )

    summary, _arrays = pm.materialize(include_reward=False, conditions=(condition,))

    row = summary["rows"][0]
    assert row["metrics"]["verdict"] == "exact_oracle_sanity_pass"
    assert row["metrics"]["response_map_mismatch"]["max_aggregate_mismatch"] == 0.0
    assert row["metrics"]["aggregate_action_energy_mismatch"] == 0.0


def test_reward_condition_runs_bounded_training(monkeypatch) -> None:
    plant = _tiny_plant()
    schedule = SimpleNamespace(
        Q=np.broadcast_to(np.eye(2), (3, 2, 2)),
        R=np.broadcast_to(np.eye(1), (3, 1, 1)),
        Q_f=np.eye(2),
    )
    gains = np.array([[[0.2, 0.0]], [[0.1, 0.1]], [[0.0, 0.2]]])
    reference = SimpleNamespace(
        plant=plant,
        schedule=schedule,
        lqr_solution=SimpleNamespace(K=gains),
    )
    reference_clean = SimpleNamespace(
        x=np.array([[1.0, 0.0], [1.0, -0.2], [0.98, -0.3], [0.95, -0.25]]),
        x_hat=np.array([[1.0, 0.0], [1.0, -0.2], [0.98, -0.3], [0.95, -0.25]]),
        u=np.array([[-0.2], [-0.08], [0.06]]),
    )
    monkeypatch.setattr(pm, "materialize_reference", lambda gamma_factors: reference)
    monkeypatch.setattr(
        pm, "make_cs_output_feedback_initial_state", lambda p, c: np.array([1.0, 0.0])
    )
    monkeypatch.setattr(pm, "rollout_with_kalman_estimator", lambda p, k, x: reference_clean)
    monkeypatch.setattr(
        pm,
        "output_feedback_cost",
        lambda schedule, rollout: SimpleNamespace(total_without_disturbance_penalty=1.0),
    )
    monkeypatch.setattr(
        pm, "delayed_observation_matrix", lambda plant, config: np.array([[1.0, 0.0]])
    )
    monkeypatch.setattr(pm, "kalman_estimator_gains", lambda plant, K, config: np.zeros((3, 2, 1)))
    monkeypatch.setattr(pm, "process_covariance", lambda plant, config: np.zeros((2, 2)))
    condition = pm.PhaseModulatedCondition(
        label="pm_linrec_r3_tiny_scratch_reward",
        row_family="reward_lens",
        rank=3,
        training_distribution="nominal",
        evaluation_lens="nominal_clean",
        n_train_steps=2,
        learning_rate=1e-3,
    )

    summary, _arrays = pm.materialize(include_reward=False, conditions=(condition,))

    row = summary["rows"][0]
    assert row["spec"]["objective"] == "reward_rollout"
    assert row["spec"]["optimizer_label"] == "adam_phase_modulated_reward_rollout"
    assert row["metrics"]["optimizer"]["is_reward_trained"] is True
    assert (
        row["metrics"]["optimizer"]["reward_best_loss"]
        <= row["metrics"]["optimizer"]["reward_initial_loss"]
    )
    assert row["metrics"]["verdict"].startswith("reward_trained_")
