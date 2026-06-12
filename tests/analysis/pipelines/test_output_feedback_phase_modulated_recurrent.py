"""Tests for d6d25d6 phase-modulated recurrent bridge rows."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

import rlrmp.analysis.pipelines.output_feedback_phase_modulated_recurrent as pm
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig


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
    assert {
        condition.rank
        for condition in conditions
        if condition.row_family.startswith("supervised")
        and condition.supervised_fit_scope == "readout_only"
    } == {12, 20}
    assert {
        condition.row_family
        for condition in conditions
        if condition.label.endswith("_supervised_nominal_action_fit")
    } == {pm.SUPERVISED_READOUT_ACTION_FIT_FAMILY}
    assert {
        condition.row_family
        for condition in conditions
        if condition.label.endswith("_supervised_process_io_map_fit")
    } == {pm.SUPERVISED_READOUT_IO_MAP_FIT_FAMILY}
    assert {
        condition.row_family
        for condition in conditions
        if condition.label.endswith("_supervised_action_io_combined_fit")
        and condition.supervised_fit_scope == "readout_only"
    } == {pm.SUPERVISED_READOUT_ACTION_IO_MAP_FIT_FAMILY}
    assert {
        condition.rank
        for condition in conditions
        if condition.row_family.startswith("supervised")
        and condition.supervised_fit_scope == "full_matrix"
    } == {20, 30}
    assert {
        condition.rank
        for condition in pm.default_conditions(
            include_reward=False,
            include_supervised_extensions=True,
        )
        if condition.row_family.startswith("supervised")
        and condition.supervised_fit_scope == "readout_only"
    } == {12, 20, 30, 60}


def test_reward_conditions_include_r60_capacity_controls() -> None:
    conditions = pm.default_conditions(include_reward=True, include_supervised_extensions=True)
    by_label = {condition.label: condition for condition in conditions}

    expected = {
        "pm_linrec_r60_projected_oracle_nominal_then_reward",
        "pm_linrec_r60_projected_oracle_process_measurement_then_reward",
        "pm_linrec_r60_supervised_action_io_nominal_then_reward",
        "pm_linrec_r60_supervised_action_io_process_measurement_then_reward",
        "pm_linrec_r60_clean_scratch_reward",
        "pm_linrec_r60_process_measurement_scratch_reward",
    }

    assert expected <= set(by_label)
    for label in expected:
        condition = by_label[label]
        assert condition.rank == 60
        assert condition.n_train_steps >= pm.R60_SCRATCH_REWARD_TRAIN_STEPS
        assert condition.learning_rate == pm.R60_REWARD_LEARNING_RATE
        assert condition.gradient_clip_norm == pm.R60_REWARD_GRADIENT_CLIP_NORM
    assert (
        by_label["pm_linrec_r60_projected_oracle_nominal_then_reward"].proximal_preservation_weight
        == pm.R60_REWARD_PROXIMAL_WEIGHT
    )
    assert (
        by_label["pm_linrec_r60_supervised_action_io_process_measurement_then_reward"].row_family
        == "supervised_action_io_warm_start_then_reward_lens"
    )
    assert (
        by_label["pm_linrec_r60_process_measurement_scratch_reward"].proximal_preservation_weight
        == 0.0
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
    criteria = row["metrics"]["reward_verdict_criteria"]
    assert criteria["source"] == "external_response_map_certificate"
    assert criteria["mean_timewise_action_mismatch_role"] == "diagnostic_only"
    assert criteria["aggregate_action_energy_threshold"] == pm.REWARD_ACTION_ENERGY_PASS_THRESHOLD
    assert criteria["relevant_response_map_threshold"] == pm.REWARD_RESPONSE_MAP_PASS_THRESHOLD
    assert criteria["disturbance_to_cost_threshold"] == pm.REWARD_DISTURBANCE_COST_PASS_THRESHOLD


def test_supervised_warm_start_reward_logs_preservation_metadata(monkeypatch) -> None:
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
        label="pm_linrec_r3_supervised_action_io_nominal_then_reward",
        row_family="supervised_action_io_warm_start_then_reward_lens",
        rank=3,
        training_distribution="nominal_reward_supervised_action_io_preserve",
        evaluation_lens="nominal_clean",
        n_train_steps=1,
        learning_rate=5e-4,
        gradient_clip_norm=0.5,
        proximal_preservation_weight=1e-4,
        supervised_objective="action_and_io",
    )

    summary, _arrays = pm.materialize(include_reward=False, conditions=(condition,))

    row = summary["rows"][0]
    optimizer = row["metrics"]["optimizer"]
    assert row["spec"]["objective"] == "reward_rollout"
    assert row["spec"]["parameters"]["reward_control_mode"] == "preserve_supervised_action_io"
    assert row["spec"]["parameters"]["warm_start_source"] == "supervised_action_io_map_fit"
    assert row["spec"]["parameters"]["gradient_clip_norm"] == 0.5
    assert row["spec"]["parameters"]["proximal_preservation_weight"] == 1e-4
    assert optimizer["reward_control_mode"] == "preserve_supervised_action_io"
    assert optimizer["warm_start_supervised_objective"] == "action_and_io"
    assert optimizer["adam_gradient_clip_norm"] == 0.5
    assert optimizer["proximal_preservation_weight"] == 1e-4
    assert "reward_final_proximal_preservation_penalty" in optimizer


def test_supervised_condition_fits_maps_without_reward_label(monkeypatch) -> None:
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
        label="pm_linrec_r3_supervised_action_io_combined_fit",
        row_family=pm.SUPERVISED_READOUT_ACTION_IO_MAP_FIT_FAMILY,
        rank=3,
        training_distribution="mixed_action_process_measurement_io_supervised",
        evaluation_lens="mixed_process_measurement_io",
        disturbance_scale=0.02,
        measurement_scale=0.02,
        n_train_steps=2,
        learning_rate=1e-3,
        supervised_objective="action_and_io",
    )

    summary, _arrays = pm.materialize(include_reward=False, conditions=(condition,))

    row = summary["rows"][0]
    component_names = {component["name"] for component in row["certificate_components"]}
    assert row["spec"]["objective"] == "supervised_action_and_io"
    assert row["spec"]["evaluation_lane"] == "supervised_representation"
    assert row["metrics"]["row_family"] == pm.SUPERVISED_READOUT_ACTION_IO_MAP_FIT_FAMILY
    assert row["metrics"]["optimizer"]["is_supervised_trained"] is True
    assert row["metrics"]["optimizer"]["is_reward_trained"] is False
    assert (
        row["metrics"]["optimizer"]["fit_method"]
        == "alternating_least_squares_readout_feedthrough_maps"
    )
    assert row["metrics"]["optimizer"]["supervised_fit_scope"] == "readout_feedthrough_only"
    assert row["metrics"]["optimizer"]["supervised_dynamics_fit"] is False
    assert row["metrics"]["optimizer"]["supervised_fit_blocks"] == ["C_h", "D_y", "c"]
    assert row["metrics"]["optimizer"]["supervised_frozen_blocks"] == [
        "A_h",
        "B_y",
        "B_u",
        "b_h",
    ]
    assert (
        row["metrics"]["optimizer"]["supervised_best_loss"]
        <= (row["metrics"]["optimizer"]["supervised_initial_loss"])
    )
    assert "measurement_history_to_action_map_mismatch" in component_names
    assert "measurement_history_to_output_map_mismatch" in component_names
    assert "disturbance_history_to_output_map_mismatch" in component_names
    assert "disturbance_history_to_cost_quadratic" in component_names


def test_full_matrix_supervised_condition_uses_all_blocks(monkeypatch) -> None:
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

    def fake_adam(params, loss_fn, *, n_steps, learning_rate, max_param_abs=None):
        loss = float(loss_fn({key: value.copy() for key, value in params.items()}))
        fitted = {key: value.copy() for key, value in params.items()}
        fitted["A_h"] = fitted["A_h"] * 0.99
        return fitted, {
            "initial_loss": loss,
            "final_loss": loss,
            "last_loss": loss,
            "best_loss": loss,
        }

    monkeypatch.setattr(pm, "_adam_minimize", fake_adam)
    condition = pm.PhaseModulatedCondition(
        label="pm_linrec_r3_full_matrix_supervised_action_io_combined_fit",
        row_family="supervised_action_io_map_fit",
        rank=3,
        training_distribution="mixed_action_process_measurement_io_supervised",
        evaluation_lens="mixed_process_measurement_io",
        disturbance_scale=0.02,
        measurement_scale=0.02,
        n_train_steps=2,
        learning_rate=1e-3,
        stability_penalty=1e-3,
        smoothness_penalty=1e-5,
        proximal_penalty=1e-4,
        parameter_bound=10.0,
        supervised_objective="action_and_io",
        supervised_fit_scope="full_matrix",
    )

    summary, _arrays = pm.materialize(include_reward=False, conditions=(condition,))

    row = summary["rows"][0]
    optimizer = row["metrics"]["optimizer"]
    assert row["spec"]["optimizer_label"] == "adam_full_matrix_supervised_action_io_maps"
    assert optimizer["supervised_fit_scope"] == "full_matrix"
    assert optimizer["supervised_fit_blocks"] == ["A_h", "B_y", "B_u", "b_h", "C_h", "D_y", "c"]
    assert optimizer["smoothness_penalty"] == 1e-5
    assert optimizer["proximal_penalty"] == 1e-4
    assert optimizer["parameter_bound"] == 10.0
    assert row["spec"]["parameters"]["supervised_fit_scope"] == "full_matrix"
    assert summary["diagnostics"]["audit"]["full_matrix_supervised_rows"] == [row["spec"]["run_id"]]


def test_default_materialize_gates_reward_when_supervised_rows_do_not_pass(monkeypatch) -> None:
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

    def one_supervised_condition(ranks: tuple[int, ...]) -> tuple[pm.PhaseModulatedCondition, ...]:
        rank = 3 if ranks == pm.SUPERVISED_INITIAL_RANKS else 2
        return (
            pm.PhaseModulatedCondition(
                label=f"pm_linrec_r{rank}_supervised_nominal_action_fit",
                row_family=pm.SUPERVISED_READOUT_ACTION_FIT_FAMILY,
                rank=rank,
                training_distribution="nominal_action_supervised",
                evaluation_lens="nominal_clean",
                n_train_steps=0,
                supervised_objective="action",
            ),
        )

    monkeypatch.setattr(pm, "_exact_oracle_conditions", lambda: ())
    monkeypatch.setattr(pm, "_projection_diagnostic_conditions", lambda: ())
    monkeypatch.setattr(pm, "_supervised_conditions", one_supervised_condition)
    monkeypatch.setattr(pm, "_full_matrix_supervised_conditions", lambda: ())
    monkeypatch.setattr(
        pm,
        "_reward_conditions",
        lambda ranks=(3,): (
            pm.PhaseModulatedCondition(
                label="pm_linrec_r3_clean_scratch_reward",
                row_family="reward_lens",
                rank=3,
                training_distribution="nominal",
                evaluation_lens="nominal_clean",
                n_train_steps=0,
            ),
        ),
    )
    monkeypatch.setattr(pm, "_is_supervised_representation_pass", lambda row: False)

    summary, _arrays = pm.materialize(include_reward=True)

    families = [row["metrics"]["row_family"] for row in summary["rows"]]
    assert families == [
        pm.SUPERVISED_READOUT_ACTION_FIT_FAMILY,
        pm.SUPERVISED_READOUT_ACTION_FIT_FAMILY,
    ]
    assert summary["diagnostics"]["audit"]["supervised_extension_materialized"] is True
    assert (
        summary["diagnostics"]["audit"]["reward_gating_status"]
        == "stopped_no_supervised_action_io_representation_pass"
    )


def test_reward_verdict_uses_external_certificate_not_mean_action_mismatch() -> None:
    condition = pm.PhaseModulatedCondition(
        label="pm_linrec_r12_clean_scratch_reward",
        row_family="reward_lens",
        rank=12,
        training_distribution="nominal",
        evaluation_lens="nominal_clean",
    )
    projection = {"combined_relative_residual": 1.0}
    response = {
        "observation_to_action": 0.01,
        "disturbance_to_cost": 0.01,
    }
    action_summary = {
        "mismatch_ratio_mean": 100.0,
        "aggregate_mismatch_ratio": 0.01,
    }
    criteria = pm._reward_verdict_criteria(
        condition,
        action_summary=action_summary,
        response_metrics=response,
    )

    assert criteria["passes"] is True
    assert criteria["mean_timewise_action_mismatch_role"] == "diagnostic_only"
    assert (
        pm._row_verdict(
            condition,
            projection,
            action_summary,
            metrics={
                "response_map_mismatch": response,
                "reward_verdict_criteria": criteria,
            },
        )
        == "reward_trained_external_certificate_equivalent"
    )


def test_reward_verdict_rejects_large_action_energy_despite_small_mean_mismatch() -> None:
    condition = pm.PhaseModulatedCondition(
        label="pm_linrec_r12_clean_scratch_reward",
        row_family="reward_lens",
        rank=12,
        training_distribution="nominal",
        evaluation_lens="nominal_clean",
    )
    response = {
        "observation_to_action": 0.01,
        "disturbance_to_cost": 0.01,
    }
    action_summary = {
        "mismatch_ratio_mean": 0.01,
        "aggregate_mismatch_ratio": pm.REWARD_ACTION_ENERGY_PASS_THRESHOLD + 0.01,
    }
    criteria = pm._reward_verdict_criteria(
        condition,
        action_summary=action_summary,
        response_metrics=response,
    )

    assert criteria["action_energy_pass"] is False
    assert (
        pm._row_verdict(
            condition,
            {"combined_relative_residual": 0.0},
            action_summary,
            metrics={
                "response_map_mismatch": response,
                "reward_verdict_criteria": criteria,
            },
        )
        == "reward_trained_external_certificate_non_equivalent"
    )


def test_reward_verdict_uses_lens_relevant_response_map_and_cost_sidecar() -> None:
    condition = pm.PhaseModulatedCondition(
        label="pm_linrec_r12_mixed_process_observer_reward",
        row_family="reward_lens",
        rank=12,
        training_distribution="mixed",
        evaluation_lens="mixed_process_observer",
        disturbance_scale=0.02,
    )
    action_summary = {
        "mismatch_ratio_mean": 0.01,
        "aggregate_mismatch_ratio": 0.01,
    }
    response = {
        "observation_to_action": 0.0,
        "disturbance_to_action": 0.0,
        "disturbance_to_state": 0.0,
        "disturbance_to_output": 0.0,
        "measurement_to_action": pm.REWARD_RESPONSE_MAP_PASS_THRESHOLD + 0.01,
        "measurement_to_output": 0.0,
        "disturbance_to_cost": 0.0,
    }
    criteria = pm._reward_verdict_criteria(
        condition,
        action_summary=action_summary,
        response_metrics=response,
    )

    assert criteria["relevant_response_map_keys"] == [
        "observation_to_action",
        "disturbance_to_action",
        "disturbance_to_state",
        "disturbance_to_output",
        "measurement_to_action",
        "measurement_to_output",
    ]
    assert criteria["response_map_pass"] is False
    assert (
        pm._row_verdict(
            condition,
            {"combined_relative_residual": 0.0},
            action_summary,
            metrics={
                "response_map_mismatch": response,
                "reward_verdict_criteria": criteria,
            },
        )
        == "reward_trained_external_certificate_non_equivalent"
    )
