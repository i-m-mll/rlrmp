"""Tests for phase-aware linear recurrent output-feedback bridge rows."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

import rlrmp.analysis.output_feedback_linear_recurrent as recurrent
from rlrmp.analysis.bridge_certificates import (
    BELLMAN_HESSIAN_RESIDUAL,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    RECURRENCE_GRU_DIAGNOSTICS,
    STATE_WEIGHTED_ACTION_MISMATCH,
    VALUE_POLICY_GAP,
)
from rlrmp.analysis.bridge_controllers import LinearRecurrentController
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


def test_phase_time_features_are_polynomial_clock() -> None:
    features = recurrent.phase_time_features(5)

    assert features.shape == (5, 3)
    np.testing.assert_allclose(features[:, 0], np.ones(5))
    np.testing.assert_allclose(features[:, 1], np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(features[:, 2], np.linspace(0.0, 1.0, 5) ** 2)


def test_phase_aware_rollout_appends_phase_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        recurrent,
        "delayed_observation_matrix",
        lambda plant, config: np.array([[1.0, 0.0]]),
    )
    plant = _tiny_plant()
    controller = LinearRecurrentController(
        recurrent_weights=0.5 * np.eye(4),
        observation_weights=np.ones((4, 1)),
        previous_action_weights=np.zeros((4, 1)),
        phase_weights=np.eye(4, 3),
        readout_weights=np.ones((1, 4)),
        feedthrough_weights=np.zeros((1, 1)),
        readout_phase_weights=np.zeros((1, 3)),
    )

    batch = recurrent.rollout_phase_aware_linear_recurrent(
        controller=controller,
        plant=plant,
        x0=np.array([1.0, 0.0]),
        horizon=3,
        output_config=OutputFeedbackConfig(n_phys=1, delay_steps=1),
    )

    assert batch.observations is not None
    assert batch.hidden_states is not None
    assert batch.observations.shape == (1, 3, 4)
    np.testing.assert_allclose(batch.observations[0, :, 1:], recurrent.phase_time_features(3))
    assert batch.metadata["diagnostics"]["phase_time_input_used"] is True
    assert batch.metadata["diagnostics"]["phase_time_input_dim"] == 3


def test_recurrent_manifest_marks_formal_static_components_not_applicable(monkeypatch) -> None:
    monkeypatch.setattr(
        recurrent,
        "delayed_observation_matrix",
        lambda plant, config: np.array([[1.0, 0.0]]),
    )
    plant = _tiny_plant()
    controller = LinearRecurrentController(
        recurrent_weights=0.5 * np.eye(4),
        observation_weights=np.ones((4, 1)),
        previous_action_weights=np.zeros((4, 1)),
        phase_weights=np.eye(4, 3),
        readout_weights=np.zeros((1, 4)),
        feedthrough_weights=np.zeros((1, 1)),
        readout_phase_weights=np.zeros((1, 3)),
    )
    rollout = recurrent.rollout_phase_aware_linear_recurrent(
        controller=controller,
        plant=plant,
        x0=np.array([1.0, 0.0]),
        horizon=3,
        output_config=OutputFeedbackConfig(n_phys=1, delay_steps=1),
    )
    reference_clean = SimpleNamespace(u=np.zeros((3, 1)))
    condition = recurrent.LinearRecurrentCondition(
        label="no_coverage__scratch_seed_0",
        training_distribution="none",
        initialization="scratch_seed_0",
        objective="reward_rollout",
        hidden_dim=4,
        n_train_steps=1,
    )

    manifest = recurrent._manifest_for_condition(
        condition=condition,
        rollout=rollout,
        reference_clean=reference_clean,
        reference_clean_cost=1.0,
        candidate_cost=1.2,
        fit_metadata={"fit_method": "scratch_random_linear_readout"},
    )
    by_name = {component.name: component for component in manifest.certificate_components}

    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH].status == "available"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH].status == "not_applicable"
    assert by_name[VALUE_POLICY_GAP].status == "not_applicable"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL].status == "not_applicable"
    assert by_name[RECURRENCE_GRU_DIAGNOSTICS].status == "available"
    assert "aggregate_action_energy_mismatch" in manifest.metrics


def test_materialize_no_coverage_with_fake_reference(monkeypatch) -> None:
    plant = _tiny_plant()
    schedule = SimpleNamespace(
        Q=np.broadcast_to(np.eye(2), (3, 2, 2)),
        R=np.broadcast_to(np.eye(1), (3, 1, 1)),
        Q_f=np.eye(2),
    )
    K = np.array([[[0.2, 0.0]], [[0.1, 0.1]], [[0.0, 0.2]]])
    reference = SimpleNamespace(
        plant=plant,
        schedule=schedule,
        lqr_solution=SimpleNamespace(K=K),
    )
    reference_clean = SimpleNamespace(
        x=np.array([[1.0, 0.0], [1.0, -0.2], [0.98, -0.3], [0.95, -0.25]]),
        x_hat=np.array([[1.0, 0.0], [1.0, -0.2], [0.98, -0.3], [0.95, -0.25]]),
        u=np.array([[-0.2], [-0.08], [0.06]]),
    )
    monkeypatch.setattr(recurrent, "materialize_reference", lambda gamma_factors: reference)
    monkeypatch.setattr(
        recurrent,
        "make_cs_output_feedback_initial_state",
        lambda p, c: np.array([1.0, 0.0]),
    )
    monkeypatch.setattr(recurrent, "rollout_with_kalman_estimator", lambda p, k, x: reference_clean)
    monkeypatch.setattr(
        recurrent,
        "output_feedback_cost",
        lambda schedule, rollout: SimpleNamespace(total_without_disturbance_penalty=1.0),
    )
    monkeypatch.setattr(
        recurrent,
        "delayed_observation_matrix",
        lambda plant, config: np.array([[1.0, 0.0]]),
    )
    monkeypatch.setattr(
        recurrent,
        "kalman_estimator_gains",
        lambda plant, K, config: np.zeros((3, 2, 1)),
    )
    monkeypatch.setattr(recurrent, "process_covariance", lambda plant, config: np.zeros((2, 2)))

    summary, arrays = recurrent.materialize(include_coverage=False)

    assert summary["issue"] == recurrent.ISSUE_ID
    assert [row["spec"]["training_distribution"] for row in summary["rows"]] == [
        "clean_nominal",
        "riccati_epsilon",
    ]
    assert len(summary["failure_decomposition"]["rows"]) == 2
    assert any(key.endswith("__hidden_states") for key in arrays)
    component_counts = summary["diagnostics"]["component_status_counts"]
    assert component_counts[f"{CLOSED_LOOP_TRANSITION_MISMATCH}:not_applicable"] == 2
