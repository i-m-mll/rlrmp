"""Contract tests for the Phase 0 C&S analytical game card."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from rlrmp.analysis.math.cs_game_card import (
    DIAGNOSTIC_GAMMA_FACTOR,
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID,
    PRIMARY_GAMMA_FACTOR,
    assert_physical_selector_bw,
    build_canonical_game,
    build_zoh_sensitivity_game,
    materialize_reference,
    reference_summary,
    riccati_worst_case_policy,
)


@pytest.fixture(scope="module")
def reference():
    return materialize_reference(
        gamma_factors=(
            PRIMARY_GAMMA_FACTOR,
            OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
            DIAGNOSTIC_GAMMA_FACTOR,
            3.0,
        )
    )


def test_canonical_game_uses_physical_selector_bw():
    plant, schedule = build_canonical_game()

    assert plant.n == 48
    assert plant.m_w == 8
    assert plant.discretization == "euler"
    assert schedule.T == 60
    assert_physical_selector_bw(plant)

    assert jnp.allclose(plant.Bw[:8, :], jnp.eye(8, dtype=jnp.float64), atol=1e-14)
    assert jnp.allclose(plant.Bw[8:, :], 0.0, atol=1e-14)


def test_zoh_sensitivity_game_remains_selectable():
    canonical, canonical_schedule = build_canonical_game()
    sensitivity, sensitivity_schedule = build_zoh_sensitivity_game()

    assert canonical.discretization == "euler"
    assert sensitivity.discretization == "zoh"
    assert canonical_schedule.T == sensitivity_schedule.T == 60
    assert_physical_selector_bw(sensitivity)
    assert not jnp.allclose(canonical.A[:8, :8], sensitivity.A[:8, :8], atol=1e-14)
    assert jnp.allclose(canonical.Bw, sensitivity.Bw, atol=1e-14)


def test_gamma_frontier_marks_105_as_primary_target(reference):
    summary = reference_summary(reference)
    by_factor = {row["factor"]: row for row in summary["frontier"]}

    assert summary["rerun_metadata"]["discretization"] == "euler"
    assert summary["rerun_metadata"]["lane"] == "deterministic_analytical"
    assert summary["primary_gamma_factor"] == PRIMARY_GAMMA_FACTOR
    assert (
        summary["output_feedback_certificate_gamma_factor"]
        == OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
    )
    assert (
        summary["output_feedback_gamma_selection_issue"] == OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID
    )
    assert PRIMARY_GAMMA_FACTOR in by_factor
    assert OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR in by_factor
    assert DIAGNOSTIC_GAMMA_FACTOR in by_factor

    primary_delta = by_factor[PRIMARY_GAMMA_FACTOR]["delta_v_percent"]
    diagnostic_delta = by_factor[DIAGNOSTIC_GAMMA_FACTOR]["delta_v_percent"]
    far_delta = by_factor[3.0]["delta_v_percent"]

    assert 5.5 < primary_delta < 9.0
    assert diagnostic_delta < primary_delta
    assert far_delta < diagnostic_delta
    assert by_factor[PRIMARY_GAMMA_FACTOR]["gamma"] > summary["gamma_star"]


def test_worst_case_policy_is_feedback_policy_not_open_loop_sequence(reference):
    primary = next(ref for ref in reference.gamma_references if ref.factor == PRIMARY_GAMMA_FACTOR)
    F = riccati_worst_case_policy(reference.plant, primary.solution)

    assert F.shape == (reference.schedule.T, reference.plant.m_w, reference.plant.n)
    assert primary.epsilon_on_nominal.shape == (reference.schedule.T, reference.plant.m_w)
    assert primary.worst_case_rollout.epsilon.shape == (reference.schedule.T, reference.plant.m_w)

    # The open-loop sequences are realizations of F_t x_t on different
    # trajectories, so they need not be identical.
    assert not jnp.allclose(
        primary.epsilon_on_nominal,
        primary.worst_case_rollout.epsilon,
        atol=1e-10,
    )
    assert primary.worst_case_cost.disturbance_energy > 0.0


def test_worst_case_policy_satisfies_stationarity(reference):
    primary = next(ref for ref in reference.gamma_references if ref.factor == PRIMARY_GAMMA_FACTOR)
    plant = reference.plant
    t = 10
    x_t = primary.nominal_rollout.x[t]
    p_next = primary.solution.P[t + 1]
    a_cl = plant.A - plant.B @ primary.solution.K[t]
    eps_t = primary.worst_case_policy[t] @ x_t

    lhs = (
        primary.gamma * primary.gamma * jnp.eye(plant.m_w, dtype=jnp.float64)
        - plant.Bw.T @ p_next @ plant.Bw
    ) @ eps_t
    rhs = plant.Bw.T @ p_next @ a_cl @ x_t

    assert jnp.allclose(lhs, rhs, rtol=1e-8, atol=1e-6)
