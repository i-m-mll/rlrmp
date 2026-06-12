"""Contract tests for Phase 1 adversary-equivalence helpers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from rlrmp.analysis.math.adversary_equivalence import (
    OpenLoopOptimizationConfig,
    optimize_open_loop_epsilon,
    project_l2_ball,
    quadratic_rollout_cost,
    rollout_arrays_with_open_loop_epsilon,
)
from rlrmp.analysis.math.cs_game_card import (
    PRIMARY_GAMMA_FACTOR,
    materialize_reference,
    riccati_worst_case_policy,
    rollout_with_disturbance_policy,
)
from rlrmp.analysis.math.hinf_riccati import make_reach_initial_state


def test_project_l2_ball_clamps_rollout_energy():
    epsilon = jnp.ones((3, 2), dtype=jnp.float64)
    projected = project_l2_ball(epsilon, radius=0.5)

    assert float(jnp.linalg.norm(projected)) == pytest.approx(0.5, abs=1e-12)
    assert jnp.allclose(projected / projected[0, 0], epsilon / epsilon[0, 0])


def test_open_loop_rollout_matches_riccati_realized_epsilon():
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    primary = reference.gamma_references[0]
    x0 = make_reach_initial_state(
        reference.plant,
        init_pos=jnp.array([0.0, 0.0], dtype=jnp.float64),
        target_pos=jnp.array([0.15, 0.0], dtype=jnp.float64),
    )
    F = riccati_worst_case_policy(reference.plant, primary.solution)
    riccati = rollout_with_disturbance_policy(reference.plant, primary.solution.K, F, x0)

    x_open, u_open = rollout_arrays_with_open_loop_epsilon(
        reference.plant,
        primary.solution.K,
        x0,
        riccati.epsilon,
    )

    assert jnp.allclose(x_open, riccati.x, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(u_open, riccati.u, rtol=1e-10, atol=1e-10)


def test_zero_step_open_loop_optimizer_preserves_riccati_candidate():
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    primary = reference.gamma_references[0]
    x0 = make_reach_initial_state(
        reference.plant,
        init_pos=jnp.array([0.0, 0.0], dtype=jnp.float64),
        target_pos=jnp.array([0.15, 0.0], dtype=jnp.float64),
    )
    F = riccati_worst_case_policy(reference.plant, primary.solution)
    riccati = rollout_with_disturbance_policy(reference.plant, primary.solution.K, F, x0)
    budget = float(jnp.sum(riccati.epsilon**2))

    result = optimize_open_loop_epsilon(
        reference.plant,
        reference.schedule,
        primary.solution.K,
        x0,
        config=OpenLoopOptimizationConfig(
            budget=budget,
            n_steps=0,
            n_restarts=1,
            learning_rate=1e-2,
        ),
        gamma=primary.gamma,
        initial_candidates=(riccati.epsilon,),
    )
    riccati_cost = quadratic_rollout_cost(
        reference.schedule,
        riccati.x,
        riccati.u,
        riccati.epsilon,
        gamma=primary.gamma,
    )

    assert jnp.allclose(result.epsilon, riccati.epsilon, rtol=1e-10, atol=1e-10)
    assert result.cost.total_without_disturbance_penalty == pytest.approx(
        riccati_cost.total_without_disturbance_penalty,
        rel=1e-10,
    )


def test_open_loop_optimizer_keeps_best_seen_riccati_incumbent():
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    primary = reference.gamma_references[0]
    x0 = make_reach_initial_state(
        reference.plant,
        init_pos=jnp.array([0.0, 0.0], dtype=jnp.float64),
        target_pos=jnp.array([0.15, 0.0], dtype=jnp.float64),
    )
    F = riccati_worst_case_policy(reference.plant, primary.solution)
    riccati = rollout_with_disturbance_policy(reference.plant, primary.solution.K, F, x0)
    riccati_cost = quadratic_rollout_cost(
        reference.schedule,
        riccati.x,
        riccati.u,
        riccati.epsilon,
        gamma=primary.gamma,
    )

    result = optimize_open_loop_epsilon(
        reference.plant,
        reference.schedule,
        primary.solution.K,
        x0,
        config=OpenLoopOptimizationConfig(
            budget=riccati_cost.disturbance_energy,
            n_steps=5,
            n_restarts=1,
            learning_rate=3e-2,
        ),
        gamma=primary.gamma,
        initial_candidates=(riccati.epsilon,),
    )

    assert (
        result.cost.total_without_disturbance_penalty
        >= riccati_cost.total_without_disturbance_penalty - 1e-8
    )
