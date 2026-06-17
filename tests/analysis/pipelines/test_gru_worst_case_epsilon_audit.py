"""Tests for the GRU worst-case epsilon audit helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.pipelines.gru_worst_case_epsilon_audit import (
    declared_epsilon_l2_radius,
    optimize_epsilon_sequence,
    project_l2_ball,
)


def test_project_l2_ball_preserves_inside_and_projects_outside() -> None:
    inside = jnp.asarray([3.0, 4.0])
    np.testing.assert_allclose(project_l2_ball(inside, 5.0), inside)

    outside = project_l2_ball(jnp.asarray([6.0, 8.0]), 5.0)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(outside)), 5.0)
    np.testing.assert_allclose(outside, jnp.asarray([3.0, 4.0]))

    zero = project_l2_ball(jnp.zeros((2, 3)), 1.0)
    np.testing.assert_allclose(zero, 0.0)


def test_declared_epsilon_radius_applies_budget_and_reach_scaling() -> None:
    run_spec = {
        "hps": {
            "broad_epsilon_training": {
                "budget_scale": 2.0,
                "reach_length_scaling": True,
                "budget_contract": {
                    "effective_l2_radius_15cm": 0.1,
                    "reference_reach_m": 0.15,
                },
            }
        }
    }

    radius = declared_epsilon_l2_radius(run_spec, reach_length_m=0.30)

    assert radius == 0.4


def test_declared_epsilon_radius_accepts_level_override_without_run_broad_config() -> None:
    radius = declared_epsilon_l2_radius(
        {"hps": {}},
        reach_length_m=0.30,
        budget_level_override="strong",
    )

    assert radius > 0.0
    np.testing.assert_allclose(radius, 0.0023284905801002004 * 2.0)


def test_declared_epsilon_radius_uses_active_pgd_schedule_when_broad_disabled() -> None:
    run_spec = {
        "hps": {
            "broad_epsilon_training": {
                "enabled": False,
                "budget_contract": {"effective_l2_radius_15cm": 0.1},
            },
            "broad_epsilon_pgd_training": {
                "enabled": True,
                "budget_scale": 2.0,
                "reach_length_scaling": False,
                "budget_contract": {
                    "active_max_l2_radius_15cm": 0.2,
                    "effective_l2_radius_15cm": 0.1,
                },
                "budget_schedule": {
                    "mode": "sisu_energy_fraction",
                    "max_l2_radius_15cm": 0.3,
                },
            },
        }
    }

    radius = declared_epsilon_l2_radius(run_spec, reach_length_m=0.30)

    np.testing.assert_allclose(radius, 0.6)


def test_declared_epsilon_radius_scales_level_override() -> None:
    radius = declared_epsilon_l2_radius(
        {"hps": {}},
        reach_length_m=0.30,
        budget_level_override="strong",
        budget_scale_override=3.0,
    )

    np.testing.assert_allclose(radius, 0.0023284905801002004 * 3.0 * 2.0)


def test_optimize_epsilon_sequence_improves_quadratic_objective() -> None:
    target = jnp.asarray([[0.3, 0.4], [0.0, 0.0]], dtype=jnp.float64)

    def objective(epsilon):
        return -jnp.sum(jnp.square(epsilon - target))

    result = optimize_epsilon_sequence(
        objective,
        shape=(2, 2),
        radius=1.0,
        n_steps=8,
        n_restarts=1,
        step_size=0.2,
        seed=0,
        initial_candidates=(jnp.zeros((2, 2), dtype=jnp.float64),),
    )

    assert result.objective > result.initial_objective
    assert result.l2_norm <= 1.0 + 1e-12
    np.testing.assert_allclose(result.epsilon, np.asarray(target), atol=0.21)
