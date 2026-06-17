"""Tests for the GRU worst-case epsilon audit helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.pipelines.gru_worst_case_epsilon_audit import (
    declared_epsilon_l2_radius,
    optimize_epsilon_sequence,
    project_l2_ball,
)


def _assert_numeric_dict_sequence_close(
    actual: tuple[dict[str, float | int], ...],
    expected: tuple[dict[str, float | int], ...],
) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row.keys() == expected_row.keys()
        for key, expected_value in expected_row.items():
            actual_value = actual_row[key]
            if isinstance(expected_value, int):
                assert actual_value == expected_value
            else:
                np.testing.assert_allclose(actual_value, expected_value, rtol=1e-12, atol=1e-12)


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


def test_staged_optimizer_matches_serial_with_multiple_restarts() -> None:
    weights = jnp.asarray([[1.0, 0.5], [0.25, 1.5]], dtype=jnp.float64)
    target = jnp.asarray([[0.35, -0.15], [0.1, 0.25]], dtype=jnp.float64)

    def objective(epsilon):
        return -jnp.sum(weights * jnp.square(epsilon - target)) + 0.05 * jnp.sum(epsilon)

    kwargs = {
        "shape": (2, 2),
        "radius": 0.9,
        "n_steps": 5,
        "n_restarts": 4,
        "step_size": 0.17,
        "seed": 21,
        "initial_candidates": (
            jnp.zeros((2, 2), dtype=jnp.float64),
            jnp.asarray([[0.8, 0.0], [0.0, 0.0]], dtype=jnp.float64),
        ),
    }

    serial = optimize_epsilon_sequence(objective, backend="serial", **kwargs)
    staged = optimize_epsilon_sequence(objective, backend="staged", **kwargs)

    assert staged.restart_index == serial.restart_index
    np.testing.assert_allclose(staged.epsilon, serial.epsilon, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(staged.objective, serial.objective, rtol=1e-12, atol=1e-12)
    _assert_numeric_dict_sequence_close(staged.history, serial.history)
    _assert_numeric_dict_sequence_close(staged.restart_summaries, serial.restart_summaries)


def test_staged_optimizer_matches_serial_zero_step_candidate_edge() -> None:
    candidates = (
        jnp.asarray([[0.2, 0.0], [0.0, 0.0]], dtype=jnp.float64),
        jnp.asarray([[0.0, 0.4], [0.0, 0.0]], dtype=jnp.float64),
    )

    def objective(epsilon):
        return epsilon[0, 1] - 0.5 * epsilon[0, 0]

    kwargs = {
        "shape": (2, 2),
        "radius": 1.0,
        "n_steps": 0,
        "n_restarts": 0,
        "step_size": 0.25,
        "seed": 3,
        "initial_candidates": candidates,
    }

    serial = optimize_epsilon_sequence(objective, backend="serial", **kwargs)
    staged = optimize_epsilon_sequence(objective, backend="staged", **kwargs)

    assert staged.restart_index == serial.restart_index == 1
    np.testing.assert_allclose(staged.epsilon, serial.epsilon, rtol=1e-12, atol=1e-12)
    assert staged.history == serial.history == (
        {"step": 0, "objective": 0.4, "epsilon_l2": 0.4},
    )
    assert len(staged.restart_summaries) == len(candidates)
    assert staged.restart_summaries == serial.restart_summaries
