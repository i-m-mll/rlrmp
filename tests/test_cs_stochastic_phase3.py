"""Tests for the Phase 3 released-code stochastic evaluation lane."""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    PRIMARY_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.cs_stochastic_phase3 import (
    Phase3ControllerSpec,
    Phase3StochasticConfig,
    process_noise_sweep_summary,
    result_summary,
    run_phase3_process_noise_sweep,
    run_phase3_stochastic_evaluation,
)


def _small_controller_specs() -> tuple[Phase3ControllerSpec, ...]:
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    K_ref = reference.lqr_solution.K
    return (
        Phase3ControllerSpec(
            label="analytical_lqr_reference",
            source="analytical_lqr_reference",
            K=K_ref,
            deterministic_gain_relative_error=0.0,
            deterministic_objective_ratio_to_reference=1.0,
        ),
        Phase3ControllerSpec(
            label="scratch_smoke",
            source="deterministic_phase3_scratch_fit",
            K=jnp.zeros_like(K_ref),
            deterministic_gain_relative_error=1.0,
            deterministic_objective_ratio_to_reference=None,
        ),
        Phase3ControllerSpec(
            label="preservation_smoke",
            source="deterministic_phase3_preservation_init_fit",
            K=K_ref,
            deterministic_gain_relative_error=0.0,
            deterministic_objective_ratio_to_reference=1.0,
        ),
    )


def test_phase3_stochastic_evaluation_is_reproducible() -> None:
    config = Phase3StochasticConfig(n_trials=3, seed=17)
    first = run_phase3_stochastic_evaluation(
        config=config,
        controllers=_small_controller_specs(),
    )
    second = run_phase3_stochastic_evaluation(
        config=config,
        controllers=_small_controller_specs(),
    )
    first_summary = result_summary(first)
    second_summary = result_summary(second)

    assert first_summary["evaluations"] == second_summary["evaluations"]
    assert first_summary["evaluations"][0]["action_mismatch_to_reference_mean"] == 0.0
    assert first_summary["evaluations"][1]["action_mismatch_to_reference_mean"] > 0.0


def test_phase3_stochastic_manifest_marks_lane_and_no_bellman_parity_claim() -> None:
    result = run_phase3_stochastic_evaluation(
        config=Phase3StochasticConfig(n_trials=2, seed=3),
        controllers=_small_controller_specs(),
    )
    summary = result_summary(result)

    assert summary["rerun_metadata"]["discretization"] == "euler"
    assert summary["rerun_metadata"]["lane"] == "released_stochastic"
    assert (
        summary["output_feedback_certificate_gamma_factor"]
        == OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
    )
    assert summary["claims"]["bellman_stochastic_parity"] is False
    assert "stochastic Bellman objective" in summary["claims"]["note"]
    assert "Bellman parity claim" in summary["non_goals"]


def test_phase3_stochastic_result_reports_required_metrics() -> None:
    result = run_phase3_stochastic_evaluation(
        config=Phase3StochasticConfig(n_trials=2, seed=5),
        controllers=_small_controller_specs(),
    )
    row = result_summary(result)["evaluations"][0]

    for key in (
        "cost_mean",
        "cost_std",
        "cost_ratio_to_reference_mean",
        "peak_forward_velocity_mean",
        "terminal_error_mean",
        "action_mismatch_to_reference_mean",
        "deterministic_exact_l2_cost_ratio_to_lqr",
        "deterministic_lambda_over_gamma_squared",
        "deterministic_gamma_penalized_feasible",
    ):
        assert key in row

    assert row["deterministic_exact_l2_cost_ratio_to_lqr"] > 0.0
    assert row["deterministic_lambda_over_gamma_squared"] > 0.0


def test_phase3_stochastic_result_retains_certificate_trajectory_arrays() -> None:
    result = run_phase3_stochastic_evaluation(
        config=Phase3StochasticConfig(n_trials=2, seed=6),
        controllers=_small_controller_specs(),
    )
    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
    horizon = reference.schedule.T
    state_dim = reference.plant.n
    action_dim = reference.plant.m_u

    for label in ("analytical_lqr_reference", "scratch_smoke", "preservation_smoke"):
        key = label.replace("/", "_").replace(" ", "_").replace(".", "p").replace("-", "_")
        assert result.arrays[f"{key}_x"].shape == (2, horizon + 1, state_dim)
        assert result.arrays[f"{key}_x_hat"].shape == (2, horizon + 1, state_dim)
        assert result.arrays[f"{key}_u_command"].shape == (2, horizon, action_dim)
        assert result.arrays[f"{key}_u_applied"].shape == (2, horizon, action_dim)


def test_phase3_process_noise_sweep_propagates_explicit_scale_cells() -> None:
    result = run_phase3_process_noise_sweep(
        config=Phase3StochasticConfig(
            n_trials=1,
            seed=7,
            process_covariance_scale=None,
        ),
        process_covariance_scales=(0.0, 0.3),
        controllers=_small_controller_specs(),
    )

    assert result.process_covariance_scales == (0.0, 0.3)
    assert [cell.label for cell in result.cells] == ["0.0", "0.3"]
    assert [cell.process_covariance_scale for cell in result.cells] == [0.0, 0.3]
    assert [cell.result.config.process_covariance_scale for cell in result.cells] == [0.0, 0.3]
    assert result.base_config.process_covariance_scale is None


def test_phase3_process_noise_sweep_summary_reports_cell_scales_and_shape() -> None:
    result = run_phase3_process_noise_sweep(
        config=Phase3StochasticConfig(n_trials=1, seed=11),
        process_covariance_scales=(0.0, 1.0, 3.0),
        controllers=_small_controller_specs(),
    )
    summary = process_noise_sweep_summary(result)

    assert summary["process_covariance_scales"] == [0.0, 1.0, 3.0]
    assert summary["base_monte_carlo"]["process_covariance_scale"] is None
    assert [cell["label"] for cell in summary["cells"]] == ["0.0", "1.0", "3.0"]
    assert [cell["process_covariance_scale"] for cell in summary["cells"]] == [0.0, 1.0, 3.0]
    assert [cell["monte_carlo"]["process_covariance_scale"] for cell in summary["cells"]] == [
        0.0,
        1.0,
        3.0,
    ]
    assert all(
        len(cell["evaluations"]) == len(_small_controller_specs()) for cell in summary["cells"]
    )
