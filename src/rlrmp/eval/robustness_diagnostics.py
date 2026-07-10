"""Shared context and orchestration primitives for robustness diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    build_no_integrator_game,
)
from rlrmp.analysis.math.cs_released_simulation import (
    simulate_robust_released_forward,
    zero_forward_noise_draws,
    zero_noise_covariances,
)
from rlrmp.analysis.math.hinf_riccati import find_gamma_star, solve_hinf_riccati
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_output_feedback_gains,
)


def build_robust_output_feedback_6d_context(
    *,
    evaluation_from_rollout: Callable[..., Any],
    gamma_factor: float = OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
) -> dict[str, Any]:
    """Build the deterministic 6D no-integrator output-feedback H-infinity context."""

    plant, schedule = build_no_integrator_game()
    config = OutputFeedbackConfig(n_phys=6)
    gamma_star = find_gamma_star(plant, schedule)
    solution = solve_hinf_riccati(plant, schedule, gamma_factor * gamma_star)
    covariances = robust_estimator_covariances(
        plant,
        schedule,
        solution.gamma,
        config,
    )
    gains = robust_output_feedback_gains(
        plant,
        schedule,
        solution,
        covariances,
        config,
    )
    x0 = make_cs_output_feedback_initial_state(plant, config)
    base_rollout = simulate_robust_released_forward(
        plant,
        schedule,
        solution,
        x0,
        draws=zero_forward_noise_draws(T=schedule.T, plant=plant, config=config),
        covariances=zero_noise_covariances(plant, config),
        gains=gains,
        config=config,
    )
    if int(plant.n) != 36 or int(config.n_phys) != 6:
        raise ValueError(
            f"unexpected H-inf context dimensions: plant.n={plant.n}, "
            f"n_phys={config.n_phys}"
        )
    return {
        "plant": plant,
        "schedule": schedule,
        "config": config,
        "solution": solution,
        "gains": gains,
        "gamma_factor": gamma_factor,
        "gamma": float(solution.gamma),
        "gamma_star": float(gamma_star),
        "base_initial_state": np.asarray(x0, dtype=np.float64),
        "base_evaluation": evaluation_from_rollout(base_rollout, initial_state=x0),
        "contract": {
            "label": "6D output-feedback H-infinity",
            "state_dim": int(plant.n),
            "physical_dim": int(config.n_phys),
            "disturbance_dim": int(plant.m_w),
            "control_dim": int(plant.m_u),
            "delay_steps": int(config.delay_steps),
            "disturbance_integrators_exposed": False,
            "game_source": "rlrmp.analysis.math.cs_game_card.build_no_integrator_game",
            "config": "rlrmp.analysis.math.output_feedback.OutputFeedbackConfig(n_phys=6)",
            "gamma_factor": float(gamma_factor),
            "gamma_star": float(gamma_star),
            "gamma": float(solution.gamma),
            "admissible": bool(solution.admissible),
        },
    }
