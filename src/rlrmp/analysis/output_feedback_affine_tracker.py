"""Affine tracker output-feedback bridge for issue 50c260d.

The controller family is

``u_t = u_ref,t - K_t (xhat_t - xhat_ref,t)``.

When ``u_ref,t = -K_ref,t xhat_ref,t`` and ``K_t = K_ref,t`` this is exactly
the analytical LQR/Kalman output-feedback controller.  The bridge rows below
separate feedforward replay from feedback correction while keeping the game,
estimator, and certificate lane unchanged.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.optimize as scipy_opt
from jaxtyping import Array, Float

from rlrmp.analysis.bridge_certificates import build_standard_certificate_components
from rlrmp.analysis.bridge_contracts import BridgeRunManifest, BridgeRunSpec, make_bridge_run_id
from rlrmp.analysis.bridge_controllers import TimeConstrainedGainParameterization
from rlrmp.analysis.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.hinf_riccati import PlantLinearization
from rlrmp.analysis.linear_round_trip import LinearTrainingConfig, rollout_task_cost
from rlrmp.analysis.output_feedback import (
    OutputFeedbackConfig,
    OutputFeedbackRollout,
    delayed_observation_matrix,
    exact_output_feedback_adversary_audit,
    kalman_estimator_gains,
    kalman_estimator_joint_matrices,
    make_cs_output_feedback_initial_state,
    output_feedback_cost,
    process_covariance,
    robust_estimator_covariances,
    robust_estimator_fixed_adversary_policy,
    robust_output_feedback_gains,
    rollout_with_kalman_estimator,
    rollout_with_robust_estimator_policy,
)
from rlrmp.analysis.output_feedback import (
    _gamma_penalized_quadratic_diagnostic,
    _maximize_quadratic_on_l2_ball,
    _rollout_summary_fields,
)
from rlrmp.analysis.output_feedback_rollout_recovery import (
    EigenspectrumCoverageConfig,
    ObserverErrorCoverageConfig,
    _eigenspectrum_coverage_samples,
    _empty_coverage_samples,
    _observer_error_coverage_samples,
    _state_scales,
    _training_ensemble,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


jax.config.update("jax_enable_x64", True)

ISSUE_ID = "50c260d"
UMBRELLA_ID = "43e8728"
TIME_CONSTRAINED_ISSUE_ID = "87edaae"
STANDARD_CERTIFICATE_ISSUE_ID = "d01c35a"
FAILURE_DECOMPOSITION_ISSUE_ID = "c45adde"
DEFAULT_SPLINE_RANK = 20

AffineRowKind = Literal[
    "reference_replay",
    "feedforward_only",
    "gain_only",
    "both_from_scratch",
    "spline_tracker",
    "staged_reward",
    "action_match_diagnostic",
]
PerturbationFamily = Literal["clean", "riccati_eps", "state_eig", "observer_error", "mixed"]
InitializationSource = Literal["scratch", "reference", "previous_stage"]


@dataclass(frozen=True)
class AffineTrackerCondition:
    """One affine tracker bridge row."""

    label: str
    row_kind: AffineRowKind
    train_feedforward: bool
    train_gain: bool
    perturbation_family: PerturbationFamily = "clean"
    objective_family: Literal["reward_rollout", "supervised_action_match"] = "reward_rollout"
    initialization_source: InitializationSource = "scratch"
    stage_source_label: str | None = None
    clean_objective_weight: float = 1.0
    perturbation_objective_weight: float = 0.0
    gain_basis_rank: int | None = None
    optimizer: str = "lbfgsb"
    maxiter: int = 200
    learning_rate: float = 3e-3
    adam_clip_norm: float | None = 1e4
    eigenspectrum_coverage: EigenspectrumCoverageConfig | None = None
    observer_error_coverage: ObserverErrorCoverageConfig | None = None

    @property
    def training_distribution(self) -> str:
        """Shared bridge manifest training-distribution label."""

        if self.eigenspectrum_coverage is not None:
            return f"eigenspectrum_{self.eigenspectrum_coverage.objective}"
        if self.observer_error_coverage is not None:
            return f"observer_error_{self.observer_error_coverage.objective}"
        if self.perturbation_family != "clean":
            return self.perturbation_family
        return "nominal" if self.row_kind != "reference_replay" else "none"

    @property
    def is_diagnostic(self) -> bool:
        """Whether this row is diagnostic rather than a scratch bridge claim."""

        return self.objective_family == "supervised_action_match"


@dataclass(frozen=True)
class AffineTrackerFit:
    """One materialized affine tracker row."""

    condition: AffineTrackerCondition
    K: Float[Array, "T m_u n"]
    u_ref: Float[Array, "T m_u"]
    xhat_ref: Float[Array, "T_plus_1 n"]
    objective_initial: float
    objective_final: float
    objective_reference: float
    objective_zero: float
    objective_ratio_to_reference: float
    feedforward_relative_error: float
    gain_relative_error: float
    gradient_norm_initial: float | None
    gradient_norm_final: float | None
    projected_gradient_norm_final: float | None
    optimizer_status: str
    optimizer_success: bool
    n_iterations: int
    n_function_evaluations: int
    clean_rollout: OutputFeedbackRollout
    clean_cost: float
    clean_action_mismatch_ratio: float
    under_epsilon_rollout: OutputFeedbackRollout
    under_epsilon_cost: float
    under_epsilon_cost_ratio_to_lqr: float
    under_epsilon_action_mismatch_ratio: float
    exact_l2_cost: float
    exact_l2_cost_ratio_to_lqr: float
    exact_l2_cost_ratio_to_hinf: float
    gamma_penalized_feasible: bool
    gamma_penalized_lambda_over_gamma_squared: float

    @property
    def label(self) -> str:
        """Stable row label."""

        return self.condition.label


@dataclass(frozen=True)
class AffineTrackerResult:
    """Complete affine tracker bridge materialization."""

    issue_id: str
    fits: tuple[AffineTrackerFit, ...]
    diagnostics: dict[str, Any]
    arrays: dict[str, np.ndarray]


def baseline_conditions(
    *,
    maxiter: int = 200,
    spline_rank: int = DEFAULT_SPLINE_RANK,
) -> tuple[AffineTrackerCondition, ...]:
    """Return required no-coverage affine rows in issue order."""

    return (
        AffineTrackerCondition(
            label="reference_affine_replay",
            row_kind="reference_replay",
            train_feedforward=False,
            train_gain=False,
            maxiter=0,
        ),
        AffineTrackerCondition(
            label="feedforward_only_k_ref_frozen",
            row_kind="feedforward_only",
            train_feedforward=True,
            train_gain=False,
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="gain_only_u_ref_frozen",
            row_kind="gain_only",
            train_feedforward=False,
            train_gain=True,
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="both_from_scratch",
            row_kind="both_from_scratch",
            train_feedforward=True,
            train_gain=True,
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label=f"spline_tracker_r{spline_rank}",
            row_kind="spline_tracker",
            train_feedforward=True,
            train_gain=True,
            gain_basis_rank=spline_rank,
            maxiter=maxiter,
        ),
    )


def selected_coverage_conditions(
    *,
    maxiter: int = 200,
    spline_rank: int = DEFAULT_SPLINE_RANK,
    base_labels: tuple[str, ...] = ("both_from_scratch", f"spline_tracker_r{DEFAULT_SPLINE_RANK}"),
) -> tuple[AffineTrackerCondition, ...]:
    """Return selected state-coverage rows from the 50c260d acceptance criteria."""

    coverage_specs: tuple[
        tuple[str, EigenspectrumCoverageConfig | ObserverErrorCoverageConfig], ...
    ]
    coverage_specs = (
        (
            "state_eigenspectrum_m4_s1_w0p1",
            EigenspectrumCoverageConfig(n_modes=4, scale=1.0, weight=0.1, objective="state"),
        ),
        (
            "state_eigenspectrum_m4_s3_w0p1",
            EigenspectrumCoverageConfig(n_modes=4, scale=3.0, weight=0.1, objective="state"),
        ),
        (
            "observer_error_state_m1_s0p3_w0p1",
            ObserverErrorCoverageConfig(n_modes=1, scale=0.3, weight=0.1, objective="state"),
        ),
    )
    base_by_label = {
        condition.label: condition for condition in baseline_conditions(maxiter=maxiter)
    }
    if (
        f"spline_tracker_r{DEFAULT_SPLINE_RANK}" not in base_by_label
        and spline_rank != DEFAULT_SPLINE_RANK
    ):
        base_by_label[f"spline_tracker_r{spline_rank}"] = baseline_conditions(
            maxiter=maxiter,
            spline_rank=spline_rank,
        )[-1]

    rows: list[AffineTrackerCondition] = []
    for base_label in base_labels:
        base = base_by_label[base_label]
        for suffix, coverage in coverage_specs:
            kwargs: dict[str, Any] = {"label": f"{base.label}__{suffix}", "maxiter": maxiter}
            if isinstance(coverage, EigenspectrumCoverageConfig):
                kwargs["eigenspectrum_coverage"] = coverage
            else:
                kwargs["observer_error_coverage"] = coverage
            rows.append(replace(base, **kwargs))
    return tuple(rows)


def staged_curriculum_conditions(*, maxiter: int = 80) -> tuple[AffineTrackerCondition, ...]:
    """Return the 50c260d staged nominal-only affine curriculum rows.

    The main rows use reward/task rollout objectives.  The two action-match rows
    are labeled diagnostic and are kept separate from from-scratch bridge claims.
    """

    return (
        AffineTrackerCondition(
            label="affine_clean_scratch_baseline",
            row_kind="staged_reward",
            train_feedforward=True,
            train_gain=True,
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_ff_clean_stage",
            row_kind="staged_reward",
            train_feedforward=True,
            train_gain=False,
            initialization_source="reference",
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_fb_riccati_eps",
            row_kind="staged_reward",
            train_feedforward=False,
            train_gain=True,
            perturbation_family="riccati_eps",
            initialization_source="previous_stage",
            stage_source_label="affine_ff_clean_stage",
            clean_objective_weight=0.0,
            perturbation_objective_weight=1.0,
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_joint_riccati_eps",
            row_kind="staged_reward",
            train_feedforward=True,
            train_gain=True,
            perturbation_family="riccati_eps",
            initialization_source="previous_stage",
            stage_source_label="affine_fb_riccati_eps",
            clean_objective_weight=1.0,
            perturbation_objective_weight=1.0,
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_fb_state_eig",
            row_kind="staged_reward",
            train_feedforward=False,
            train_gain=True,
            perturbation_family="state_eig",
            initialization_source="previous_stage",
            stage_source_label="affine_ff_clean_stage",
            clean_objective_weight=0.0,
            perturbation_objective_weight=1.0,
            eigenspectrum_coverage=EigenspectrumCoverageConfig(
                n_modes=4,
                scale=1.0,
                weight=1.0,
                objective="state",
            ),
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_joint_state_eig",
            row_kind="staged_reward",
            train_feedforward=True,
            train_gain=True,
            perturbation_family="state_eig",
            initialization_source="previous_stage",
            stage_source_label="affine_fb_state_eig",
            clean_objective_weight=1.0,
            perturbation_objective_weight=1.0,
            eigenspectrum_coverage=EigenspectrumCoverageConfig(
                n_modes=4,
                scale=1.0,
                weight=1.0,
                objective="state",
            ),
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_fb_observer_error",
            row_kind="staged_reward",
            train_feedforward=False,
            train_gain=True,
            perturbation_family="observer_error",
            initialization_source="previous_stage",
            stage_source_label="affine_ff_clean_stage",
            clean_objective_weight=0.0,
            perturbation_objective_weight=1.0,
            observer_error_coverage=ObserverErrorCoverageConfig(
                n_modes=1,
                scale=0.3,
                weight=1.0,
                objective="state",
            ),
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_joint_observer_error",
            row_kind="staged_reward",
            train_feedforward=True,
            train_gain=True,
            perturbation_family="observer_error",
            initialization_source="previous_stage",
            stage_source_label="affine_fb_observer_error",
            clean_objective_weight=1.0,
            perturbation_objective_weight=1.0,
            observer_error_coverage=ObserverErrorCoverageConfig(
                n_modes=1,
                scale=0.3,
                weight=1.0,
                objective="state",
            ),
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_fb_mixed",
            row_kind="staged_reward",
            train_feedforward=False,
            train_gain=True,
            perturbation_family="mixed",
            initialization_source="previous_stage",
            stage_source_label="affine_ff_clean_stage",
            clean_objective_weight=0.0,
            perturbation_objective_weight=1.0,
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_joint_mixed",
            row_kind="staged_reward",
            train_feedforward=True,
            train_gain=True,
            perturbation_family="mixed",
            initialization_source="previous_stage",
            stage_source_label="affine_fb_mixed",
            clean_objective_weight=1.0,
            perturbation_objective_weight=1.0,
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_feedback_action_match_riccati_eps",
            row_kind="action_match_diagnostic",
            train_feedforward=False,
            train_gain=True,
            perturbation_family="riccati_eps",
            objective_family="supervised_action_match",
            initialization_source="previous_stage",
            stage_source_label="affine_ff_clean_stage",
            clean_objective_weight=0.0,
            perturbation_objective_weight=1.0,
            maxiter=maxiter,
        ),
        AffineTrackerCondition(
            label="affine_feedback_action_match_mixed",
            row_kind="action_match_diagnostic",
            train_feedforward=False,
            train_gain=True,
            perturbation_family="mixed",
            objective_family="supervised_action_match",
            initialization_source="previous_stage",
            stage_source_label="affine_ff_clean_stage",
            clean_objective_weight=0.0,
            perturbation_objective_weight=1.0,
            maxiter=maxiter,
        ),
    )


def rollout_with_affine_tracker(
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    u_ref: Float[Array, "T m_u"],
    xhat_ref: Float[Array, "T_plus_1 n"],
    x0: Float[Array, " n"],
    epsilon: Float[Array, "T m_w"] | None = None,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> OutputFeedbackRollout:
    """Roll the affine tracker through the same Kalman estimator lane."""

    T = K.shape[0]
    eps = jnp.zeros((T, plant.m_w), dtype=jnp.float64) if epsilon is None else epsilon
    x, x_hat, y, u, covs = affine_tracker_rollout_arrays(
        plant=plant,
        K=K,
        u_ref=u_ref,
        xhat_ref=xhat_ref,
        x0=x0,
        epsilon=eps,
        config=config,
    )
    peak, peak_idx, terminal, effort = _rollout_summary_fields(plant, x, u)
    return OutputFeedbackRollout(
        x=x,
        x_hat=x_hat,
        y=y,
        u=u,
        epsilon=eps,
        estimator_covariances=covs,
        peak_forward_velocity=peak,
        peak_forward_velocity_idx=peak_idx,
        terminal_position_error=terminal,
        control_effort=effort,
    )


def affine_tracker_rollout_arrays(
    *,
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    u_ref: Float[Array, "T m_u"],
    xhat_ref: Float[Array, "T_plus_1 n"],
    x0: Float[Array, " n"],
    epsilon: Float[Array, "T m_w"],
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> tuple[
    Float[Array, "T_plus_1 n"],
    Float[Array, "T_plus_1 n"],
    Float[Array, "T obs"],
    Float[Array, "T m_u"],
    Float[Array, "T_plus_1 n n"],
]:
    """Return differentiable affine tracker rollout arrays."""

    H = delayed_observation_matrix(plant, config)
    gains = kalman_estimator_gains(plant, K, config)
    Q_proc = process_covariance(plant, config)
    Sigma0 = jnp.eye(plant.n, dtype=jnp.float64) * jnp.asarray(
        config.estimator_initial_covariance,
        dtype=jnp.float64,
    )

    def step(carry, inputs):
        x_t, xhat_t, Sigma_t = carry
        eps_t, K_t, u_ref_t, xhat_ref_t, G_t = inputs
        y_t = H @ x_t
        u_t = u_ref_t - K_t @ (xhat_t - xhat_ref_t)
        xhat_next = plant.A @ xhat_t + plant.B @ u_t + G_t @ (y_t - H @ xhat_t)
        x_next = plant.A @ x_t + plant.B @ u_t + plant.Bw @ eps_t
        Sigma_next = (plant.A - G_t @ H) @ Sigma_t @ plant.A.T + Q_proc
        Sigma_next = 0.5 * (Sigma_next + Sigma_next.T)
        return (x_next, xhat_next, Sigma_next), (x_next, xhat_next, y_t, u_t, Sigma_next)

    (_, _, _), (x_tail, xhat_tail, y, u, cov_tail) = jax.lax.scan(
        step,
        (x0.astype(jnp.float64), x0.astype(jnp.float64), Sigma0),
        (
            epsilon.astype(jnp.float64),
            K.astype(jnp.float64),
            u_ref.astype(jnp.float64),
            xhat_ref[:-1].astype(jnp.float64),
            gains,
        ),
    )
    x = jnp.concatenate([x0[None].astype(jnp.float64), x_tail], axis=0)
    x_hat = jnp.concatenate([x0[None].astype(jnp.float64), xhat_tail], axis=0)
    covs = jnp.concatenate([Sigma0[None], cov_tail], axis=0)
    return x, x_hat, y, u, covs


def run_affine_tracker_bridge(
    *,
    conditions: tuple[AffineTrackerCondition, ...] | None = None,
    include_selected_coverage: bool = True,
    maxiter: int = 80,
    training_config: LinearTrainingConfig = LinearTrainingConfig(n_steps=500),
    output_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> AffineTrackerResult:
    """Run affine tracker rows and keep bulk arrays for certificates."""

    if conditions is None:
        conditions = staged_curriculum_conditions(maxiter=maxiter)
        if include_selected_coverage:
            conditions = conditions

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    plant = reference.plant
    schedule = reference.schedule
    K_ref = reference.lqr_solution.K
    x0 = make_cs_output_feedback_initial_state(plant, output_config)
    states, weights = _training_ensemble(plant, training_config, output_config)
    state_scales = _state_scales(states, weights)
    lqr_clean = rollout_with_kalman_estimator(plant, K_ref, x0, config=output_config)
    u_ref = lqr_clean.u
    xhat_ref = lqr_clean.x_hat
    covs = robust_estimator_covariances(plant, schedule, gamma_ref.gamma, output_config)
    robust_gains = robust_output_feedback_gains(
        plant,
        schedule,
        gamma_ref.solution,
        covs,
        output_config,
    )
    robust_policy = robust_estimator_fixed_adversary_policy(
        plant,
        schedule,
        gamma_ref.solution,
        robust_gains,
        covs,
        output_config,
    )
    robust_rollout = rollout_with_robust_estimator_policy(
        plant,
        schedule,
        gamma_ref.solution,
        x0,
        robust_policy,
        gains=robust_gains,
        config=output_config,
    )
    riccati_epsilon = robust_rollout.epsilon
    budget = float(jnp.sum(riccati_epsilon**2))
    budget_l2 = float(np.sqrt(max(budget, 0.0)))
    lqr_under_eps = rollout_with_kalman_estimator(
        plant,
        K_ref,
        x0,
        riccati_epsilon,
        config=output_config,
    )
    lqr_exact = affine_tracker_exact_adversary_audit(
        label="reference_affine_replay",
        plant=plant,
        schedule=schedule,
        K=K_ref,
        u_ref=u_ref,
        xhat_ref=xhat_ref,
        x0=x0,
        budget=budget,
        penalty_gamma=gamma_ref.gamma,
        config=output_config,
    )
    hinf_exact = exact_output_feedback_adversary_audit(
        label="analytical_hinf_output_feedback",
        plant=plant,
        schedule=schedule,
        controller_gains=robust_gains,
        x0=x0,
        budget=budget,
        estimator_kind="robust",
        solution=gamma_ref.solution,
        penalty_gamma=gamma_ref.gamma,
        config=output_config,
    )

    arrays: dict[str, np.ndarray] = {
        "lqr_reference_K": np.asarray(K_ref),
        "lqr_reference_u_ref": np.asarray(u_ref),
        "lqr_reference_xhat_ref": np.asarray(xhat_ref),
        "lqr_clean_x": np.asarray(lqr_clean.x),
        "lqr_clean_x_hat": np.asarray(lqr_clean.x_hat),
        "lqr_clean_u": np.asarray(lqr_clean.u),
        "lqr_under_eps_x": np.asarray(lqr_under_eps.x),
        "lqr_under_eps_x_hat": np.asarray(lqr_under_eps.x_hat),
        "lqr_under_eps_u": np.asarray(lqr_under_eps.u),
        "riccati_epsilon": np.asarray(riccati_epsilon),
    }
    fits = []
    fits_by_label: dict[str, AffineTrackerFit] = {}
    for condition in conditions:
        coverage = _coverage_samples(
            condition=condition,
            plant=plant,
            schedule=schedule,
            K_ref=K_ref,
            x0=x0,
            budget_l2=budget_l2,
            gamma=gamma_ref.gamma,
            output_config=output_config,
        )
        initial_K, initial_u_ref = _initial_parts_for_condition(
            condition,
            K_ref=K_ref,
            u_ref=u_ref,
            fits_by_label=fits_by_label,
        )
        arrays.update({f"{condition.label}_{key}": value for key, value in coverage.arrays.items()})
        fit = _fit_condition(
            condition=condition,
            plant=plant,
            schedule=schedule,
            K_ref=K_ref,
            u_ref_ref=u_ref,
            xhat_ref=xhat_ref,
            states=states,
            weights=weights,
            state_scales=state_scales,
            coverage=coverage,
            initial_K=initial_K,
            initial_u_ref=initial_u_ref,
            x0=x0,
            riccati_epsilon=riccati_epsilon,
            gamma=gamma_ref.gamma,
            lqr_clean_rollout=lqr_clean,
            lqr_under_epsilon_rollout=lqr_under_eps,
            lqr_exact_cost=float(lqr_exact["cost"].total_without_disturbance_penalty),
            hinf_exact_cost=float(hinf_exact["cost"].total_without_disturbance_penalty),
            output_config=output_config,
        )
        fits.append(fit)
        fits_by_label[fit.label] = fit
        _store_fit_arrays(arrays, fit)

    diagnostics = {
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "source_issues": [TIME_CONSTRAINED_ISSUE_ID],
        "standard_certificate_issue": STANDARD_CERTIFICATE_ISSUE_ID,
        "failure_decomposition_issue": FAILURE_DECOMPOSITION_ISSUE_ID,
        "gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "gamma": float(gamma_ref.gamma),
        "budget": budget,
        "budget_l2": budget_l2,
        "training_config": asdict(training_config),
        "output_config": asdict(output_config),
        "reference_clean_cost": output_feedback_cost(
            schedule, lqr_clean
        ).total_without_disturbance_penalty,
        "reference_under_epsilon_cost": output_feedback_cost(
            schedule,
            lqr_under_eps,
        ).total_without_disturbance_penalty,
        "reference_exact_l2_cost": float(lqr_exact["cost"].total_without_disturbance_penalty),
        "robust_exact_l2_cost": float(hinf_exact["cost"].total_without_disturbance_penalty),
    }
    return AffineTrackerResult(
        issue_id=ISSUE_ID,
        fits=tuple(fits),
        diagnostics=diagnostics,
        arrays=arrays,
    )


def _initial_parts_for_condition(
    condition: AffineTrackerCondition,
    *,
    K_ref: Float[Array, "T m_u n"],
    u_ref: Float[Array, "T m_u"],
    fits_by_label: dict[str, AffineTrackerFit],
) -> tuple[Float[Array, "T m_u n"], Float[Array, "T m_u"]]:
    """Resolve scratch/reference/previous-stage initialization for a row."""

    if condition.initialization_source == "reference":
        return K_ref, u_ref
    if condition.initialization_source == "previous_stage":
        if condition.stage_source_label is None:
            raise ValueError(f"{condition.label} requires stage_source_label.")
        try:
            source = fits_by_label[condition.stage_source_label]
        except KeyError as exc:
            raise ValueError(
                f"{condition.label} depends on missing stage {condition.stage_source_label!r}."
            ) from exc
        return source.K, source.u_ref
    return jnp.zeros_like(K_ref), jnp.zeros_like(u_ref)


@dataclass(frozen=True)
class _CoverageBundle:
    epsilons: Float[Array, "samples T m_w"]
    trajectory_weights: Float[Array, " samples"]
    x: Float[Array, "state_samples n"]
    xhat: Float[Array, "state_samples n"]
    times: Float[Array, " state_samples"]
    state_weights: Float[Array, " state_samples"]
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _coverage_samples(
    *,
    condition: AffineTrackerCondition,
    plant: PlantLinearization,
    schedule: Any,
    K_ref: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    budget_l2: float,
    gamma: float,
    output_config: OutputFeedbackConfig,
) -> _CoverageBundle:
    if condition.perturbation_family == "riccati_eps":
        raw = _riccati_epsilon_coverage_samples(
            plant=plant,
            schedule=schedule,
            K_ref=K_ref,
            x0=x0,
            budget_l2=budget_l2,
            gamma=gamma,
            output_config=output_config,
        )
    elif condition.perturbation_family == "mixed":
        raw = _mixed_coverage_samples(
            plant=plant,
            schedule=schedule,
            K_ref=K_ref,
            x0=x0,
            budget_l2=budget_l2,
            gamma=gamma,
            output_config=output_config,
        )
    elif condition.eigenspectrum_coverage is not None:
        raw = _eigenspectrum_coverage_samples(
            plant=plant,
            schedule=schedule,
            K_ref=K_ref,
            x0=x0,
            budget_l2=budget_l2,
            gamma=gamma,
            output_config=output_config,
            coverage_config=condition.eigenspectrum_coverage,
        )
    elif condition.observer_error_coverage is not None:
        raw = _observer_error_coverage_samples(
            plant=plant,
            schedule=schedule,
            K_ref=K_ref,
            x0=x0,
            budget_l2=budget_l2,
            output_config=output_config,
            coverage_config=condition.observer_error_coverage,
        )
    else:
        raw = (*_empty_coverage_samples(plant), {"enabled": False}, {})
    eps, traj_w, x, xhat, times, state_w, metadata, arrays = raw
    return _CoverageBundle(eps, traj_w, x, xhat, times, state_w, metadata, arrays)


def _riccati_epsilon_coverage_samples(
    *,
    plant: PlantLinearization,
    schedule: Any,
    K_ref: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    budget_l2: float,
    gamma: float,
    output_config: OutputFeedbackConfig,
):
    gamma_solution = (
        materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
        .gamma_references[0]
        .solution
    )
    covs = robust_estimator_covariances(plant, schedule, gamma, output_config)
    robust_gains = robust_output_feedback_gains(
        plant,
        schedule,
        gamma_solution,
        covs,
        output_config,
    )
    robust_policy = robust_estimator_fixed_adversary_policy(
        plant,
        schedule,
        gamma_solution,
        robust_gains,
        covs,
        output_config,
    )
    robust_rollout = rollout_with_robust_estimator_policy(
        plant,
        schedule,
        gamma_solution,
        x0,
        robust_policy,
        gains=robust_gains,
        config=output_config,
    )
    eps = robust_rollout.epsilon[None, :, :]
    return (
        eps,
        jnp.ones((1,), dtype=jnp.float64),
        jnp.zeros((0, plant.n), dtype=jnp.float64),
        jnp.zeros((0, plant.n), dtype=jnp.float64),
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.float64),
        {
            "enabled": True,
            "objective": "trajectory",
            "family": "riccati_eps",
            "budget_l2": float(budget_l2),
        },
        {"coverage_riccati_epsilons": np.asarray(eps), "coverage_weights": np.ones((1,))},
    )


def _mixed_coverage_samples(
    *,
    plant: PlantLinearization,
    schedule: Any,
    K_ref: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    budget_l2: float,
    gamma: float,
    output_config: OutputFeedbackConfig,
):
    ric = _riccati_epsilon_coverage_samples(
        plant=plant,
        schedule=schedule,
        K_ref=K_ref,
        x0=x0,
        budget_l2=budget_l2,
        gamma=gamma,
        output_config=output_config,
    )
    eig = _eigenspectrum_coverage_samples(
        plant=plant,
        schedule=schedule,
        K_ref=K_ref,
        x0=x0,
        budget_l2=budget_l2,
        gamma=gamma,
        output_config=output_config,
        coverage_config=EigenspectrumCoverageConfig(
            n_modes=4,
            scale=1.0,
            weight=0.25,
            objective="state",
        ),
    )
    obs = _observer_error_coverage_samples(
        plant=plant,
        schedule=schedule,
        K_ref=K_ref,
        x0=x0,
        budget_l2=budget_l2,
        output_config=output_config,
        coverage_config=ObserverErrorCoverageConfig(
            n_modes=1,
            scale=0.3,
            weight=0.25,
            objective="state",
        ),
    )
    eps = ric[0]
    traj_w = ric[1] * 2.0
    x = jnp.concatenate([eig[2], obs[2]], axis=0)
    xhat = jnp.concatenate([eig[3], obs[3]], axis=0)
    times = jnp.concatenate([eig[4], obs[4]], axis=0)
    state_w = jnp.concatenate([eig[5], obs[5]], axis=0)
    arrays = {
        "coverage_mixed_riccati_epsilons": np.asarray(eps),
        "coverage_mixed_x": np.asarray(x),
        "coverage_mixed_x_hat": np.asarray(xhat),
        "coverage_mixed_times": np.asarray(times),
        "coverage_mixed_state_weights": np.asarray(state_w),
    }
    return (
        eps,
        traj_w,
        x,
        xhat,
        times,
        state_w,
        {"enabled": True, "objective": "mixed", "family": "mixed", "riccati_primary": True},
        arrays,
    )


def _fit_condition(
    *,
    condition: AffineTrackerCondition,
    plant: PlantLinearization,
    schedule: Any,
    K_ref: Float[Array, "T m_u n"],
    u_ref_ref: Float[Array, "T m_u"],
    xhat_ref: Float[Array, "T_plus_1 n"],
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    state_scales: Float[Array, " n"],
    coverage: _CoverageBundle,
    initial_K: Float[Array, "T m_u n"],
    initial_u_ref: Float[Array, "T m_u"],
    x0: Float[Array, " n"],
    riccati_epsilon: Float[Array, "T m_w"],
    gamma: float,
    lqr_clean_rollout: OutputFeedbackRollout,
    lqr_under_epsilon_rollout: OutputFeedbackRollout,
    lqr_exact_cost: float,
    hinf_exact_cost: float,
    output_config: OutputFeedbackConfig,
) -> AffineTrackerFit:
    maps = _make_parameter_maps(
        condition,
        K_ref,
        u_ref_ref,
        state_scales,
        initial_K=initial_K,
        initial_u_ref=initial_u_ref,
    )
    theta0 = maps["initial_theta"]
    theta_ref = maps["reference_theta"]
    theta_zero = maps["zero_theta"]

    def objective_from_parts(K: Float[Array, "T m_u n"], u_ref: Float[Array, "T m_u"]):
        if condition.objective_family == "supervised_action_match":
            return _affine_action_match_objective(
                plant=plant,
                K=K,
                u_ref=u_ref,
                xhat_ref=xhat_ref,
                reference_K=K_ref,
                reference_u_ref=u_ref_ref,
                reference_xhat_ref=xhat_ref,
                coverage=coverage,
                x0=x0,
                riccati_epsilon=riccati_epsilon,
                config=output_config,
            )
        clean = _affine_clean_objective(
            plant=plant,
            schedule=schedule,
            K=K,
            u_ref=u_ref,
            xhat_ref=xhat_ref,
            states=states,
            weights=weights,
            config=output_config,
        )
        perturbation = _affine_coverage_objective(
            plant=plant,
            schedule=schedule,
            K=K,
            u_ref=u_ref,
            xhat_ref=xhat_ref,
            coverage=coverage,
            config=output_config,
        )
        return (
            condition.clean_objective_weight * clean
            + condition.perturbation_objective_weight * perturbation
        )

    def objective_theta(theta_flat: Float[Array, " flat"]) -> Float[Array, ""]:
        K, u_ref = maps["from_theta"](theta_flat)
        return objective_from_parts(K, u_ref)

    value_and_grad = jax.jit(jax.value_and_grad(objective_theta))

    def scipy_value_and_grad(theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(jnp.asarray(theta_np, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    objective_initial, grad_initial = scipy_value_and_grad(np.asarray(theta0, dtype=np.float64))
    objective_reference = float(objective_theta(theta_ref))
    objective_zero = float(objective_theta(theta_zero))
    if condition.row_kind == "reference_replay":
        theta_final = np.asarray(theta_ref, dtype=np.float64)
        objective_final, grad_final = scipy_value_and_grad(theta_final)
        status = "reference replay; no optimizer"
        success = True
        nit = 0
        nfev = 1
    elif condition.optimizer == "lbfgsb":
        scipy_result = scipy_opt.minimize(
            scipy_value_and_grad,
            np.asarray(theta0, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": condition.maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
        )
        theta_final = np.asarray(scipy_result.x, dtype=np.float64)
        objective_final, grad_final = scipy_value_and_grad(theta_final)
        status = str(scipy_result.message)
        success = bool(scipy_result.success)
        nit = int(scipy_result.nit)
        nfev = int(scipy_result.nfev)
    elif condition.optimizer == "adamw":
        theta_final, objective_final, grad_final, nit = _run_adam(
            value_and_grad,
            np.asarray(theta0, dtype=np.float64),
            condition,
        )
        status = "adamw"
        success = True
        nfev = nit
    else:
        raise ValueError(f"Unknown optimizer {condition.optimizer!r}.")

    K, u_ref = maps["from_theta"](jnp.asarray(theta_final, dtype=jnp.float64))
    clean_rollout = rollout_with_affine_tracker(
        plant,
        K,
        u_ref,
        xhat_ref,
        x0,
        config=output_config,
    )
    under_eps_rollout = rollout_with_affine_tracker(
        plant,
        K,
        u_ref,
        xhat_ref,
        x0,
        riccati_epsilon,
        config=output_config,
    )
    clean_cost = output_feedback_cost(schedule, clean_rollout).total_without_disturbance_penalty
    under_eps_cost = output_feedback_cost(
        schedule,
        under_eps_rollout,
    ).total_without_disturbance_penalty
    exact = affine_tracker_exact_adversary_audit(
        label=condition.label,
        plant=plant,
        schedule=schedule,
        K=K,
        u_ref=u_ref,
        xhat_ref=xhat_ref,
        x0=x0,
        budget=float(jnp.sum(riccati_epsilon**2)),
        penalty_gamma=gamma,
        config=output_config,
    )
    gamma_penalized = exact["gamma_penalized"]
    return AffineTrackerFit(
        condition=condition,
        K=K,
        u_ref=u_ref,
        xhat_ref=xhat_ref,
        objective_initial=float(objective_initial),
        objective_final=float(objective_final),
        objective_reference=float(objective_reference),
        objective_zero=float(objective_zero),
        objective_ratio_to_reference=float(objective_final / max(objective_reference, 1e-12)),
        feedforward_relative_error=float(
            jnp.linalg.norm(u_ref - u_ref_ref) / jnp.maximum(jnp.linalg.norm(u_ref_ref), 1e-12)
        ),
        gain_relative_error=float(
            jnp.linalg.norm(K - K_ref) / jnp.maximum(jnp.linalg.norm(K_ref), 1e-12)
        ),
        gradient_norm_initial=float(np.linalg.norm(grad_initial)),
        gradient_norm_final=float(np.linalg.norm(grad_final)),
        projected_gradient_norm_final=float(np.linalg.norm(grad_final)),
        optimizer_status=status,
        optimizer_success=success,
        n_iterations=nit,
        n_function_evaluations=nfev,
        clean_rollout=clean_rollout,
        clean_cost=float(clean_cost),
        clean_action_mismatch_ratio=_action_mismatch_ratio(
            clean_rollout.u,
            lqr_clean_rollout.u,
            floor=output_config.denominator_floor,
        ),
        under_epsilon_rollout=under_eps_rollout,
        under_epsilon_cost=float(under_eps_cost),
        under_epsilon_cost_ratio_to_lqr=float(
            under_eps_cost
            / max(
                lqr_under_epsilon_rollout.control_effort * 0.0
                + output_feedback_cost(
                    schedule,
                    lqr_under_epsilon_rollout,
                ).total_without_disturbance_penalty,
                1e-12,
            )
        ),
        under_epsilon_action_mismatch_ratio=_action_mismatch_ratio(
            under_eps_rollout.u,
            lqr_under_epsilon_rollout.u,
            floor=output_config.denominator_floor,
        ),
        exact_l2_cost=float(exact["cost"].total_without_disturbance_penalty),
        exact_l2_cost_ratio_to_lqr=float(
            exact["cost"].total_without_disturbance_penalty / max(lqr_exact_cost, 1e-12)
        ),
        exact_l2_cost_ratio_to_hinf=float(
            exact["cost"].total_without_disturbance_penalty / max(hinf_exact_cost, 1e-12)
        ),
        gamma_penalized_feasible=bool(gamma_penalized.get("feasible")),
        gamma_penalized_lambda_over_gamma_squared=float(
            gamma_penalized.get("max_eigenvalue_over_gamma_squared", np.nan)
        ),
    )


def _run_adam(value_and_grad, theta0: np.ndarray, condition: AffineTrackerCondition):
    theta = jnp.asarray(theta0, dtype=jnp.float64)
    optimizer = optax.adamw(condition.learning_rate)
    opt_state = optimizer.init(theta)
    last_value = jnp.asarray(np.nan, dtype=jnp.float64)
    last_grad = jnp.zeros_like(theta)
    for _ in range(condition.maxiter):
        value, grad = value_and_grad(theta)
        if condition.adam_clip_norm is not None:
            norm = jnp.linalg.norm(grad)
            grad = grad * jnp.minimum(1.0, condition.adam_clip_norm / jnp.maximum(norm, 1e-12))
        updates, opt_state = optimizer.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        last_value = value
        last_grad = grad
    return (
        np.asarray(theta, dtype=np.float64),
        float(last_value),
        np.asarray(last_grad, dtype=np.float64),
        condition.maxiter,
    )


def _make_parameter_maps(
    condition: AffineTrackerCondition,
    K_ref: Float[Array, "T m_u n"],
    u_ref: Float[Array, "T m_u"],
    state_scales: Float[Array, " n"],
    *,
    initial_K: Float[Array, "T m_u n"] | None = None,
    initial_u_ref: Float[Array, "T m_u"] | None = None,
) -> dict[str, Any]:
    K_ref = K_ref.astype(jnp.float64)
    u_ref = u_ref.astype(jnp.float64)
    K_zero = jnp.zeros_like(K_ref)
    u_zero = jnp.zeros_like(u_ref)
    if initial_K is not None and initial_u_ref is not None:
        initial_K = initial_K.astype(jnp.float64)
        initial_u = initial_u_ref.astype(jnp.float64)
    elif condition.row_kind == "reference_replay":
        initial_K = K_ref
        initial_u = u_ref
    elif condition.row_kind == "feedforward_only":
        initial_K = K_ref
        initial_u = u_zero
    elif condition.row_kind == "gain_only":
        initial_K = K_zero
        initial_u = u_ref
    else:
        initial_K = K_zero
        initial_u = u_zero

    basis = None
    projection_ref = None
    if condition.gain_basis_rank is not None:
        basis = TimeConstrainedGainParameterization.cubic_bspline(
            horizon=K_ref.shape[0],
            n_basis=condition.gain_basis_rank,
            action_dim=K_ref.shape[1],
            input_dim=K_ref.shape[2],
        )
        projection_ref = basis.project_gains(np.asarray(K_ref))
        basis_matrix = jnp.asarray(basis.basis, dtype=jnp.float64)
    else:
        basis_matrix = None

    gain_scale = state_scales[None, None, :]

    def encode_K(K: Float[Array, "T m_u n"]) -> Float[Array, " flat"]:
        if not condition.train_gain:
            return jnp.zeros((0,), dtype=jnp.float64)
        if basis is not None:
            projected = basis.project_gains(np.asarray(K))
            theta = jnp.asarray(projected.theta, dtype=jnp.float64)
            scale = jnp.mean(gain_scale, axis=0, keepdims=True)
            return (theta * scale).reshape(-1)
        return (K * gain_scale).reshape(-1)

    def decode_K(theta: Float[Array, " flat"]) -> Float[Array, "T m_u n"]:
        if not condition.train_gain:
            return initial_K
        if basis is not None:
            theta_basis_size = basis.n_basis * K_ref.shape[1] * K_ref.shape[2]
            theta_basis = theta[:theta_basis_size].reshape(
                (basis.n_basis, K_ref.shape[1], K_ref.shape[2])
            )
            scale = jnp.mean(gain_scale, axis=0, keepdims=True)
            theta_basis = theta_basis / scale
            assert basis_matrix is not None
            return jnp.einsum("tb,bai->tai", basis_matrix, theta_basis)
        size = K_ref.size
        return theta[:size].reshape(K_ref.shape) / gain_scale

    def encode_u(values: Float[Array, "T m_u"]) -> Float[Array, " flat"]:
        if not condition.train_feedforward:
            return jnp.zeros((0,), dtype=jnp.float64)
        return values.reshape(-1)

    def decode_u(theta: Float[Array, " flat"]) -> Float[Array, "T m_u"]:
        if not condition.train_feedforward:
            return initial_u
        k_size = encode_K(initial_K).size
        return theta[k_size:].reshape(u_ref.shape)

    def to_theta(K: Float[Array, "T m_u n"], u: Float[Array, "T m_u"]) -> Float[Array, " flat"]:
        return jnp.concatenate([encode_K(K), encode_u(u)], axis=0)

    def from_theta(theta_flat: Float[Array, " flat"]):
        theta_flat = theta_flat.astype(jnp.float64)
        return decode_K(theta_flat), decode_u(theta_flat)

    return {
        "from_theta": from_theta,
        "initial_theta": to_theta(initial_K, initial_u),
        "reference_theta": to_theta(K_ref, u_ref),
        "zero_theta": to_theta(K_zero, u_zero),
        "basis_projection_relative_residual": None
        if projection_ref is None
        else projection_ref.relative_residual,
    }


def _affine_clean_objective(
    *,
    plant: PlantLinearization,
    schedule: Any,
    K: Float[Array, "T m_u n"],
    u_ref: Float[Array, "T m_u"],
    xhat_ref: Float[Array, "T_plus_1 n"],
    states: Float[Array, "batch n"],
    weights: Float[Array, " batch"],
    config: OutputFeedbackConfig,
) -> Float[Array, ""]:
    epsilon = jnp.zeros((schedule.T, plant.m_w), dtype=jnp.float64)

    def one_cost(x0):
        x, _xhat, _y, u, _covs = affine_tracker_rollout_arrays(
            plant=plant,
            K=K,
            u_ref=u_ref,
            xhat_ref=xhat_ref,
            x0=x0,
            epsilon=epsilon,
            config=config,
        )
        return rollout_task_cost(schedule, x, u)

    costs = jax.vmap(one_cost)(states)
    return jnp.mean(costs * weights)


def _affine_coverage_objective(
    *,
    plant: PlantLinearization,
    schedule: Any,
    K: Float[Array, "T m_u n"],
    u_ref: Float[Array, "T m_u"],
    xhat_ref: Float[Array, "T_plus_1 n"],
    coverage: _CoverageBundle,
    config: OutputFeedbackConfig,
) -> Float[Array, ""]:
    if not coverage.metadata.get("enabled"):
        return jnp.asarray(0.0, dtype=jnp.float64)
    objective = coverage.metadata.get("objective")
    if objective == "trajectory":
        return _affine_coverage_trajectory_objective(
            plant,
            schedule,
            K,
            u_ref,
            xhat_ref,
            coverage.epsilons,
            coverage.trajectory_weights,
            config,
        )
    if objective == "state":
        return _affine_coverage_state_objective(
            plant,
            schedule,
            K,
            u_ref,
            xhat_ref,
            coverage.x,
            coverage.xhat,
            coverage.times,
            coverage.state_weights,
            config,
        )
    if objective == "mixed":
        return _affine_coverage_trajectory_objective(
            plant,
            schedule,
            K,
            u_ref,
            xhat_ref,
            coverage.epsilons,
            coverage.trajectory_weights,
            config,
        ) + _affine_coverage_state_objective(
            plant,
            schedule,
            K,
            u_ref,
            xhat_ref,
            coverage.x,
            coverage.xhat,
            coverage.times,
            coverage.state_weights,
            config,
        )
    raise ValueError(f"Unknown coverage objective {objective!r}.")


def _affine_action_match_objective(
    *,
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    u_ref: Float[Array, "T m_u"],
    xhat_ref: Float[Array, "T_plus_1 n"],
    reference_K: Float[Array, "T m_u n"],
    reference_u_ref: Float[Array, "T m_u"],
    reference_xhat_ref: Float[Array, "T_plus_1 n"],
    coverage: _CoverageBundle,
    x0: Float[Array, " n"],
    riccati_epsilon: Float[Array, "T m_w"],
    config: OutputFeedbackConfig,
) -> Float[Array, ""]:
    """Diagnostic supervised action matching, never used as a bridge success gate."""

    if coverage.epsilons.shape[0] == 0:
        epsilons = riccati_epsilon[None, :, :]
        weights = jnp.ones((1,), dtype=jnp.float64)
    else:
        epsilons = coverage.epsilons
        weights = coverage.trajectory_weights

    def trajectory_loss(epsilon):
        _x, xhat, _y, u, _covs = affine_tracker_rollout_arrays(
            plant=plant,
            K=K,
            u_ref=u_ref,
            xhat_ref=xhat_ref,
            x0=x0,
            epsilon=epsilon,
            config=config,
        )
        _rx, rxhat, _ry, ref_u, _rcovs = affine_tracker_rollout_arrays(
            plant=plant,
            K=reference_K,
            u_ref=reference_u_ref,
            xhat_ref=reference_xhat_ref,
            x0=x0,
            epsilon=epsilon,
            config=config,
        )
        return jnp.mean((u - ref_u) ** 2) + 1e-4 * jnp.mean((xhat - rxhat) ** 2)

    losses = jax.vmap(trajectory_loss)(epsilons)
    trajectory_term = jnp.sum(weights * losses) / jnp.maximum(jnp.sum(weights), 1e-30)
    if coverage.x.shape[0] == 0:
        return trajectory_term
    state_actions = jax.vmap(lambda xhat, time: u_ref[time] - K[time] @ (xhat - xhat_ref[time]))(
        coverage.xhat, coverage.times
    )
    reference_actions = jax.vmap(
        lambda xhat, time: (
            reference_u_ref[time] - reference_K[time] @ (xhat - reference_xhat_ref[time])
        )
    )(coverage.xhat, coverage.times)
    state_losses = jnp.mean((state_actions - reference_actions) ** 2, axis=-1)
    state_term = jnp.sum(coverage.state_weights * state_losses) / jnp.maximum(
        jnp.sum(coverage.state_weights),
        1e-30,
    )
    return trajectory_term + state_term


def _affine_coverage_trajectory_objective(
    plant,
    schedule,
    K,
    u_ref,
    xhat_ref,
    coverage_epsilons,
    coverage_weights,
    config,
):
    if coverage_epsilons.shape[0] == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)
    x0 = make_cs_output_feedback_initial_state(plant, config)

    def one_cost(epsilon):
        x, _xhat, _y, u, _covs = affine_tracker_rollout_arrays(
            plant=plant,
            K=K,
            u_ref=u_ref,
            xhat_ref=xhat_ref,
            x0=x0,
            epsilon=epsilon,
            config=config,
        )
        return rollout_task_cost(schedule, x, u)

    costs = jax.vmap(one_cost)(coverage_epsilons)
    return jnp.sum(coverage_weights * costs) / jnp.maximum(jnp.sum(coverage_weights), 1e-30)


def _affine_coverage_state_objective(
    plant,
    schedule,
    K,
    u_ref,
    xhat_ref,
    coverage_x,
    coverage_xhat,
    coverage_times,
    coverage_state_weights,
    config,
):
    if coverage_x.shape[0] == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)
    gains = kalman_estimator_gains(plant, K, config)
    H = delayed_observation_matrix(plant, config)

    def one_cost(x_start, xhat_start, start_time):
        x_t = x_start.astype(jnp.float64)
        xhat_t = xhat_start.astype(jnp.float64)
        cost = jnp.asarray(0.0, dtype=jnp.float64)
        for t in range(schedule.T):
            active = jnp.asarray(t, dtype=jnp.int32) >= start_time
            u_t = u_ref[t] - K[t] @ (xhat_t - xhat_ref[t])
            y_t = H @ x_t
            stage = x_t @ schedule.Q[t] @ x_t + u_t @ schedule.R[t] @ u_t
            xhat_next = plant.A @ xhat_t + plant.B @ u_t + gains[t] @ (y_t - H @ xhat_t)
            x_next = plant.A @ x_t + plant.B @ u_t
            cost = cost + jnp.where(active, stage, 0.0)
            x_t = jnp.where(active, x_next, x_t)
            xhat_t = jnp.where(active, xhat_next, xhat_t)
        return cost + x_t @ schedule.Q_f @ x_t

    costs = jax.vmap(one_cost)(coverage_x, coverage_xhat, coverage_times)
    return jnp.sum(costs * coverage_state_weights) / jnp.maximum(
        jnp.sum(coverage_state_weights),
        1e-30,
    )


def affine_tracker_exact_adversary_audit(
    *,
    label: str,
    plant: PlantLinearization,
    schedule: Any,
    K: Float[Array, "T m_u n"],
    u_ref: Float[Array, "T m_u"],
    xhat_ref: Float[Array, "T_plus_1 n"],
    x0: Float[Array, " n"],
    budget: float,
    penalty_gamma: float | None,
    config: OutputFeedbackConfig,
) -> dict[str, Any]:
    """Exact L2/gamma sidecar for the affine tracker law."""

    H_quad, g_quad, constant = _affine_flattened_epsilon_quadratic(
        plant=plant,
        schedule=schedule,
        K=K,
        u_ref=u_ref,
        xhat_ref=xhat_ref,
        x0=x0,
        config=config,
    )
    radius = float(np.sqrt(max(budget, 0.0)))
    eps_flat, variable_value, kkt_lambda, boundary_active = _maximize_quadratic_on_l2_ball(
        H_quad,
        g_quad,
        radius=radius,
    )
    epsilon = jnp.asarray(eps_flat.reshape((schedule.T, plant.m_w)), dtype=jnp.float64)
    rollout = rollout_with_affine_tracker(plant, K, u_ref, xhat_ref, x0, epsilon, config)
    cost = output_feedback_cost(schedule, rollout)
    result: dict[str, Any] = {
        "label": label,
        "epsilon": epsilon,
        "rollout": rollout,
        "cost": cost,
        "budget": budget,
        "budget_l2": radius,
        "epsilon_energy": float(jnp.sum(epsilon**2)),
        "quadratic_total": constant + variable_value,
        "rollout_total": cost.total_without_disturbance_penalty,
        "quadratic_rollout_abs_error": abs(
            constant + variable_value - cost.total_without_disturbance_penalty
        ),
        "kkt_lambda": kkt_lambda,
        "boundary_active": boundary_active,
        "max_eigenvalue": float(np.max(np.linalg.eigvalsh(H_quad))),
    }
    if penalty_gamma is not None:
        result["gamma_penalized"] = _gamma_penalized_quadratic_diagnostic(
            H_quad,
            g_quad,
            constant,
            gamma=float(penalty_gamma),
        )
    else:
        result["gamma_penalized"] = {"feasible": False, "max_eigenvalue_over_gamma_squared": np.nan}
    return result


def _affine_flattened_epsilon_quadratic(
    *,
    plant: PlantLinearization,
    schedule: Any,
    K: Float[Array, "T m_u n"],
    u_ref: Float[Array, "T m_u"],
    xhat_ref: Float[Array, "T_plus_1 n"],
    x0: Float[Array, " n"],
    config: OutputFeedbackConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    A_joint, G_joint = kalman_estimator_joint_matrices(plant, K, config)
    K_np = np.asarray(K, dtype=np.float64)
    u_ref_np = np.asarray(u_ref, dtype=np.float64)
    xhat_ref_np = np.asarray(xhat_ref, dtype=np.float64)
    T = int(schedule.T)
    z_dim = A_joint.shape[1]
    m_w = plant.m_w
    flat_dim = T * m_w
    S = np.zeros((z_dim, flat_dim), dtype=np.float64)
    c = np.concatenate([np.asarray(x0), np.asarray(x0)]).astype(np.float64)
    H_acc = np.zeros((flat_dim, flat_dim), dtype=np.float64)
    g_acc = np.zeros((flat_dim,), dtype=np.float64)
    const = 0.0
    for t in range(T):
        x_selector = np.concatenate(
            [np.eye(plant.n), np.zeros((plant.n, plant.n))],
            axis=1,
        )
        x_bar = x_selector @ c
        x_resp = x_selector @ S
        b_t = u_ref_np[t] + K_np[t] @ xhat_ref_np[t]
        u_selector = np.concatenate([np.zeros((plant.m_u, plant.n)), -K_np[t]], axis=1)
        u_bar = u_selector @ c + b_t
        u_resp = u_selector @ S
        q_t = np.asarray(schedule.Q[t], dtype=np.float64)
        r_t = np.asarray(schedule.R[t], dtype=np.float64)
        H_acc += x_resp.T @ q_t @ x_resp + u_resp.T @ r_t @ u_resp
        g_acc += x_resp.T @ q_t @ x_bar + u_resp.T @ r_t @ u_bar
        const += float(x_bar @ q_t @ x_bar + u_bar @ r_t @ u_bar)
        plant_b = np.asarray(plant.B, dtype=np.float64)
        c_offset = np.concatenate([plant_b @ b_t, plant_b @ b_t])
        S_next = np.asarray(A_joint[t]) @ S
        S_next[:, t * m_w : (t + 1) * m_w] += np.asarray(G_joint)
        c = np.asarray(A_joint[t]) @ c + c_offset
        S = S_next
    x_selector = np.concatenate([np.eye(plant.n), np.zeros((plant.n, plant.n))], axis=1)
    x_bar = x_selector @ c
    x_resp = x_selector @ S
    q_f = np.asarray(schedule.Q_f, dtype=np.float64)
    H_acc += x_resp.T @ q_f @ x_resp
    g_acc += x_resp.T @ q_f @ x_bar
    const += float(x_bar @ q_f @ x_bar)
    return 0.5 * (H_acc + H_acc.T), g_acc, const


def build_standard_rows(
    result: AffineTrackerResult,
    *,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Build d01c35a standard-certificate rows for affine tracker fits."""

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    output_config = OutputFeedbackConfig(**result.diagnostics["output_config"])
    rows = []
    for fit in result.fits:
        rows.extend(
            _standard_rows_for_fit(
                fit=fit,
                reference=reference,
                output_config=output_config,
                source_manifest=manifest_path,
            )
        )
    return rows


def _standard_rows_for_fit(
    *,
    fit: AffineTrackerFit,
    reference: Any,
    output_config: OutputFeedbackConfig,
    source_manifest: Path,
) -> list[dict[str, Any]]:
    rows = []
    x0 = make_cs_output_feedback_initial_state(reference.plant, output_config)
    evals = (
        ("nominal_clean", fit.clean_rollout, None),
        ("riccati_epsilon_response", fit.under_epsilon_rollout, fit.under_epsilon_rollout.epsilon),
    )
    for lens, rollout, epsilon in evals:
        reference_rollout = rollout_with_kalman_estimator(
            reference.plant,
            reference.lqr_solution.K,
            x0,
            epsilon,
            config=output_config,
        )
        components = _standard_components_for_rollout(
            fit=fit,
            reference=reference,
            output_config=output_config,
            rollout=rollout,
            reference_rollout=reference_rollout,
        )
        spec = BridgeRunSpec(
            issue_id=ISSUE_ID,
            run_id=make_bridge_run_id("affine_tracker", fit.label, lens),
            objective="diagnostic" if fit.condition.is_diagnostic else "optimal",
            architecture="free_time_varying",
            controller_label=fit.label,
            optimizer_label=fit.condition.optimizer,
            training_distribution=fit.condition.training_distribution,  # type: ignore[arg-type]
            evaluation_lane="deterministic",
            reference_controller="analytical_lqr_kalman_affine_replay",
            gamma_factor=OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
            parameters=_condition_parameters(fit.condition),
            notes=(
                "Affine tracker same-game row. Standard action mismatch uses actual "
                "candidate/reference action samples so feedforward replay is included; "
                "transition/value/Bellman components are evaluated on nominal-centered "
                "deviation dynamics."
            ),
        )
        rows.append(
            BridgeRunManifest(
                spec=spec,
                status="full_standard_certificate",
                metrics=_fit_metrics(fit) | {"evaluation_lens": lens},
                artifacts={"source_manifest": _repo_relative(source_manifest)},
                certificate_components=components,
            ).to_json_dict()
        )
    return rows


def _standard_components_for_rollout(
    *,
    fit: AffineTrackerFit,
    reference: Any,
    output_config: OutputFeedbackConfig,
    rollout: OutputFeedbackRollout,
    reference_rollout: OutputFeedbackRollout,
):
    plant = reference.plant
    schedule = reference.schedule
    K_ref = np.asarray(reference.lqr_solution.K)
    x_nom = np.asarray(reference_rollout.x)
    xhat_nom = np.asarray(fit.xhat_ref)
    coupled = np.concatenate(
        [
            np.asarray(rollout.x) - x_nom,
            np.asarray(rollout.x_hat) - xhat_nom,
        ],
        axis=-1,
    )[None, :, :]
    action_states = (np.asarray(rollout.x_hat) - xhat_nom)[None, :, :]
    candidate_transition = np.asarray(
        kalman_estimator_joint_matrices(plant, fit.K, output_config)[0]
    )
    reference_transition = np.asarray(
        kalman_estimator_joint_matrices(plant, reference.lqr_solution.K, output_config)[0]
    )
    return build_standard_certificate_components(
        architecture="free_time_varying",
        states=coupled,
        action_states=action_states,
        candidate_actions=np.asarray(rollout.u)[None, :, :],
        reference_actions=np.asarray(reference_rollout.u)[None, :, :],
        candidate_gain=np.asarray(fit.K),
        reference_gain=K_ref,
        action_weight=np.asarray(schedule.R),
        candidate_transition=candidate_transition,
        reference_transition=reference_transition,
        candidate_value_matrices=_policy_value_matrices(
            schedule, np.asarray(fit.K), candidate_transition
        ),
        reference_value_matrices=_policy_value_matrices(schedule, K_ref, reference_transition),
        bellman_hessian=_bellman_hessian(schedule, plant, reference.lqr_solution.P),
        optimizer_metadata={
            "status": fit.optimizer_status,
            "success": fit.optimizer_success,
            "n_iterations": fit.n_iterations,
            "n_function_evaluations": fit.n_function_evaluations,
        },
        state_label="nominal_centered_joint_state",
        action_state_label="nominal_centered_estimated_state",
        action_label="control",
    )


def result_summary(result: AffineTrackerResult, *, manifest_path: Path) -> dict[str, Any]:
    """Return JSON-compatible tracked summary with certificate rows."""

    standard_rows = build_standard_rows(result, manifest_path=manifest_path)
    summary = {
        "format": "rlrmp.output_feedback_affine_tracker.v1",
        "issue": result.issue_id,
        "umbrella": UMBRELLA_ID,
        "source_issues": [TIME_CONSTRAINED_ISSUE_ID],
        "standard_certificate_issue": STANDARD_CERTIFICATE_ISSUE_ID,
        "failure_decomposition_issue": FAILURE_DECOMPOSITION_ISSUE_ID,
        "scope": (
            "Staged same-game affine tracker curriculum rows: clean scratch baseline, "
            "clean feedforward stage, reward-objective feedback and joint rows for "
            "Riccati-epsilon, state-eigenspectrum, observer-error, and mixed "
            "perturbation families, plus isolated supervised action-match diagnostics."
        ),
        "non_goals": (
            "No GRU, robust/H-infinity training arm, direct teacher-cloning success "
            "claim, or old Delta-v decoupling acid-test criterion."
        ),
        "diagnostics": result.diagnostics,
        "fits": [_fit_summary(fit) for fit in result.fits],
        "standard_certificate": {
            "rows": standard_rows,
            "n_rows": len(standard_rows),
            "status_counts": _counts(row["status"] for row in standard_rows),
        },
        "artifact_npz_keys": sorted(result.arrays.keys()),
    }
    return summary


def write_basic_outputs(
    *,
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    note_path: Path,
    manifest_path: Path,
    artifact_path: Path,
) -> None:
    """Write tracked note/manifest and ignored bulk arrays."""

    mkdir_p(note_path.parent)
    mkdir_p(artifact_path.parent)
    results_dir = note_path.parents[1]
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Affine tracker output-feedback bridge diagnostics for issue 50c260d. "
            "See `notes/output_feedback_affine_tracker.md`.\n",
            encoding="utf-8",
        )
    np.savez_compressed(artifact_path, **arrays)
    summary["tracked_note"] = _repo_relative(note_path)
    summary["tracked_manifest"] = _repo_relative(manifest_path)
    summary["artifact_npz"] = _repo_relative(artifact_path)
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the tracked affine tracker note."""

    fit_rows = [
        "| row | objective | training distribution | diagnostic | objective ratio | feedforward rel err | gain rel err | clean action mismatch | under-eps action mismatch | exact L2 ratio | lambda/gamma^2 | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for fit in summary["fits"]:
        fit_rows.append(
            "| "
            f"{fit['label']} | "
            f"{fit['condition']['objective_family']} | "
            f"{fit['condition']['training_distribution']} | "
            f"{fit['diagnostic_only']} | "
            f"{fit['objective_ratio_to_reference']:.8g} | "
            f"{fit['feedforward_relative_error']:.8g} | "
            f"{fit['gain_relative_error']:.8g} | "
            f"{fit['clean_action_mismatch_ratio']:.8g} | "
            f"{fit['under_epsilon_action_mismatch_ratio']:.8g} | "
            f"{fit['exact_l2_cost_ratio_to_lqr']:.8g} | "
            f"{fit['gamma_penalized_lambda_over_gamma_squared']:.8g} | "
            f"{fit['optimizer_status']} |"
        )
    certificate_rows = summary["standard_certificate"]["rows"]
    cert_table = [
        "| row | lens | status | action mismatch | transition mismatch | value gap | Bellman residual |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in certificate_rows:
        components = {component["name"]: component for component in row["certificate_components"]}
        cert_table.append(
            "| "
            f"{row['spec']['controller_label']} | "
            f"{row['metrics']['evaluation_lens']} | "
            f"{row['status']} | "
            f"{_component_value(components, 'state_weighted_action_mismatch', 'mismatch_ratio_mean')} | "
            f"{_component_value(components, 'closed_loop_transition_mismatch', 'mismatch_ratio_mean')} | "
            f"{_component_value(components, 'value_policy_gap', 'gap_ratio_mean')} | "
            f"{_component_value(components, 'bellman_hessian_residual', 'residual_ratio_mean')} |"
        )
    failure = summary.get("failure_decomposition", {})
    failure_rows = [
        "| classification | rows |",
        "|---|---:|",
    ]
    for label, count in failure.get("classification_counts", {}).items():
        failure_rows.append(f"| {label} | {count} |")
    verdict = _verdict(summary)
    return f"""# Affine Tracker Output-Feedback Bridge

Issue: `{summary["issue"]}`. Umbrella: `{summary["umbrella"]}`.

Scope: {summary["scope"]}

Non-goals: {summary["non_goals"]}

## Verdict

{verdict}

## Same-Game Affine Bridge Rows

{chr(10).join(fit_rows)}

## Standard Certificate Rows

{chr(10).join(cert_table)}

## Failure Decomposition

{chr(10).join(failure_rows)}

The failure decomposition is the `c45adde` companion diagnostic. It explains
failed rows but does not replace the standard bridge gate.

<sup>1</sup> State-weighted action mismatch and Bellman-Hessian residual can match
exactly when the Bellman action Hessian is a scalar multiple of the action cost
geometry on that row. In that case they are the same evidence expressed through
two certificate views; they diverge when downstream value geometry weights
action directions differently.

<sup>2</sup> Gain mismatch is a diagnostic sidecar, not the bridge gate. The gate
is disturbance-relevant same-game behavior under the standard certificate
components.

## Historical Regulator/Tracker Comparison

This is not the old `410d7ac` / `d448c9d` tracker MVP. Those rows were a
Delta-v decoupling acid test with degenerate or trivial `x_nom` structure. The
50c260d row replays the analytical nominal output-feedback trajectory and
action sequence from the same C&S game, then optimizes only the decomposition
between feedforward command and estimated-state feedback correction. The success
criterion is preservation or scratch discovery under the standard bridge
certificate and failure decomposition, not a revived decoupling demonstration.

## Output Files

- Tracked manifest: `{summary.get("tracked_manifest", "")}`
- Ignored bulk arrays: `{summary.get("artifact_npz", "")}`
"""


def _verdict(summary: dict[str, Any]) -> str:
    fits = summary["fits"]
    main_rows = [fit for fit in fits if fit["condition"]["objective_family"] == "reward_rollout"]
    joint_rows = [
        fit
        for fit in main_rows
        if fit["condition"]["train_feedforward"] and fit["condition"]["train_gain"]
    ]
    diagnostic_rows = [
        fit for fit in fits if fit["condition"]["objective_family"] == "supervised_action_match"
    ]
    best_joint = min(joint_rows, key=lambda fit: fit["clean_action_mismatch_ratio"])
    joint_mixed = next((fit for fit in fits if fit["label"] == "affine_joint_mixed"), None)
    final_row = joint_mixed or best_joint
    rescued = (
        final_row["clean_action_mismatch_ratio"] <= 0.05
        and final_row["exact_l2_cost_ratio_to_lqr"] <= 1.1
    )
    diagnostic_text = (
        f" The supervised diagnostic rows are present but excluded from this verdict "
        f"(`{len(diagnostic_rows)}` rows)."
        if diagnostic_rows
        else ""
    )
    return (
        "The staged feedback curriculum "
        f"{'rescues' if rescued else 'does not rescue'} nominal same-game recovery "
        "under the retained bounded materialization. "
        f"The final mixed joint row has clean action mismatch "
        f"`{final_row['clean_action_mismatch_ratio']:.3g}` and exact-L2 ratio "
        f"`{final_row['exact_l2_cost_ratio_to_lqr']:.3g}`. "
        f"The best reward-objective joint row is `{best_joint['label']}` with clean "
        f"action mismatch `{best_joint['clean_action_mismatch_ratio']:.3g}` and "
        f"exact-L2 ratio `{best_joint['exact_l2_cost_ratio_to_lqr']:.3g}`. "
        "This verdict is based only on from-scratch/reward rows, not analytical action "
        f"labels.{diagnostic_text}"
    )


def _fit_summary(fit: AffineTrackerFit) -> dict[str, Any]:
    condition = asdict(fit.condition)
    condition["training_distribution"] = fit.condition.training_distribution
    return {
        "label": fit.label,
        "condition": condition,
        "initialization": fit.condition.initialization_source,
        "diagnostic_only": fit.condition.is_diagnostic,
        "objective_initial": fit.objective_initial,
        "objective_final": fit.objective_final,
        "objective_reference": fit.objective_reference,
        "objective_zero": fit.objective_zero,
        "objective_ratio_to_reference": fit.objective_ratio_to_reference,
        "feedforward_relative_error": fit.feedforward_relative_error,
        "gain_relative_error": fit.gain_relative_error,
        "gradient_norm_initial": fit.gradient_norm_initial,
        "gradient_norm_final": fit.gradient_norm_final,
        "projected_gradient_norm_final": fit.projected_gradient_norm_final,
        "best_objective": fit.objective_final,
        "best_checkpoint_iteration": None,
        "optimizer_status": fit.optimizer_status,
        "optimizer_success": fit.optimizer_success,
        "n_iterations": fit.n_iterations,
        "n_function_evaluations": fit.n_function_evaluations,
        "clean_cost": fit.clean_cost,
        "clean_action_mismatch_ratio": fit.clean_action_mismatch_ratio,
        "under_epsilon_cost": fit.under_epsilon_cost,
        "under_epsilon_cost_ratio_to_lqr": fit.under_epsilon_cost_ratio_to_lqr,
        "under_epsilon_action_mismatch_ratio": fit.under_epsilon_action_mismatch_ratio,
        "exact_l2_cost": fit.exact_l2_cost,
        "exact_l2_cost_ratio_to_lqr": fit.exact_l2_cost_ratio_to_lqr,
        "exact_l2_cost_ratio_to_hinf": fit.exact_l2_cost_ratio_to_hinf,
        "gamma_penalized_feasible": fit.gamma_penalized_feasible,
        "gamma_penalized_lambda_over_gamma_squared": fit.gamma_penalized_lambda_over_gamma_squared,
        "clean_rollout": _rollout_summary(fit.clean_rollout),
        "under_epsilon_rollout": _rollout_summary(fit.under_epsilon_rollout),
    }


def _fit_metrics(fit: AffineTrackerFit) -> dict[str, Any]:
    summary = _fit_summary(fit)
    return {
        key: summary[key]
        for key in (
            "objective_ratio_to_reference",
            "feedforward_relative_error",
            "gain_relative_error",
            "clean_cost",
            "clean_action_mismatch_ratio",
            "under_epsilon_cost_ratio_to_lqr",
            "under_epsilon_action_mismatch_ratio",
            "exact_l2_cost_ratio_to_lqr",
            "exact_l2_cost_ratio_to_hinf",
            "gamma_penalized_lambda_over_gamma_squared",
        )
    }


def _condition_parameters(condition: AffineTrackerCondition) -> dict[str, Any]:
    data = {
        "controller_family": "affine_tracker",
        "distribution_family": "affine tracker",
        "row_kind": condition.row_kind,
        "train_feedforward": condition.train_feedforward,
        "train_gain": condition.train_gain,
        "gain_basis_rank": condition.gain_basis_rank,
        "perturbation_family": condition.perturbation_family,
        "objective_family": condition.objective_family,
        "initialization_source": condition.initialization_source,
        "stage_source_label": condition.stage_source_label,
        "clean_objective_weight": condition.clean_objective_weight,
        "perturbation_objective_weight": condition.perturbation_objective_weight,
    }
    if condition.eigenspectrum_coverage is not None:
        data["coverage_family"] = "state_eigenspectrum"
        data["coverage"] = asdict(condition.eigenspectrum_coverage)
    if condition.observer_error_coverage is not None:
        data["coverage_family"] = "observer_error"
        data["coverage"] = asdict(condition.observer_error_coverage)
    return data


def _rollout_summary(rollout: OutputFeedbackRollout) -> dict[str, Any]:
    return {
        "peak_forward_velocity": rollout.peak_forward_velocity,
        "peak_forward_velocity_idx": rollout.peak_forward_velocity_idx,
        "terminal_position_error_m": rollout.terminal_position_error,
        "control_effort": rollout.control_effort,
    }


def _store_fit_arrays(arrays: dict[str, np.ndarray], fit: AffineTrackerFit) -> None:
    prefix = fit.label
    arrays[f"{prefix}_K"] = np.asarray(fit.K)
    arrays[f"{prefix}_u_ref"] = np.asarray(fit.u_ref)
    arrays[f"{prefix}_xhat_ref"] = np.asarray(fit.xhat_ref)
    arrays[f"{prefix}_clean_x"] = np.asarray(fit.clean_rollout.x)
    arrays[f"{prefix}_clean_x_hat"] = np.asarray(fit.clean_rollout.x_hat)
    arrays[f"{prefix}_clean_u"] = np.asarray(fit.clean_rollout.u)
    arrays[f"{prefix}_under_eps_x"] = np.asarray(fit.under_epsilon_rollout.x)
    arrays[f"{prefix}_under_eps_x_hat"] = np.asarray(fit.under_epsilon_rollout.x_hat)
    arrays[f"{prefix}_under_eps_u"] = np.asarray(fit.under_epsilon_rollout.u)


def _policy_value_matrices(schedule: Any, gains: np.ndarray, transition: np.ndarray) -> np.ndarray:
    n = int(schedule.Q.shape[-1])
    zeros = np.zeros((n, n), dtype=float)
    terminal = np.block([[np.asarray(schedule.Q_f, dtype=float), zeros], [zeros, zeros]])
    next_value = terminal
    values = [terminal]
    for q_t, r_t, k_t, a_t in reversed(
        list(zip(schedule.Q, schedule.R, gains, transition, strict=True))
    ):
        stage = np.block(
            [
                [np.asarray(q_t, dtype=float), zeros],
                [zeros, np.asarray(k_t).T @ np.asarray(r_t) @ np.asarray(k_t)],
            ]
        )
        next_value = stage + np.asarray(a_t).T @ next_value @ np.asarray(a_t)
        next_value = 0.5 * (next_value + next_value.T)
        values.append(next_value)
    return np.asarray(list(reversed(values))[:-1])


def _bellman_hessian(schedule: Any, plant: PlantLinearization, P: Float[Array, "T n n"]):
    hessians = []
    for r_t, p_next in zip(schedule.R, P[1:], strict=True):
        hessians.append(np.asarray(r_t + plant.B.T @ p_next @ plant.B, dtype=float))
    return np.asarray(hessians)


def _action_mismatch_ratio(candidate_u, reference_u, *, floor: float) -> float:
    mismatch = jnp.sqrt(jnp.mean((candidate_u - reference_u) ** 2))
    reference = jnp.sqrt(jnp.mean(reference_u**2))
    return float(mismatch / jnp.maximum(reference, floor))


def _component_value(components: dict[str, Any], name: str, key: str) -> str:
    component = components.get(name)
    if component is None:
        return ""
    value = component.get("summary", {}).get(key)
    if value is None:
        return ""
    return f"{float(value):.6g}"


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        parts = path.parts
        if "_artifacts" in parts:
            idx = parts.index("_artifacts")
            return str(Path(*parts[idx:]))
        return str(path)


def timed_run(**kwargs: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run and summarize with elapsed time metadata."""

    start = time.perf_counter()
    manifest_path = kwargs.pop(
        "manifest_path",
        REPO_ROOT / "results" / ISSUE_ID / "notes" / "output_feedback_affine_tracker_manifest.json",
    )
    result = run_affine_tracker_bridge(**kwargs)
    summary = result_summary(result, manifest_path=manifest_path)
    summary["elapsed_seconds"] = time.perf_counter() - start
    return summary, result.arrays


__all__ = [
    "AffineTrackerCondition",
    "AffineTrackerFit",
    "AffineTrackerResult",
    "baseline_conditions",
    "selected_coverage_conditions",
    "staged_curriculum_conditions",
    "rollout_with_affine_tracker",
    "affine_tracker_rollout_arrays",
    "affine_tracker_exact_adversary_audit",
    "run_affine_tracker_bridge",
    "result_summary",
    "render_markdown",
    "timed_run",
    "write_basic_outputs",
]
