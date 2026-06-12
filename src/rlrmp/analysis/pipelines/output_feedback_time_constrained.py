"""Smooth time-basis output-feedback bridge for issue 87edaae."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.optimize as scipy_opt
from jaxtyping import Array, Float

from rlrmp.analysis.pipelines.bridge_controllers import TimeConstrainedGainParameterization
from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.linear_round_trip import LinearTrainingConfig, rollout_task_cost
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    OutputFeedbackRollout,
    exact_output_feedback_adversary_audit,
    make_cs_output_feedback_initial_state,
    output_feedback_clean_objective,
    output_feedback_cost,
    robust_estimator_covariances,
    robust_estimator_fixed_adversary_policy,
    robust_output_feedback_gains,
    rollout_with_kalman_estimator,
    rollout_with_robust_estimator_policy,
    train_output_feedback_lqr_bellman_controller,
)
from rlrmp.analysis.pipelines.output_feedback_rollout_recovery import (
    _action_mismatch_ratio,
    _coverage_state_objective,
    _coverage_trajectory_objective,
    _effective_rank_from_covariance,
    _eigenspectrum_coverage_samples,
    _empty_coverage_samples,
    _observer_error_coverage_samples,
    _state_scales,
    _training_ensemble,
    EigenspectrumCoverageConfig,
    ObserverErrorCoverageConfig,
)
from rlrmp.analysis.math.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    build_rerun_metadata,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


jax.config.update("jax_enable_x64", True)

ISSUE_ID = "87edaae"
UMBRELLA_ID = "43e8728"
SOURCE_ISSUE_ID = "7a459bb"
OPTIMIZER_BRIDGE_ISSUE_ID = "1c014e5"
STANDARD_CERTIFICATE_ISSUE_ID = "d01c35a"
FAILURE_DECOMPOSITION_ISSUE_ID = "c45adde"
SPLINE_RANKS: tuple[int, ...] = (3, 5, 8, 12, 20, 60)
DEFAULT_ADAMW_LRS: tuple[float, ...] = (3e-3, 1e-2)


@dataclass(frozen=True)
class TimeBasisCondition:
    """One optimizer row for the smooth time-basis bridge."""

    rank: int
    initialization: str
    optimizer: str = "lbfgsb"
    learning_rate: float = 1e-3
    maxiter: int = 2000
    polish_maxiter: int | None = None
    adam_clip_norm: float | None = 1e4
    use_whitening: bool = True
    eigenspectrum_coverage: EigenspectrumCoverageConfig | None = None
    observer_error_coverage: ObserverErrorCoverageConfig | None = None

    @property
    def label(self) -> str:
        """Stable row label."""

        if self.optimizer == "lbfgsb":
            suffix = "lbfgsb_whitened" if self.use_whitening else "lbfgsb"
        elif self.optimizer == "adamw":
            suffix = f"adamw_lr_{_float_label(self.learning_rate)}"
        elif self.optimizer == "adamw_then_lbfgsb":
            suffix = f"adamw_then_lbfgsb_lr_{_float_label(self.learning_rate)}"
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer!r}.")
        return f"spline_r{self.rank}__{self.initialization}_{suffix}{self._coverage_suffix()}"

    def _coverage_suffix(self) -> str:
        """Return a stable label suffix for optional coverage objectives."""

        if self.eigenspectrum_coverage is not None and self.observer_error_coverage is not None:
            raise ValueError("Time-basis conditions cannot combine coverage config types.")
        if self.eigenspectrum_coverage is not None:
            coverage = self.eigenspectrum_coverage
            return (
                f"_eigen_{coverage.objective}_m{coverage.n_modes}"
                f"_s{_float_label(coverage.scale)}_w{_float_label(coverage.weight)}"
            )
        if self.observer_error_coverage is not None:
            coverage = self.observer_error_coverage
            return (
                f"_observer_error_{coverage.objective}_m{coverage.n_modes}"
                f"_s{_float_label(coverage.scale)}_w{_float_label(coverage.weight)}"
            )
        return ""


@dataclass(frozen=True)
class TimeBasisProjection:
    """Projection-only representability row."""

    rank: int
    label: str
    theta: Float[Array, "rank action state"]
    K: Float[Array, "time action state"]
    residual_norm: float
    relative_residual: float
    objective_ratio_to_reference: float
    gain_relative_error: float
    clean_action_mismatch_ratio: float
    exact_l2_cost_ratio_to_lqr: float
    gamma_penalized_lambda_over_gamma_squared: float


@dataclass(frozen=True)
class TimeBasisFit:
    """One trained smooth time-basis row."""

    condition: TimeBasisCondition
    theta: Float[Array, "rank action state"]
    K: Float[Array, "time action state"]
    objective_initial: float
    objective_final: float
    objective_reference: float
    objective_zero: float
    objective_ratio_to_reference: float
    gain_relative_error: float
    gradient_norm_initial: float
    gradient_norm_final: float
    best_objective: float
    best_checkpoint_iteration: int | None
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
class TimeBasisResult:
    """Complete smooth time-basis bridge result."""

    issue_id: str
    projections: tuple[TimeBasisProjection, ...]
    fits: tuple[TimeBasisFit, ...]
    diagnostics: dict[str, Any]
    arrays: dict[str, np.ndarray]


def run_time_basis_bridge(
    *,
    ranks: tuple[int, ...] = SPLINE_RANKS,
    fit_ranks: tuple[int, ...] | None = None,
    adamw_lrs: tuple[float, ...] = DEFAULT_ADAMW_LRS,
    lbfgsb_maxiter: int = 2000,
    adamw_steps: int = 5000,
    polish_maxiter: int = 1000,
    adamw_clip_norm: float = 1e4,
    training_config: LinearTrainingConfig = LinearTrainingConfig(n_steps=500),
    output_config: OutputFeedbackConfig = OutputFeedbackConfig(),
    coverage_conditions: tuple[TimeBasisCondition, ...] = (),
) -> TimeBasisResult:
    """Run projection checks and bounded training rows."""

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    plant = reference.plant
    schedule = reference.schedule
    K_ref = reference.lqr_solution.K
    x0 = make_cs_output_feedback_initial_state(plant, output_config)
    states, weights = _training_ensemble(plant, training_config, output_config)
    state_scales = _state_scales(states, weights)
    bellman = train_output_feedback_lqr_bellman_controller(
        reference,
        LinearTrainingConfig(n_steps=200, seed=training_config.seed),
    )
    lqr_clean = rollout_with_kalman_estimator(plant, K_ref, x0, config=output_config)
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
    lqr_under_eps = rollout_with_kalman_estimator(
        plant,
        K_ref,
        x0,
        riccati_epsilon,
        config=output_config,
    )
    lqr_exact = exact_output_feedback_adversary_audit(
        label="analytical_lqr_kalman",
        plant=plant,
        schedule=schedule,
        controller_gains=K_ref,
        x0=x0,
        budget=budget,
        estimator_kind="kalman",
        penalty_gamma=gamma_ref.gamma,
        config=output_config,
    )
    hinf_exact = exact_output_feedback_adversary_audit(
        label="analytical_hinf_robust",
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
    reference_objective = float(
        output_feedback_clean_objective(
            plant,
            schedule,
            K_ref,
            states,
            weights,
            output_config,
        )
    )

    bases = {
        rank: TimeConstrainedGainParameterization.cubic_bspline(
            horizon=schedule.T,
            n_basis=rank,
            action_dim=plant.m_u,
            input_dim=plant.n,
        )
        for rank in ranks
    }
    arrays: dict[str, np.ndarray] = {
        "lqr_reference_K": np.asarray(K_ref),
        "bellman_initial_K": np.asarray(bellman.K),
        "lqr_clean_x": np.asarray(lqr_clean.x),
        "lqr_clean_x_hat": np.asarray(lqr_clean.x_hat),
        "lqr_clean_u": np.asarray(lqr_clean.u),
        "lqr_under_eps_x": np.asarray(lqr_under_eps.x),
        "lqr_under_eps_x_hat": np.asarray(lqr_under_eps.x_hat),
        "lqr_under_eps_u": np.asarray(lqr_under_eps.u),
        "riccati_epsilon": np.asarray(riccati_epsilon),
        "initial_states": np.asarray(states),
        "initial_state_weights": np.asarray(weights),
        "state_scales": np.asarray(state_scales),
    }
    projections = []
    for rank, parameterization in bases.items():
        arrays[f"spline_r{rank}_basis"] = parameterization.basis
        projection = _projection_row(
            rank=rank,
            parameterization=parameterization,
            K_ref=K_ref,
            reference_objective=reference_objective,
            plant=plant,
            schedule=schedule,
            states=states,
            weights=weights,
            x0=x0,
            budget=budget,
            gamma=gamma_ref.gamma,
            lqr_clean=lqr_clean,
            lqr_exact_cost=lqr_exact["cost"].total_without_disturbance_penalty,
            output_config=output_config,
        )
        projections.append(projection)
        arrays[f"{projection.label}_K"] = np.asarray(projection.K)
        arrays[f"{projection.label}_theta"] = np.asarray(projection.theta)

    retained = fit_ranks or _default_retained_ranks(projections)
    conditions = _conditions_for_ranks(
        retained,
        adamw_lrs=adamw_lrs,
        lbfgsb_maxiter=lbfgsb_maxiter,
        adamw_steps=adamw_steps,
        polish_maxiter=polish_maxiter,
        adamw_clip_norm=adamw_clip_norm,
    ) + coverage_conditions
    fits = []
    eigenspectrum_coverage_by_label: dict[str, dict[str, Any]] = {}
    observer_error_coverage_by_label: dict[str, dict[str, Any]] = {}
    coverage_arrays: dict[str, np.ndarray] = {}
    for condition in conditions:
        _validate_condition(condition)
        if condition.eigenspectrum_coverage is None and condition.observer_error_coverage is None:
            (
                coverage_epsilons,
                coverage_trajectory_weights,
                coverage_x,
                coverage_xhat,
                coverage_times,
                coverage_state_weights,
            ) = _empty_coverage_samples(plant)
        elif condition.eigenspectrum_coverage is not None:
            (
                coverage_epsilons,
                coverage_trajectory_weights,
                coverage_x,
                coverage_xhat,
                coverage_times,
                coverage_state_weights,
                coverage_metadata,
                condition_arrays,
            ) = _eigenspectrum_coverage_samples(
                plant=plant,
                schedule=schedule,
                K_ref=K_ref,
                x0=x0,
                budget_l2=float(jnp.sqrt(jnp.asarray(budget))),
                gamma=gamma_ref.gamma,
                output_config=output_config,
                coverage_config=condition.eigenspectrum_coverage,
            )
            eigenspectrum_coverage_by_label[condition.label] = coverage_metadata
            for key, value in condition_arrays.items():
                coverage_arrays[f"{condition.label}_{key}"] = value
        else:
            assert condition.observer_error_coverage is not None
            (
                coverage_epsilons,
                coverage_trajectory_weights,
                coverage_x,
                coverage_xhat,
                coverage_times,
                coverage_state_weights,
                coverage_metadata,
                condition_arrays,
            ) = _observer_error_coverage_samples(
                plant=plant,
                schedule=schedule,
                K_ref=K_ref,
                x0=x0,
                budget_l2=float(jnp.sqrt(jnp.asarray(budget))),
                output_config=output_config,
                coverage_config=condition.observer_error_coverage,
            )
            observer_error_coverage_by_label[condition.label] = coverage_metadata
            for key, value in condition_arrays.items():
                coverage_arrays[f"{condition.label}_{key}"] = value
        initial_K = jnp.zeros_like(K_ref)
        if condition.initialization == "bellman_projected":
            initial_K = bellman.K
        fit = _fit_condition(
            condition=condition,
            parameterization=bases[condition.rank],
            initial_K=initial_K,
            K_ref=K_ref,
            reference_objective=reference_objective,
            plant=plant,
            schedule=schedule,
            states=states,
            weights=weights,
            state_scales=state_scales,
            coverage_epsilons=coverage_epsilons,
            coverage_trajectory_weights=coverage_trajectory_weights,
            coverage_x=coverage_x,
            coverage_xhat=coverage_xhat,
            coverage_times=coverage_times,
            coverage_state_weights=coverage_state_weights,
            x0=x0,
            riccati_epsilon=riccati_epsilon,
            budget=budget,
            gamma=gamma_ref.gamma,
            lqr_clean=lqr_clean,
            lqr_under_eps=lqr_under_eps,
            lqr_exact_cost=lqr_exact["cost"].total_without_disturbance_penalty,
            hinf_exact_cost=hinf_exact["cost"].total_without_disturbance_penalty,
            output_config=output_config,
        )
        fits.append(fit)
        arrays[f"{fit.label}_K"] = np.asarray(fit.K)
        arrays[f"{fit.label}_theta"] = np.asarray(fit.theta)
        arrays[f"{fit.label}_clean_x"] = np.asarray(fit.clean_rollout.x)
        arrays[f"{fit.label}_clean_x_hat"] = np.asarray(fit.clean_rollout.x_hat)
        arrays[f"{fit.label}_clean_u"] = np.asarray(fit.clean_rollout.u)
        arrays[f"{fit.label}_under_eps_x"] = np.asarray(fit.under_epsilon_rollout.x)
        arrays[f"{fit.label}_under_eps_x_hat"] = np.asarray(fit.under_epsilon_rollout.x_hat)
        arrays[f"{fit.label}_under_eps_u"] = np.asarray(fit.under_epsilon_rollout.u)

    diagnostics = {
        "training_config": training_config.__dict__,
        "output_config": output_config.__dict__,
        "gamma": gamma_ref.gamma,
        "gamma_factor": gamma_ref.factor,
        "gamma_star": reference.gamma_star,
        "budget": budget,
        "budget_l2": float(jnp.sqrt(jnp.asarray(budget))),
        "rank_grid": list(ranks),
        "retained_fit_ranks": list(retained),
        "basis_family": "cardinal_cubic_b_spline",
        "eigenspectrum_coverage": eigenspectrum_coverage_by_label,
        "observer_error_coverage": observer_error_coverage_by_label,
        "initial_state_ensemble": _effective_rank_from_covariance(
            _weighted_covariance(states, weights)
        ),
        "analytical_exact_audits": {
            "lqr": {
                "cost": lqr_exact["cost"].total_without_disturbance_penalty,
                "lambda_over_gamma_squared": lqr_exact["gamma_penalized"][
                    "max_eigenvalue_over_gamma_squared"
                ],
                "gamma_penalized_feasible": lqr_exact["gamma_penalized"]["feasible"],
            },
            "hinf": {
                "cost": hinf_exact["cost"].total_without_disturbance_penalty,
                "lambda_over_gamma_squared": hinf_exact["gamma_penalized"][
                    "max_eigenvalue_over_gamma_squared"
                ],
                "gamma_penalized_feasible": hinf_exact["gamma_penalized"]["feasible"],
            },
        },
    }
    arrays.update(coverage_arrays)
    return TimeBasisResult(
        issue_id=ISSUE_ID,
        projections=tuple(projections),
        fits=tuple(fits),
        diagnostics=diagnostics,
        arrays=arrays,
    )


def result_summary(
    result: TimeBasisResult,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Return JSON-serializable summary."""

    return {
        "format": "rlrmp.output_feedback_time_constrained.v1",
        "issue": result.issue_id,
        "umbrella": UMBRELLA_ID,
        "source_issue": SOURCE_ISSUE_ID,
        "optimizer_bridge_issue": OPTIMIZER_BRIDGE_ISSUE_ID,
        "standard_certificate_issue": STANDARD_CERTIFICATE_ISSUE_ID,
        "failure_decomposition_issue": FAILURE_DECOMPOSITION_ISSUE_ID,
        "rerun_metadata": build_rerun_metadata(
            discretization=discretization,
            lane=lane,
            materializer="output_feedback_time_constrained",
        ),
        "scope": (
            "Smooth spline time-basis output-feedback bridge. Projection rows "
            "check representability; scratch rows test discovery; Bellman-projected "
            "rows are preservation anchors only."
        ),
        "non_goals": (
            "No GRU, linear recurrence, coverage/noise sweeps, robust training "
            "variants, or direct teacher-cloning claims."
        ),
        "diagnostics": result.diagnostics,
        "projections": [_projection_summary(row) for row in result.projections],
        "fits": [_fit_summary(row) for row in result.fits],
    }


def write_basic_outputs(
    *,
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    note_path: Path,
    manifest_path: Path,
    artifact_path: Path,
) -> None:
    """Write note, manifest, and bulk arrays for a completed run."""

    mkdir_p(note_path.parent)
    mkdir_p(manifest_path.parent)
    mkdir_p(artifact_path.parent)
    results_dir = mkdir_p(REPO_ROOT / "results" / ISSUE_ID)
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Smooth time-basis output-feedback bridge for the Phase 3 controller "
            "ladder. See `notes/output_feedback_time_constrained.md`.\n",
            encoding="utf-8",
        )
    np.savez_compressed(artifact_path, **arrays)
    summary["tracked_note"] = _repo_relative(note_path)
    summary["tracked_manifest"] = _repo_relative(manifest_path)
    summary["artifact_npz"] = _repo_relative(artifact_path)
    summary["artifact_npz_keys"] = sorted(arrays)
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    """Render tracked markdown for the smooth time-basis bridge."""

    projection_rows = [
        "| rank | label | projection residual | objective ratio | clean mismatch | exact L2 ratio | lambda/gamma^2 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["projections"]:
        projection_rows.append(
            "| "
            f"{row['rank']} | {row['label']} | "
            f"{row['relative_residual']:.8g} | "
            f"{row['objective_ratio_to_reference']:.8g} | "
            f"{row['clean_action_mismatch_ratio']:.8g} | "
            f"{row['exact_l2_cost_ratio_to_lqr']:.8g} | "
            f"{row['gamma_penalized_lambda_over_gamma_squared']:.8g} |"
        )
    fit_rows = [
        "| label | objective ratio | gain rel err | clean mismatch | under-eps ratio | exact L2 ratio | lambda/gamma^2 | iters | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["fits"]:
        fit_rows.append(
            "| "
            f"{row['label']} | "
            f"{row['objective_ratio_to_reference']:.8g} | "
            f"{row['gain_relative_error']:.8g} | "
            f"{row['clean_action_mismatch_ratio']:.8g} | "
            f"{row['under_epsilon_cost_ratio_to_lqr']:.8g} | "
            f"{row['exact_l2_cost_ratio_to_lqr']:.8g} | "
            f"{row['gamma_penalized_lambda_over_gamma_squared']:.8g} | "
            f"{row['n_iterations']} | "
            f"{row['optimizer_status']} |"
        )
    return f"""# Smooth Time-Basis Output-Feedback Bridge

Issue: `{summary["issue"]}`. Umbrella: `{summary["umbrella"]}`.
Source issue: `{summary["source_issue"]}`.

Scope: {summary["scope"]}

Non-goals: {summary["non_goals"]}

Runtime: `{summary.get("runtime_seconds", 0.0):.2f}` seconds.

Rank grid: `{summary["diagnostics"]["rank_grid"]}`.
Retained fit ranks: `{summary["diagnostics"]["retained_fit_ranks"]}`.

## Projection-Only Representability

{"\n".join(projection_rows)}

## Training Rows

{"\n".join(fit_rows)}
"""


def timed_run(**kwargs: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the bridge and return summary plus arrays with runtime metadata."""

    start = time.perf_counter()
    result = run_time_basis_bridge(**kwargs)
    summary = result_summary(result)
    summary["runtime_seconds"] = time.perf_counter() - start
    return summary, result.arrays


def _projection_row(
    *,
    rank: int,
    parameterization: TimeConstrainedGainParameterization,
    K_ref: Float[Array, "time action state"],
    reference_objective: float,
    plant: Any,
    schedule: Any,
    states: Float[Array, "batch state"],
    weights: Float[Array, " batch"],
    x0: Float[Array, " state"],
    budget: float,
    gamma: float,
    lqr_clean: OutputFeedbackRollout,
    lqr_exact_cost: float,
    output_config: OutputFeedbackConfig,
) -> TimeBasisProjection:
    projection = parameterization.project_gains(np.asarray(K_ref))
    K_projected = jnp.asarray(projection.reconstructed_gains, dtype=jnp.float64)
    clean = rollout_with_kalman_estimator(plant, K_projected, x0, config=output_config)
    audit = exact_output_feedback_adversary_audit(
        label=f"spline_r{rank}_projection",
        plant=plant,
        schedule=schedule,
        controller_gains=K_projected,
        x0=x0,
        budget=budget,
        estimator_kind="kalman",
        penalty_gamma=gamma,
        config=output_config,
    )
    objective = float(
        output_feedback_clean_objective(
            plant,
            schedule,
            K_projected,
            states,
            weights,
            output_config,
        )
    )
    return TimeBasisProjection(
        rank=rank,
        label=f"spline_r{rank}__projection",
        theta=jnp.asarray(projection.theta, dtype=jnp.float64),
        K=K_projected,
        residual_norm=projection.residual_norm,
        relative_residual=projection.relative_residual,
        objective_ratio_to_reference=objective / reference_objective,
        gain_relative_error=float(jnp.linalg.norm(K_projected - K_ref) / jnp.linalg.norm(K_ref)),
        clean_action_mismatch_ratio=_action_mismatch_ratio(
            clean.u,
            lqr_clean.u,
            floor=output_config.denominator_floor,
        ),
        exact_l2_cost_ratio_to_lqr=(
            audit["cost"].total_without_disturbance_penalty / lqr_exact_cost
        ),
        gamma_penalized_lambda_over_gamma_squared=float(
            audit["gamma_penalized"]["max_eigenvalue_over_gamma_squared"]
        ),
    )


def _fit_condition(
    *,
    condition: TimeBasisCondition,
    parameterization: TimeConstrainedGainParameterization,
    initial_K: Float[Array, "time action state"],
    K_ref: Float[Array, "time action state"],
    reference_objective: float,
    plant: Any,
    schedule: Any,
    states: Float[Array, "batch state"],
    weights: Float[Array, " batch"],
    state_scales: Float[Array, " state"],
    coverage_epsilons: Float[Array, "coverage time disturbance"],
    coverage_trajectory_weights: Float[Array, " coverage"],
    coverage_x: Float[Array, "coverage state"],
    coverage_xhat: Float[Array, "coverage state"],
    coverage_times: Float[Array, " coverage"],
    coverage_state_weights: Float[Array, " coverage"],
    x0: Float[Array, " state"],
    riccati_epsilon: Float[Array, "time disturbance"],
    budget: float,
    gamma: float,
    lqr_clean: OutputFeedbackRollout,
    lqr_under_eps: OutputFeedbackRollout,
    lqr_exact_cost: float,
    hinf_exact_cost: float,
    output_config: OutputFeedbackConfig,
) -> TimeBasisFit:
    basis = jnp.asarray(parameterization.basis, dtype=jnp.float64)
    scale = state_scales if condition.use_whitening else jnp.ones_like(state_scales)
    theta0_np = parameterization.project_gains(np.asarray(initial_K)).theta
    theta0 = np.asarray(jnp.asarray(theta0_np, dtype=jnp.float64) * scale[None, None, :])
    theta_shape = theta0.shape

    def to_K(theta_tree: Float[Array, "rank action state"]) -> Float[Array, "time action state"]:
        theta = theta_tree / scale[None, None, :]
        return jnp.einsum("tr,rai->tai", basis, theta)

    def coverage_objective(gains: Float[Array, "time action state"]) -> Float[Array, ""]:
        coverage = condition.eigenspectrum_coverage or condition.observer_error_coverage
        if coverage is None:
            return jnp.asarray(0.0, dtype=jnp.float64)
        if coverage.objective == "trajectory":
            return _coverage_trajectory_objective(
                plant,
                schedule,
                gains,
                x0,
                coverage_epsilons,
                coverage_trajectory_weights,
                output_config,
            )
        if coverage.objective == "state":
            return _coverage_state_objective(
                plant,
                schedule,
                gains,
                coverage_x,
                coverage_xhat,
                coverage_times,
                coverage_state_weights,
                output_config,
            )
        raise ValueError(f"Unknown coverage objective={coverage.objective!r}.")

    objective_reference = float(
        output_feedback_clean_objective(
            plant,
            schedule,
            K_ref,
            states,
            weights,
            output_config,
        )
        + coverage_objective(K_ref)
    )

    def objective_theta(theta_tree: Float[Array, "rank action state"]) -> Float[Array, ""]:
        gains = to_K(theta_tree)
        objective = output_feedback_clean_objective(
            plant,
            schedule,
            gains,
            states,
            weights,
            output_config,
        )
        if (
            condition.eigenspectrum_coverage is not None
            or condition.observer_error_coverage is not None
        ):
            objective = objective + coverage_objective(gains)
        return objective

    @jax.jit
    def value_and_grad_flat(theta: Float[Array, " flat"]) -> tuple[Float[Array, ""], Array]:
        theta_tree = theta.reshape(theta_shape)
        value, grad = jax.value_and_grad(objective_theta)(theta_tree)
        return value, grad.reshape(-1)

    @jax.jit
    def value_flat(theta: Float[Array, " flat"]) -> Float[Array, ""]:
        return objective_theta(theta.reshape(theta_shape))

    def scipy_value_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad_flat(jnp.asarray(theta, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    theta_current = theta0.reshape(-1)
    objective_initial = float(value_flat(jnp.asarray(theta_current, dtype=jnp.float64)))
    _initial_value, initial_grad = scipy_value_and_grad(theta_current)
    best_objective = objective_initial
    best_iteration: int | None = 0
    optimizer_success = True
    total_iterations = 0
    total_evaluations = 0
    messages: list[str] = []

    def run_lbfgsb(theta_start: np.ndarray, *, maxiter: int) -> tuple[np.ndarray, str, bool, int, int]:
        result = scipy_opt.minimize(
            scipy_value_and_grad,
            theta_start,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
        )
        return (
            np.asarray(result.x, dtype=np.float64),
            str(result.message),
            bool(result.success),
            int(result.nit),
            int(result.nfev),
        )

    if condition.optimizer == "lbfgsb":
        theta_current, message, success, nit, nfev = run_lbfgsb(
            theta_current,
            maxiter=condition.maxiter,
        )
        total_iterations += nit
        total_evaluations += nfev
        optimizer_success = success
        best_objective = float(value_flat(jnp.asarray(theta_current, dtype=jnp.float64)))
        best_iteration = total_iterations
        messages.append(message)
    else:
        transforms = []
        if condition.adam_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(condition.adam_clip_norm))
        transforms.append(optax.adamw(learning_rate=condition.learning_rate))
        optimizer = optax.chain(*transforms)

        @jax.jit
        def adam_stage(theta: Float[Array, " flat"], opt_state: optax.OptState):
            best_theta = theta
            best_value = value_flat(theta)
            best_step = jnp.asarray(0, dtype=jnp.int64)
            all_finite = jnp.asarray(True)

            def step(carry, step_index):
                theta, opt_state, best_theta, best_value, best_step, all_finite = carry
                value, grad = value_and_grad_flat(theta)
                updates, opt_state = optimizer.update(grad, opt_state, theta)
                theta = optax.apply_updates(theta, updates)
                clean_value = value_flat(theta)
                finite = jnp.isfinite(value) & jnp.isfinite(clean_value)
                finite = finite & jnp.all(jnp.isfinite(grad)) & jnp.all(jnp.isfinite(theta))
                improved = finite & (clean_value < best_value)
                best_theta = jnp.where(improved, theta, best_theta)
                best_value = jnp.where(improved, clean_value, best_value)
                best_step = jnp.where(improved, step_index + 1, best_step)
                all_finite = all_finite & finite
                return (theta, opt_state, best_theta, best_value, best_step, all_finite), None

            return jax.lax.scan(
                step,
                (theta, opt_state, best_theta, best_value, best_step, all_finite),
                jnp.arange(condition.maxiter),
            )[0]

        opt_state = optimizer.init(jnp.asarray(theta_current, dtype=jnp.float64))
        _theta, _opt_state, best_theta, best_value, best_step, all_finite = adam_stage(
            jnp.asarray(theta_current, dtype=jnp.float64),
            opt_state,
        )
        theta_current = np.asarray(best_theta, dtype=np.float64)
        best_objective = float(best_value)
        best_iteration = int(best_step)
        total_iterations += condition.maxiter
        total_evaluations += condition.maxiter
        optimizer_success = bool(all_finite)
        messages.append(
            f"AdamW completed {condition.maxiter} full-batch steps "
            f"(lr={condition.learning_rate:g}, clip={condition.adam_clip_norm})"
        )
        if condition.optimizer == "adamw_then_lbfgsb":
            polish_maxiter = condition.polish_maxiter or min(condition.maxiter, 500)
            polished, message, success, nit, nfev = run_lbfgsb(
                theta_current,
                maxiter=polish_maxiter,
            )
            total_iterations += nit
            total_evaluations += nfev
            optimizer_success = optimizer_success and success
            polished_objective = float(value_flat(jnp.asarray(polished, dtype=jnp.float64)))
            if polished_objective <= best_objective:
                theta_current = polished
                best_objective = polished_objective
                best_iteration = total_iterations
            messages.append(f"L-BFGS-B polish maxiter={polish_maxiter}: {message}")

    objective_final = float(value_flat(jnp.asarray(theta_current, dtype=jnp.float64)))
    objective_zero = float(value_flat(jnp.zeros_like(jnp.asarray(theta_current))))
    _value, final_grad = scipy_value_and_grad(theta_current)
    theta_final = jnp.asarray(theta_current, dtype=jnp.float64).reshape(theta_shape)
    K_final = to_K(theta_final)
    clean = rollout_with_kalman_estimator(plant, K_final, x0, config=output_config)
    under_eps = rollout_with_kalman_estimator(
        plant,
        K_final,
        x0,
        riccati_epsilon,
        config=output_config,
    )
    clean_cost = float(rollout_task_cost(schedule, clean.x, clean.u))
    under_epsilon_cost = output_feedback_cost(schedule, under_eps).total_without_disturbance_penalty
    audit = exact_output_feedback_adversary_audit(
        label=condition.label,
        plant=plant,
        schedule=schedule,
        controller_gains=K_final,
        x0=x0,
        budget=budget,
        estimator_kind="kalman",
        penalty_gamma=gamma,
        config=output_config,
    )
    return TimeBasisFit(
        condition=condition,
        theta=theta_final,
        K=K_final,
        objective_initial=objective_initial,
        objective_final=objective_final,
        objective_reference=objective_reference,
        objective_zero=objective_zero,
        objective_ratio_to_reference=objective_final / objective_reference,
        gain_relative_error=float(jnp.linalg.norm(K_final - K_ref) / jnp.linalg.norm(K_ref)),
        gradient_norm_initial=float(np.linalg.norm(initial_grad)),
        gradient_norm_final=float(np.linalg.norm(final_grad)),
        best_objective=best_objective,
        best_checkpoint_iteration=best_iteration,
        optimizer_status="; ".join(messages),
        optimizer_success=optimizer_success,
        n_iterations=total_iterations,
        n_function_evaluations=total_evaluations,
        clean_rollout=clean,
        clean_cost=clean_cost,
        clean_action_mismatch_ratio=_action_mismatch_ratio(
            clean.u,
            lqr_clean.u,
            floor=output_config.denominator_floor,
        ),
        under_epsilon_rollout=under_eps,
        under_epsilon_cost=under_epsilon_cost,
        under_epsilon_cost_ratio_to_lqr=(
            under_epsilon_cost
            / output_feedback_cost(
                schedule,
                lqr_under_eps,
            ).total_without_disturbance_penalty
        ),
        under_epsilon_action_mismatch_ratio=_action_mismatch_ratio(
            under_eps.u,
            lqr_under_eps.u,
            floor=output_config.denominator_floor,
        ),
        exact_l2_cost=audit["cost"].total_without_disturbance_penalty,
        exact_l2_cost_ratio_to_lqr=(
            audit["cost"].total_without_disturbance_penalty / lqr_exact_cost
        ),
        exact_l2_cost_ratio_to_hinf=(
            audit["cost"].total_without_disturbance_penalty / hinf_exact_cost
        ),
        gamma_penalized_feasible=bool(audit["gamma_penalized"]["feasible"]),
        gamma_penalized_lambda_over_gamma_squared=float(
            audit["gamma_penalized"]["max_eigenvalue_over_gamma_squared"]
        ),
    )


def _conditions_for_ranks(
    ranks: tuple[int, ...],
    *,
    adamw_lrs: tuple[float, ...],
    lbfgsb_maxiter: int,
    adamw_steps: int,
    polish_maxiter: int,
    adamw_clip_norm: float,
) -> tuple[TimeBasisCondition, ...]:
    rows = []
    for rank in ranks:
        rows.append(TimeBasisCondition(rank=rank, initialization="scratch", maxiter=lbfgsb_maxiter))
        for lr in adamw_lrs:
            rows.append(
                TimeBasisCondition(
                    rank=rank,
                    initialization="scratch",
                    optimizer="adamw",
                    learning_rate=lr,
                    maxiter=adamw_steps,
                    adam_clip_norm=adamw_clip_norm,
                )
            )
        best_lr = adamw_lrs[-1]
        rows.append(
            TimeBasisCondition(
                rank=rank,
                initialization="scratch",
                optimizer="adamw_then_lbfgsb",
                learning_rate=best_lr,
                maxiter=adamw_steps,
                polish_maxiter=polish_maxiter,
                adam_clip_norm=adamw_clip_norm,
            )
        )
        rows.append(
            TimeBasisCondition(
                rank=rank,
                initialization="bellman_projected",
                optimizer="adamw_then_lbfgsb",
                learning_rate=best_lr,
                maxiter=adamw_steps,
                polish_maxiter=polish_maxiter,
                adam_clip_norm=adamw_clip_norm,
            )
        )
    return tuple(rows)


def r12_state_eigenspectrum_coverage_conditions(
    *,
    modes: tuple[int, ...] = (1, 4),
    scales: tuple[float, ...] = (0.3, 1.0, 3.0),
    weight: float = 0.1,
    maxiter: int = 2000,
    learning_rate: float = 1e-2,
    polish_maxiter: int = 1000,
    adam_clip_norm: float = 1e4,
) -> tuple[TimeBasisCondition, ...]:
    """Return the planned r=12 state-eigenspectrum coverage rows."""

    return state_eigenspectrum_coverage_conditions(
        rank=12,
        modes=modes,
        scales=scales,
        weight=weight,
        maxiter=maxiter,
        learning_rate=learning_rate,
        polish_maxiter=polish_maxiter,
        adam_clip_norm=adam_clip_norm,
    )


def r12_observer_error_state_coverage_conditions(
    *,
    modes: tuple[int, ...] = (1,),
    scales: tuple[float, ...] = (0.3, 1.0),
    weight: float = 0.1,
    maxiter: int = 2000,
    learning_rate: float = 1e-2,
    polish_maxiter: int = 1000,
    adam_clip_norm: float = 1e4,
) -> tuple[TimeBasisCondition, ...]:
    """Return the planned r=12 observer-error state coverage rows."""

    return observer_error_state_coverage_conditions(
        rank=12,
        modes=modes,
        scales=scales,
        weight=weight,
        maxiter=maxiter,
        learning_rate=learning_rate,
        polish_maxiter=polish_maxiter,
        adam_clip_norm=adam_clip_norm,
    )


def state_eigenspectrum_coverage_conditions(
    *,
    rank: int,
    modes: tuple[int, ...],
    scales: tuple[float, ...],
    weight: float,
    maxiter: int,
    learning_rate: float,
    polish_maxiter: int,
    adam_clip_norm: float = 1e4,
) -> tuple[TimeBasisCondition, ...]:
    """Return state-eigenspectrum coverage rows for one spline rank."""

    return tuple(
        TimeBasisCondition(
            rank=rank,
            initialization="scratch",
            optimizer="adamw_then_lbfgsb",
            learning_rate=learning_rate,
            maxiter=maxiter,
            polish_maxiter=polish_maxiter,
            adam_clip_norm=adam_clip_norm,
            eigenspectrum_coverage=EigenspectrumCoverageConfig(
                n_modes=n_modes,
                scale=scale,
                weight=weight,
                objective="state",
            ),
        )
        for n_modes in modes
        for scale in scales
    )


def observer_error_state_coverage_conditions(
    *,
    rank: int,
    modes: tuple[int, ...],
    scales: tuple[float, ...],
    weight: float,
    maxiter: int,
    learning_rate: float,
    polish_maxiter: int,
    adam_clip_norm: float = 1e4,
) -> tuple[TimeBasisCondition, ...]:
    """Return observer-error state coverage rows for one spline rank."""

    return tuple(
        TimeBasisCondition(
            rank=rank,
            initialization="scratch",
            optimizer="adamw_then_lbfgsb",
            learning_rate=learning_rate,
            maxiter=maxiter,
            polish_maxiter=polish_maxiter,
            adam_clip_norm=adam_clip_norm,
            observer_error_coverage=ObserverErrorCoverageConfig(
                n_modes=n_modes,
                scale=scale,
                weight=weight,
                objective="state",
            ),
        )
        for n_modes in modes
        for scale in scales
    )


def r20_state_eigenspectrum_coverage_conditions(
    *,
    modes: tuple[int, ...] = (4,),
    scales: tuple[float, ...] = (1.0, 3.0),
    weight: float = 0.1,
    maxiter: int = 2000,
    learning_rate: float = 1e-2,
    polish_maxiter: int = 1000,
    adam_clip_norm: float = 1e4,
) -> tuple[TimeBasisCondition, ...]:
    """Return the focused r=20 state-eigenspectrum coverage rows."""

    return state_eigenspectrum_coverage_conditions(
        rank=20,
        modes=modes,
        scales=scales,
        weight=weight,
        maxiter=maxiter,
        learning_rate=learning_rate,
        polish_maxiter=polish_maxiter,
        adam_clip_norm=adam_clip_norm,
    )


def r20_observer_error_state_coverage_conditions(
    *,
    modes: tuple[int, ...] = (1,),
    scales: tuple[float, ...] = (0.3,),
    weight: float = 0.1,
    maxiter: int = 2000,
    learning_rate: float = 1e-2,
    polish_maxiter: int = 1000,
    adam_clip_norm: float = 1e4,
) -> tuple[TimeBasisCondition, ...]:
    """Return the focused r=20 observer-error state coverage rows."""

    return observer_error_state_coverage_conditions(
        rank=20,
        modes=modes,
        scales=scales,
        weight=weight,
        maxiter=maxiter,
        learning_rate=learning_rate,
        polish_maxiter=polish_maxiter,
        adam_clip_norm=adam_clip_norm,
    )


def r20_state_coverage_conditions() -> tuple[TimeBasisCondition, ...]:
    """Return the focused r=20 state-only coverage closure row set."""

    return (
        r20_state_eigenspectrum_coverage_conditions()
        + r20_observer_error_state_coverage_conditions()
    )


def r12_state_coverage_conditions() -> tuple[TimeBasisCondition, ...]:
    """Return the default r=12 state-only coverage follow-up row set."""

    return (
        r12_state_eigenspectrum_coverage_conditions()
        + r12_observer_error_state_coverage_conditions()
    )


def _validate_condition(condition: TimeBasisCondition) -> None:
    if (
        condition.eigenspectrum_coverage is not None
        and condition.observer_error_coverage is not None
    ):
        raise ValueError(
            f"Condition {condition.label!r} cannot combine eigenspectrum and "
            "observer-error coverage."
        )
    coverage = condition.eigenspectrum_coverage or condition.observer_error_coverage
    if coverage is not None and coverage.objective not in {"trajectory", "state"}:
        raise ValueError(f"Unknown coverage objective={coverage.objective!r}.")


def _default_retained_ranks(projections: list[TimeBasisProjection]) -> tuple[int, ...]:
    ranks = [
        row.rank
        for row in projections
        if row.relative_residual <= 0.25 or row.rank in {12, 20, max(p.rank for p in projections)}
    ]
    return tuple(dict.fromkeys(ranks))


def _projection_summary(row: TimeBasisProjection) -> dict[str, Any]:
    return {
        "rank": row.rank,
        "label": row.label,
        "residual_norm": row.residual_norm,
        "relative_residual": row.relative_residual,
        "objective_ratio_to_reference": row.objective_ratio_to_reference,
        "gain_relative_error": row.gain_relative_error,
        "clean_action_mismatch_ratio": row.clean_action_mismatch_ratio,
        "exact_l2_cost_ratio_to_lqr": row.exact_l2_cost_ratio_to_lqr,
        "gamma_penalized_lambda_over_gamma_squared": (
            row.gamma_penalized_lambda_over_gamma_squared
        ),
    }


def _fit_summary(row: TimeBasisFit) -> dict[str, Any]:
    condition = row.condition.__dict__.copy()
    condition.pop("eigenspectrum_coverage", None)
    condition.pop("observer_error_coverage", None)
    if row.condition.eigenspectrum_coverage is not None:
        condition["eigenspectrum_coverage"] = row.condition.eigenspectrum_coverage.__dict__
    if row.condition.observer_error_coverage is not None:
        condition["observer_error_coverage"] = row.condition.observer_error_coverage.__dict__
    return {
        "label": row.label,
        "condition": condition,
        "initialization": row.condition.initialization,
        "optimizer_status": row.optimizer_status,
        "optimizer_success": row.optimizer_success,
        "n_iterations": row.n_iterations,
        "n_function_evaluations": row.n_function_evaluations,
        "objective_initial": row.objective_initial,
        "objective_final": row.objective_final,
        "objective_reference": row.objective_reference,
        "objective_zero": row.objective_zero,
        "objective_ratio_to_reference": row.objective_ratio_to_reference,
        "gain_relative_error": row.gain_relative_error,
        "gradient_norm_initial": row.gradient_norm_initial,
        "gradient_norm_final": row.gradient_norm_final,
        "projected_gradient_norm_final": row.gradient_norm_final,
        "best_objective": row.best_objective,
        "best_checkpoint_iteration": row.best_checkpoint_iteration,
        "clean_cost": row.clean_cost,
        "clean_action_mismatch_ratio": row.clean_action_mismatch_ratio,
        "under_epsilon_cost": row.under_epsilon_cost,
        "under_epsilon_cost_ratio_to_lqr": row.under_epsilon_cost_ratio_to_lqr,
        "under_epsilon_action_mismatch_ratio": row.under_epsilon_action_mismatch_ratio,
        "exact_l2_cost": row.exact_l2_cost,
        "exact_l2_cost_ratio_to_lqr": row.exact_l2_cost_ratio_to_lqr,
        "exact_l2_cost_ratio_to_hinf": row.exact_l2_cost_ratio_to_hinf,
        "gamma_penalized_feasible": row.gamma_penalized_feasible,
        "gamma_penalized_lambda_over_gamma_squared": (
            row.gamma_penalized_lambda_over_gamma_squared
        ),
    }


def _weighted_covariance(
    samples: Float[Array, "batch state"],
    weights: Float[Array, " batch"],
) -> Float[Array, "state state"]:
    weights = weights.astype(jnp.float64)
    weights = weights / jnp.sum(weights)
    centered = samples - jnp.sum(samples * weights[:, None], axis=0)
    return (centered * weights[:, None]).T @ centered


def _float_label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.absolute().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_ADAMW_LRS",
    "ISSUE_ID",
    "SPLINE_RANKS",
    "TimeBasisCondition",
    "observer_error_state_coverage_conditions",
    "r12_observer_error_state_coverage_conditions",
    "r12_state_coverage_conditions",
    "r12_state_eigenspectrum_coverage_conditions",
    "r20_observer_error_state_coverage_conditions",
    "r20_state_coverage_conditions",
    "r20_state_eigenspectrum_coverage_conditions",
    "render_markdown",
    "result_summary",
    "run_time_basis_bridge",
    "state_eigenspectrum_coverage_conditions",
    "timed_run",
    "write_basic_outputs",
]
