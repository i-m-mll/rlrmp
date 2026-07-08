"""LEGACY (frozen 2026-07-03, issue 64d5f13).

This materializer is not contract-native: it predates the feedbax recipe,
bundle, and manifest contracts. It may not run without deliberate realignment.
Do not copy it as a pattern for new analyses. The port-or-delete decision is
deferred to the report-stage era (feedbax 132f98c) / publication.

Interpolated-start probes for the free output-feedback bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.linear_round_trip import LinearTrainingConfig
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    exact_output_feedback_adversary_audit,
    make_cs_output_feedback_initial_state,
    output_feedback_lqr_bellman_objective,
    robust_estimator_covariances,
    robust_estimator_fixed_adversary_policy,
    robust_output_feedback_gains,
    rollout_with_kalman_estimator,
    rollout_with_robust_estimator_policy,
)
from rlrmp.analysis.pipelines.output_feedback_rollout_recovery import (
    STRONG_OPTIMIZER_WHITENED,
    RolloutRecoveryCondition,
    RolloutRecoveryFit,
    _empty_coverage_samples,
    _fit_one_condition,
    _fit_summary,
    _state_scales,
    _time_block_scales,
    _training_ensemble,
)
from rlrmp.analysis.math.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    build_rerun_metadata,
)
from rlrmp.analysis.math import require_jax_x64
from rlrmp.paths import REPO_ROOT, mkdir_p

ISSUE_ID = "7cea1b7"
SOURCE_ISSUE_ID = "7a459bb"
UMBRELLA_ID = "43e8728"
ADAMW_BRIDGE_ISSUE_ID = "1c014e5"
STANDARD_CERTIFICATE_ISSUE_ID = "d01c35a"
FAILURE_DECOMPOSITION_ISSUE_ID = "c45adde"
INTERPOLATED_ALPHAS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75)
DEFAULT_SOURCE_ARTIFACT = (
    REPO_ROOT
    / "_artifacts"
    / SOURCE_ISSUE_ID
    / "output_feedback_rollout_recovery"
    / "output_feedback_rollout_recovery.npz"
)
DEFAULT_CONDITION = replace(
    STRONG_OPTIMIZER_WHITENED,
    initializations=tuple(f"k_alpha_{str(alpha).replace('.', 'p')}" for alpha in INTERPOLATED_ALPHAS),
)


@dataclass(frozen=True)
class InterpolatedInitialization:
    """One K_alpha initialization for the free output-feedback bridge.

    Attributes:
        label: Stable manifest/array label, e.g. ``k_alpha_0p25``.
        alpha: Interpolation coefficient in ``K_alpha = (1 - alpha) K_scratch + alpha K_ref``.
        K: Gain tensor with shape ``(time, action, estimated_state)``.
    """

    label: str
    alpha: float
    K: Float[Array, "T m_u n"]


@dataclass(frozen=True)
class InterpolatedStartResult:
    """Complete interpolated-start probe bundle."""

    issue_id: str
    source_issue_id: str
    source_artifact: str
    condition: RolloutRecoveryCondition
    initializations: tuple[InterpolatedInitialization, ...]
    fits: tuple[RolloutRecoveryFit, ...]
    diagnostics: dict[str, Any]
    arrays: dict[str, np.ndarray]


def alpha_label(alpha: float) -> str:
    """Return the stable label suffix for an interpolation coefficient."""

    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    text = f"{alpha:g}".replace(".", "p")
    return f"k_alpha_{text}"


def build_interpolated_initializations(
    K_scratch: Float[Array, "T m_u n"],
    K_ref: Float[Array, "T m_u n"],
    *,
    alphas: tuple[float, ...] = INTERPOLATED_ALPHAS,
) -> tuple[InterpolatedInitialization, ...]:
    """Construct K_alpha starts between failed scratch and reference gains."""

    scratch = jnp.asarray(K_scratch, dtype=jnp.float64)
    reference = jnp.asarray(K_ref, dtype=jnp.float64)
    if scratch.shape != reference.shape:
        raise ValueError(
            f"K_scratch and K_ref must have the same shape, got {scratch.shape} and "
            f"{reference.shape}."
        )
    starts = []
    seen_labels = set()
    for alpha in alphas:
        alpha_value = float(alpha)
        if not 0.0 <= alpha_value <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha_value}.")
        label = alpha_label(alpha_value)
        if label in seen_labels:
            raise ValueError(f"duplicate interpolation label {label!r}.")
        seen_labels.add(label)
        K_alpha = (1.0 - alpha_value) * scratch + alpha_value * reference
        starts.append(InterpolatedInitialization(label=label, alpha=alpha_value, K=K_alpha))
    return tuple(starts)


def load_interpolated_initializations(
    source_artifact: Path = DEFAULT_SOURCE_ARTIFACT,
    *,
    scratch_key: str = "strong_optimizer_whitened__scratch_K",
    reference_key: str = "lqr_reference_K",
    alphas: tuple[float, ...] = INTERPOLATED_ALPHAS,
) -> tuple[InterpolatedInitialization, ...]:
    """Load scratch/reference gains from a rollout-recovery artifact and interpolate."""

    with np.load(source_artifact) as archive:
        missing = [key for key in (scratch_key, reference_key) if key not in archive.files]
        if missing:
            raise KeyError(
                f"{source_artifact} is missing required gain arrays: {', '.join(missing)}"
            )
        K_scratch = np.asarray(archive[scratch_key])
        K_ref = np.asarray(archive[reference_key])
    return build_interpolated_initializations(K_scratch, K_ref, alphas=alphas)


def run_interpolated_start_probe(
    *,
    source_artifact: Path = DEFAULT_SOURCE_ARTIFACT,
    condition: RolloutRecoveryCondition = DEFAULT_CONDITION,
    training_config: LinearTrainingConfig = LinearTrainingConfig(n_steps=500),
    output_config: OutputFeedbackConfig = OutputFeedbackConfig(),
    alphas: tuple[float, ...] = INTERPOLATED_ALPHAS,
) -> InterpolatedStartResult:
    """Run L-BFGS-B from K_alpha starts without changing the base recovery runner."""

    require_jax_x64("output-feedback interpolated-start probe")
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    plant = reference.plant
    schedule = reference.schedule
    K_ref = reference.lqr_solution.K
    initializations = load_interpolated_initializations(
        source_artifact,
        reference_key="lqr_reference_K",
        alphas=alphas,
    )
    expected_labels = tuple(start.label for start in initializations)
    condition = replace(condition, initializations=expected_labels)

    x0 = make_cs_output_feedback_initial_state(plant, output_config)
    states, weights = _training_ensemble(plant, training_config, output_config)
    scales = _state_scales(states, weights)
    time_block_scales = _time_block_scales(
        plant,
        schedule.T,
        states,
        weights,
        output_config,
    )
    bellman_p_next = reference.lqr_solution.P[1:].astype(jnp.float64)
    bellman_reference_objective = float(
        output_feedback_lqr_bellman_objective(
            plant,
            schedule,
            bellman_p_next,
            K_ref,
            states,
            weights,
        )
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
    (
        coverage_epsilons,
        coverage_trajectory_weights,
        coverage_x,
        coverage_xhat,
        coverage_times,
        coverage_state_weights,
    ) = _empty_coverage_samples(plant)

    fits = []
    for initialization in initializations:
        fits.append(
            _fit_one_condition(
                condition=condition,
                initialization=initialization.label,
                initial_K=initialization.K,
                plant=plant,
                schedule=schedule,
                K_ref=K_ref,
                states=states,
                weights=weights,
                coverage_epsilons=coverage_epsilons,
                coverage_trajectory_weights=coverage_trajectory_weights,
                coverage_x=coverage_x,
                coverage_xhat=coverage_xhat,
                coverage_times=coverage_times,
                coverage_state_weights=coverage_state_weights,
                state_scales=scales,
                time_block_scales=time_block_scales,
                bellman_p_next=bellman_p_next,
                bellman_reference_objective=bellman_reference_objective,
                x0=x0,
                riccati_epsilon=riccati_epsilon,
                budget=budget,
                gamma=gamma_ref.gamma,
                lqr_clean_rollout=lqr_clean,
                lqr_under_epsilon_rollout=lqr_under_eps,
                lqr_exact_cost=lqr_exact["cost"].total_without_disturbance_penalty,
                hinf_exact_cost=hinf_exact["cost"].total_without_disturbance_penalty,
                output_config=output_config,
            )
        )

    diagnostics = {
        "training_config": training_config.__dict__,
        "output_config": output_config.__dict__,
        "gamma": gamma_ref.gamma,
        "gamma_factor": gamma_ref.factor,
        "gamma_star": reference.gamma_star,
        "budget": budget,
        "budget_l2": float(jnp.sqrt(jnp.asarray(budget))),
        "source_artifact": _repo_relative(source_artifact),
        "scratch_key": "strong_optimizer_whitened__scratch_K",
        "reference_key": "lqr_reference_K",
        "alpha_grid": [start.alpha for start in initializations],
        "standard_certificate_inputs": (
            "Raw K, nominal-clean rollout arrays, and Riccati-epsilon rollout arrays "
            "are retained for every interpolated row."
        ),
        "failure_decomposition_inputs": (
            "The same raw arrays support objective/gradient, visited-subspace, and "
            "learned-to-reference interpolation diagnostics without retraining."
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
    arrays: dict[str, np.ndarray] = {
        "lqr_reference_K": np.asarray(K_ref),
        "lqr_clean_x": np.asarray(lqr_clean.x),
        "lqr_clean_x_hat": np.asarray(lqr_clean.x_hat),
        "lqr_clean_u": np.asarray(lqr_clean.u),
        "lqr_under_eps_x": np.asarray(lqr_under_eps.x),
        "lqr_under_eps_x_hat": np.asarray(lqr_under_eps.x_hat),
        "lqr_under_eps_u": np.asarray(lqr_under_eps.u),
        "riccati_epsilon": np.asarray(riccati_epsilon),
        "initial_states": np.asarray(states),
        "initial_state_weights": np.asarray(weights),
        "state_scales": np.asarray(scales),
        "time_block_scales": np.asarray(time_block_scales),
    }
    for initialization in initializations:
        arrays[f"{initialization.label}_initial_K"] = np.asarray(initialization.K)
    for fit in fits:
        arrays[f"{fit.label}_K"] = np.asarray(fit.K)
        arrays[f"{fit.label}_clean_x"] = np.asarray(fit.clean_rollout.x)
        arrays[f"{fit.label}_clean_x_hat"] = np.asarray(fit.clean_rollout.x_hat)
        arrays[f"{fit.label}_clean_u"] = np.asarray(fit.clean_rollout.u)
        arrays[f"{fit.label}_under_eps_x"] = np.asarray(fit.under_epsilon_rollout.x)
        arrays[f"{fit.label}_under_eps_x_hat"] = np.asarray(fit.under_epsilon_rollout.x_hat)
        arrays[f"{fit.label}_under_eps_u"] = np.asarray(fit.under_epsilon_rollout.u)

    return InterpolatedStartResult(
        issue_id=ISSUE_ID,
        source_issue_id=SOURCE_ISSUE_ID,
        source_artifact=_repo_relative(source_artifact),
        condition=condition,
        initializations=initializations,
        fits=tuple(fits),
        diagnostics=diagnostics,
        arrays=arrays,
    )


def result_summary(
    result: InterpolatedStartResult,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Return JSON-serializable interpolated-start summary."""

    condition = result.condition.__dict__.copy()
    optimizer = condition.get("optimizer", "lbfgsb")
    return {
        "issue": result.issue_id,
        "source_issue": result.source_issue_id,
        "umbrella": UMBRELLA_ID,
        "rerun_metadata": build_rerun_metadata(
            discretization=discretization,
            lane=lane,
            materializer="output_feedback_interpolated_starts",
        ),
        "related_issues": {
            "adamw_bridge": ADAMW_BRIDGE_ISSUE_ID,
            "standard_certificate": STANDARD_CERTIFICATE_ISSUE_ID,
            "failure_decomposition": FAILURE_DECOMPOSITION_ISSUE_ID,
        },
        "scope": (
            "Basin-access diagnostic for the free output-feedback bridge: run "
            f"{optimizer} from K_alpha = (1-alpha) K_scratch + alpha K_ref under "
            "the deterministic no-coverage clean-rollout objective and current "
            "whitened scaling."
        ),
        "non_goals": (
            "No coverage/noise/GRU/basis-constrained runs and no standard-certificate "
            "schema changes."
        ),
        "source_artifact": result.source_artifact,
        "condition": condition,
        "initializations": [
            {"label": start.label, "alpha": start.alpha} for start in result.initializations
        ],
        "diagnostics": result.diagnostics,
        "fits": [_fit_summary(fit) for fit in result.fits],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the tracked interpolated-start note."""

    rows = [
        "| alpha | label | objective ratio | gain rel err | clean action mismatch | "
        "under-epsilon ratio | exact L2 ratio | lambda/gamma^2 | iters | status |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    alpha_by_label = {row["label"]: row["alpha"] for row in summary["initializations"]}
    for row in summary["fits"]:
        alpha = alpha_by_label[row["initialization"]]
        rows.append(
            "| "
            f"{alpha:g} | "
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
    return f"""# Interpolated Starts for the Free Output-Feedback Bridge

Issue: `{summary["issue"]}`. Source issue: `{summary["source_issue"]}`.
Umbrella: `{summary["umbrella"]}`.

Scope: {summary["scope"]}

Non-goals: {summary["non_goals"]}

Source artifact: `{summary["source_artifact"]}`.

Alpha grid: `{summary["diagnostics"]["alpha_grid"]}`.

The saved artifact retains the fitted gains plus nominal-clean and
Riccati-epsilon rollout arrays for each row, so the standard certificate and
failure-decomposition companion can be materialized without rerunning these
optimizer fits.

## Run Matrix

{"\n".join(rows)}
"""


def write_outputs(
    issue_id: str = ISSUE_ID,
    *,
    source_artifact: Path = DEFAULT_SOURCE_ARTIFACT,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Write tracked interpolated-start note/manifest and bulk arrays."""

    require_jax_x64("output-feedback interpolated-start materialization")
    result = run_interpolated_start_probe(source_artifact=source_artifact)
    summary = result_summary(result, discretization=discretization, lane=lane)
    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    artifact_dir = mkdir_p(REPO_ROOT / "_artifacts" / issue_id / "output_feedback_interpolated_starts")
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Interpolated-start basin diagnostics for the Phase 3 free "
            "output-feedback bridge. See `notes/output_feedback_interpolated_starts.md`.\n",
            encoding="utf-8",
        )
    npz_path = artifact_dir / "output_feedback_interpolated_starts.npz"
    np.savez_compressed(npz_path, **result.arrays)
    summary["tracked_note"] = f"results/{issue_id}/notes/output_feedback_interpolated_starts.md"
    summary["tracked_manifest"] = (
        f"results/{issue_id}/notes/output_feedback_interpolated_starts_manifest.json"
    )
    summary["artifact_npz"] = (
        f"_artifacts/{issue_id}/output_feedback_interpolated_starts/{npz_path.name}"
    )
    summary["artifact_npz_keys"] = sorted(result.arrays.keys())
    note_path = notes_dir / "output_feedback_interpolated_starts.md"
    manifest_path = notes_dir / "output_feedback_interpolated_starts_manifest.json"
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_CONDITION",
    "DEFAULT_SOURCE_ARTIFACT",
    "INTERPOLATED_ALPHAS",
    "InterpolatedInitialization",
    "InterpolatedStartResult",
    "alpha_label",
    "build_interpolated_initializations",
    "load_interpolated_initializations",
    "render_markdown",
    "result_summary",
    "run_interpolated_start_probe",
    "write_outputs",
]
