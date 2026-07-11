"""LEGACY (frozen 2026-07-03, issues 64d5f13 and ef8e1df; relocated from src/).

This materializer is not contract-native: it predates the feedbax recipe,
bundle, and manifest contracts. It may not run without deliberate realignment.
Do not copy it as a pattern for new analyses. The port-or-delete decision is
deferred to the report-stage era (feedbax 132f98c) / publication.

Phase 3 released-code stochastic rollout-recovery evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float

from rlrmp.analysis.math.cs_game_card import (
    CostBreakdown,
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.cs_released_simulation import (
    CSNoiseCovariances,
    CSReleasedStochasticNoiseConfig,
    CSStochasticRollout,
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    simulate_lqg_released_forward,
)
from rlrmp.analysis.math.hinf_riccati import CostSchedule, PlantLinearization
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    exact_output_feedback_adversary_audit,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_estimator_fixed_adversary_policy,
    robust_output_feedback_gains,
    rollout_with_robust_estimator_policy,
)
from rlrmp.analysis.math.rerun_metadata import build_rerun_metadata
from rlrmp.analysis.math import require_jax_x64
from rlrmp.paths import REPO_ROOT, mkdir_p

ISSUE_ID = "dd232cd"
PHASE3_ISSUE_ID = "7a459bb"
UMBRELLA_ID = "43e8728"
DEFAULT_SOURCE_NPZ = (
    REPO_ROOT
    / "_artifacts"
    / PHASE3_ISSUE_ID
    / "output_feedback_rollout_recovery"
    / "output_feedback_rollout_recovery.npz"
)
DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT
    / "results"
    / PHASE3_ISSUE_ID
    / "notes"
    / "output_feedback_rollout_recovery_manifest.json"
)


@dataclass(frozen=True)
class Phase3StochasticConfig:
    """Monte Carlo settings for the released-code stochastic Phase 3 lane."""

    n_trials: int = 24
    seed: int = 2323
    noise_config: CSReleasedStochasticNoiseConfig = field(
        default_factory=lambda: DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG
    )

    @property
    def motor_covariance_scale(self) -> float:
        return self.noise_config.motor_covariance_scale

    @property
    def process_covariance_scale(self) -> float:
        return self.noise_config.process_covariance_scale

    @property
    def signal_dependent_scale(self) -> float:
        return self.noise_config.signal_dependent_scale


@dataclass(frozen=True)
class Phase3ControllerSpec:
    """One deterministic Phase 3 controller to evaluate stochastically."""

    label: str
    source: str
    K: Float[Array, "T m_u n"]
    deterministic_gain_relative_error: float | None = None
    deterministic_objective_ratio_to_reference: float | None = None


@dataclass(frozen=True)
class Phase3StochasticEvaluation:
    """Monte Carlo aggregate for one Phase 3 controller."""

    spec: Phase3ControllerSpec
    cost_mean: float
    cost_std: float
    cost_ratio_to_reference_mean: float
    cost_ratio_to_reference_std: float
    peak_forward_velocity_mean: float
    peak_forward_velocity_std: float
    terminal_error_mean: float
    terminal_error_std: float
    control_effort_mean: float
    control_effort_std: float
    action_mismatch_to_reference_mean: float
    action_mismatch_to_reference_std: float
    deterministic_exact_l2_cost: float
    deterministic_exact_l2_cost_ratio_to_lqr: float
    deterministic_exact_l2_cost_ratio_to_hinf: float
    deterministic_gamma_penalized_feasible: bool
    deterministic_lambda_over_gamma_squared: float


@dataclass(frozen=True)
class Phase3StochasticResult:
    """Complete released-code stochastic Phase 3 evaluation bundle."""

    issue_id: str
    phase3_issue_id: str
    config: Phase3StochasticConfig
    controllers: tuple[Phase3ControllerSpec, ...]
    evaluations: tuple[Phase3StochasticEvaluation, ...]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class Phase3ProcessNoiseSweepCell:
    """One released-stochastic process-noise scale cell."""

    label: str
    process_covariance_scale: float
    result: Phase3StochasticResult


@dataclass(frozen=True)
class Phase3ProcessNoiseSweepResult:
    """Released-stochastic Phase 3 process-noise scale sweep bundle."""

    issue_id: str
    phase3_issue_id: str
    base_config: Phase3StochasticConfig
    process_covariance_scales: tuple[float, ...]
    cells: tuple[Phase3ProcessNoiseSweepCell, ...]


def load_default_controller_specs(
    *,
    npz_path: Path = DEFAULT_SOURCE_NPZ,
    manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
) -> tuple[Phase3ControllerSpec, ...]:
    """Load deterministic Phase 3 gain arrays for stochastic evaluation."""

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing deterministic Phase 3 artifact: {npz_path}. "
            "Run scripts/materialize_output_feedback_rollout_recovery.py first."
        )
    arrays = np.load(npz_path)
    fit_metadata = _fit_metadata_by_label(manifest_path)

    def from_key(label: str, source: str, key: str) -> Phase3ControllerSpec:
        metadata = fit_metadata.get(label, {})
        return Phase3ControllerSpec(
            label=label,
            source=source,
            K=jnp.asarray(arrays[key], dtype=jnp.float64),
            deterministic_gain_relative_error=metadata.get("gain_relative_error"),
            deterministic_objective_ratio_to_reference=metadata.get("objective_ratio_to_reference"),
        )

    return (
        Phase3ControllerSpec(
            label="analytical_lqr_reference",
            source="analytical_lqr_reference",
            K=jnp.asarray(arrays["lqr_reference_K"], dtype=jnp.float64),
            deterministic_gain_relative_error=0.0,
            deterministic_objective_ratio_to_reference=1.0,
        ),
        from_key(
            "strong_optimizer_whitened__scratch",
            "deterministic_phase3_scratch_fit",
            "strong_optimizer_whitened__scratch_K",
        ),
        from_key(
            "strong_optimizer_whitened_block_time__scratch",
            "deterministic_phase3_scratch_fit",
            "strong_optimizer_whitened_block_time__scratch_K",
        ),
        from_key(
            "strong_optimizer_whitened__bellman_init",
            "deterministic_phase3_preservation_init_fit",
            "strong_optimizer_whitened__bellman_init_K",
        ),
        Phase3ControllerSpec(
            label="deterministic_bellman_initialization_raw",
            source="deterministic_phase3_initial_controller_only",
            K=jnp.asarray(arrays["bellman_initial_K"], dtype=jnp.float64),
            deterministic_gain_relative_error=None,
            deterministic_objective_ratio_to_reference=None,
        ),
    )


def run_phase3_process_noise_sweep(
    *,
    config: Phase3StochasticConfig = Phase3StochasticConfig(),
    process_covariance_scales: tuple[float, ...] = (0.0, 0.3, 1.0, 3.0),
    controllers: tuple[Phase3ControllerSpec, ...] | None = None,
    output_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Phase3ProcessNoiseSweepResult:
    """Evaluate Phase 3 controllers across released-stochastic process-noise scales."""

    require_jax_x64("C&S stochastic phase3 process-noise sweep")
    if not process_covariance_scales:
        raise ValueError("At least one process covariance scale is required.")

    controller_specs = load_default_controller_specs() if controllers is None else controllers
    cells = []
    normalized_scales = tuple(float(scale) for scale in process_covariance_scales)
    for scale in normalized_scales:
        cell_config = replace(
            config,
            noise_config=replace(config.noise_config, process_covariance_scale=scale),
        )
        result = run_phase3_stochastic_evaluation(
            config=cell_config,
            controllers=controller_specs,
            output_config=output_config,
        )
        cells.append(
            Phase3ProcessNoiseSweepCell(
                label=str(scale),
                process_covariance_scale=scale,
                result=result,
            )
        )

    return Phase3ProcessNoiseSweepResult(
        issue_id=ISSUE_ID,
        phase3_issue_id=PHASE3_ISSUE_ID,
        base_config=config,
        process_covariance_scales=normalized_scales,
        cells=tuple(cells),
    )


def run_phase3_stochastic_evaluation(
    *,
    config: Phase3StochasticConfig = Phase3StochasticConfig(),
    controllers: tuple[Phase3ControllerSpec, ...] | None = None,
    output_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Phase3StochasticResult:
    """Evaluate deterministic Phase 3 controllers with released stochastic noise."""

    require_jax_x64("C&S stochastic phase3 evaluation")
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    plant = reference.plant
    schedule = reference.schedule
    x0 = make_cs_output_feedback_initial_state(plant, output_config)
    controller_specs = load_default_controller_specs() if controllers is None else controllers
    if not controller_specs:
        raise ValueError("At least one controller spec is required.")

    covariances = _phase3_noise_covariances(plant, output_config, config)
    trial_keys = jr.split(jr.PRNGKey(config.seed), config.n_trials)
    reference_spec = controller_specs[0]
    reference_rollouts = [
        _simulate_one(plant, reference_spec.K, x0, key, covariances, output_config)
        for key in trial_keys
    ]
    reference_costs = jnp.asarray(
        [
            _stochastic_cost(schedule, rollout).total_without_disturbance_penalty
            for rollout in reference_rollouts
        ],
        dtype=jnp.float64,
    )
    reference_controls = [rollout.u_command for rollout in reference_rollouts]
    certificate_context = _deterministic_certificate_context(
        plant,
        schedule,
        gamma_ref.gamma,
        gamma_ref.solution,
        x0,
        output_config,
    )

    evaluations = []
    arrays: dict[str, np.ndarray] = {
        "reference_costs": np.asarray(reference_costs),
        "trial_keys": np.asarray(trial_keys),
    }
    for spec in controller_specs:
        rollouts = [
            _simulate_one(plant, spec.K, x0, key, covariances, output_config) for key in trial_keys
        ]
        costs = jnp.asarray(
            [
                _stochastic_cost(schedule, rollout).total_without_disturbance_penalty
                for rollout in rollouts
            ],
            dtype=jnp.float64,
        )
        ratios = costs / reference_costs
        peaks = jnp.asarray([rollout.peak_forward_velocity for rollout in rollouts])
        terminals = jnp.asarray([rollout.terminal_position_error for rollout in rollouts])
        efforts = jnp.asarray([rollout.control_effort for rollout in rollouts])
        mismatches = jnp.asarray(
            [
                _action_mismatch_ratio(rollout.u_command, reference_u)
                for rollout, reference_u in zip(rollouts, reference_controls, strict=True)
            ],
            dtype=jnp.float64,
        )
        certificate = _deterministic_certificate_for_controller(
            plant=plant,
            schedule=schedule,
            spec=spec,
            x0=x0,
            gamma=gamma_ref.gamma,
            budget=certificate_context["budget"],
            lqr_exact_cost=certificate_context["lqr_exact_cost"],
            hinf_exact_cost=certificate_context["hinf_exact_cost"],
            output_config=output_config,
        )
        evaluations.append(
            Phase3StochasticEvaluation(
                spec=spec,
                cost_mean=_mean(costs),
                cost_std=_std(costs),
                cost_ratio_to_reference_mean=_mean(ratios),
                cost_ratio_to_reference_std=_std(ratios),
                peak_forward_velocity_mean=_mean(peaks),
                peak_forward_velocity_std=_std(peaks),
                terminal_error_mean=_mean(terminals),
                terminal_error_std=_std(terminals),
                control_effort_mean=_mean(efforts),
                control_effort_std=_std(efforts),
                action_mismatch_to_reference_mean=_mean(mismatches),
                action_mismatch_to_reference_std=_std(mismatches),
                deterministic_exact_l2_cost=certificate["exact_l2_cost"],
                deterministic_exact_l2_cost_ratio_to_lqr=(
                    certificate["exact_l2_cost_ratio_to_lqr"]
                ),
                deterministic_exact_l2_cost_ratio_to_hinf=(
                    certificate["exact_l2_cost_ratio_to_hinf"]
                ),
                deterministic_gamma_penalized_feasible=certificate["gamma_penalized_feasible"],
                deterministic_lambda_over_gamma_squared=certificate["lambda_over_gamma_squared"],
            )
        )
        key = _safe_key(spec.label)
        arrays[f"{key}_costs"] = np.asarray(costs)
        arrays[f"{key}_cost_ratios"] = np.asarray(ratios)
        arrays[f"{key}_peak_forward_velocities"] = np.asarray(peaks)
        arrays[f"{key}_terminal_errors"] = np.asarray(terminals)
        arrays[f"{key}_control_efforts"] = np.asarray(efforts)
        arrays[f"{key}_action_mismatch_to_reference"] = np.asarray(mismatches)
        arrays[f"{key}_x"] = np.asarray(jnp.stack([rollout.x for rollout in rollouts]))
        arrays[f"{key}_x_hat"] = np.asarray(jnp.stack([rollout.x_hat for rollout in rollouts]))
        arrays[f"{key}_u_command"] = np.asarray(
            jnp.stack([rollout.u_command for rollout in rollouts])
        )
        arrays[f"{key}_u_applied"] = np.asarray(
            jnp.stack([rollout.u_applied for rollout in rollouts])
        )

    return Phase3StochasticResult(
        issue_id=ISSUE_ID,
        phase3_issue_id=PHASE3_ISSUE_ID,
        config=config,
        controllers=controller_specs,
        evaluations=tuple(evaluations),
        arrays=arrays,
    )


def process_noise_sweep_summary(result: Phase3ProcessNoiseSweepResult) -> dict[str, Any]:
    """Return a JSON-serializable process-noise sweep summary."""

    base_monte_carlo = _monte_carlo_summary(result.base_config)
    return {
        "issue": result.issue_id,
        "phase3_issue": result.phase3_issue_id,
        "umbrella": UMBRELLA_ID,
        "rerun_metadata": build_rerun_metadata(
            discretization="euler",
            lane="released_stochastic",
            materializer="cs_stochastic_phase3_process_noise_sweep",
        ),
        "base_monte_carlo": base_monte_carlo,
        "base_noise_contract": _noise_contract_summary(result.base_config.noise_config),
        "process_covariance_scales": list(result.process_covariance_scales),
        "scope": (
            "Released-code stochastic evaluation sweep over explicit process "
            "covariance scales. Controllers and common-random-number seeds are "
            "held fixed across cells; no stochastic training objective is added."
        ),
        "non_goals": (
            "No deterministic rollout-recovery objective change, no stochastic "
            "training, no robust Bellman objective, and no GRU evaluation."
        ),
        "cells": [
            {
                "label": cell.label,
                "process_covariance_scale": cell.process_covariance_scale,
                "monte_carlo": _monte_carlo_summary(cell.result.config),
                "noise_contract": _noise_contract_summary(cell.result.config.noise_config),
                "evaluations": [
                    _evaluation_summary(evaluation) for evaluation in cell.result.evaluations
                ],
                "verdict": _verdict(cell.result),
            }
            for cell in result.cells
        ],
    }


def _monte_carlo_summary(config: Phase3StochasticConfig) -> dict[str, int]:
    return {
        "n_trials": config.n_trials,
        "seed": config.seed,
    }


def _noise_contract_summary(config: CSReleasedStochasticNoiseConfig) -> dict[str, float | str]:
    return config.summary()


def result_summary(result: Phase3StochasticResult) -> dict[str, Any]:
    """Return a JSON-serializable released stochastic Phase 3 summary."""

    noise_contract = _noise_contract_summary(result.config.noise_config)
    return {
        "issue": result.issue_id,
        "phase3_issue": result.phase3_issue_id,
        "umbrella": UMBRELLA_ID,
        "rerun_metadata": build_rerun_metadata(
            discretization="euler",
            lane="released_stochastic",
            materializer="cs_stochastic_phase3",
        ),
        "source_artifacts": {
            "deterministic_phase3_npz": str(DEFAULT_SOURCE_NPZ.relative_to(REPO_ROOT)),
            "deterministic_phase3_manifest": str(DEFAULT_SOURCE_MANIFEST.relative_to(REPO_ROOT)),
        },
        "monte_carlo": _monte_carlo_summary(result.config),
        "noise_contract": noise_contract,
        "output_feedback_certificate_gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "claims": {
            "bellman_stochastic_parity": False,
            "extlqg_full_parity": True,
            "note": (
                "Deterministic init labels identify source controllers only; this "
                "lane evaluates released-code stochastic forward simulation and "
                "does not derive or claim a stochastic Bellman objective. The "
                "deterministic certificate columns are re-audits of the same "
                "controller gains, not stochastic induced-gain certificates."
            ),
        },
        "scope": (
            "Small common-random-number Monte Carlo evaluation of deterministic "
            "Phase 3 rollout-recovery controllers under Euler plus sampled sensory, "
            "input-image additive motor, signal-dependent input-image motor, and "
            "separate process/load noise."
        ),
        "non_goals": (
            "No initial-state jitter sweep, no stochastic training, and no "
            "stochastic Bellman parity claim."
        ),
        "evaluations": [_evaluation_summary(evaluation) for evaluation in result.evaluations],
        "verdict": _verdict(result),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the tracked released stochastic Phase 3 note."""

    rows = [
        "| controller | source | cost mean | cost std | cost ratio | "
        "peak v mean | terminal err mean | action mismatch | deterministic gain err | "
        "exact L2 ratio | lambda/gamma^2 | finite-gamma feasible |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["evaluations"]:
        rows.append(
            "| "
            f"{row['label']} | "
            f"{row['source']} | "
            f"{row['cost_mean']:.8g} | "
            f"{row['cost_std']:.8g} | "
            f"{row['cost_ratio_to_reference_mean']:.8g} +/- "
            f"{row['cost_ratio_to_reference_std']:.3g} | "
            f"{row['peak_forward_velocity_mean']:.8g} | "
            f"{row['terminal_error_mean']:.8g} | "
            f"{row['action_mismatch_to_reference_mean']:.8g} +/- "
            f"{row['action_mismatch_to_reference_std']:.3g} | "
            f"{_format_optional(row['deterministic_gain_relative_error'])} | "
            f"{row['deterministic_exact_l2_cost_ratio_to_lqr']:.8g} | "
            f"{row['deterministic_lambda_over_gamma_squared']:.8g} | "
            f"{row['deterministic_gamma_penalized_feasible']} |"
        )
    mc = summary["monte_carlo"]
    noise = summary["noise_contract"]
    return f"""# Phase 3 Released Stochastic Rollout Recovery

Issue: `{summary["issue"]}`. Phase 3 issue: `{summary["phase3_issue"]}`.
Umbrella: `{summary["umbrella"]}`.

Rerun metadata:

- Discretization: `{summary["rerun_metadata"]["discretization"]}`.
- Lane: `{summary["rerun_metadata"]["lane"]}`.
- Lane scope: {summary["rerun_metadata"]["lane_description"]}

Scope: {summary["scope"]}

Non-goals: {summary["non_goals"]}

Monte Carlo settings:

- Trials: `{mc["n_trials"]}`
- Seed: `{mc["seed"]}`
- Certificate gamma factor: `{summary["output_feedback_certificate_gamma_factor"]}`

Noise contract:

- Contract: `{noise["contract"]}`
- Additive motor covariance: `{noise["additive_motor_covariance"]}`
- Motor covariance scale: `{noise["motor_covariance_scale"]}`
- Process/load noise: `{noise["process_noise"]}`
- Process covariance scale: `{noise["process_covariance_scale"]}`
- Signal-dependent motor noise: `{noise["signal_dependent_motor_noise"]}`
- Signal-dependent scale: `{noise["signal_dependent_scale"]}`

Claims guardrail: {summary["claims"]["note"]}

Source artifacts:

- `{summary["source_artifacts"]["deterministic_phase3_npz"]}`
- `{summary["source_artifacts"]["deterministic_phase3_manifest"]}`

## Controller Matrix

{"\n".join(rows)}

## Verdict

{summary["verdict"]}
"""


def write_outputs(
    issue_id: str = ISSUE_ID,
    *,
    config: Phase3StochasticConfig = Phase3StochasticConfig(),
) -> dict[str, Any]:
    """Write tracked stochastic Phase 3 note/manifest and bulk metric arrays."""

    require_jax_x64("C&S stochastic phase3 materialization")
    result = run_phase3_stochastic_evaluation(config=config)
    summary = result_summary(result)
    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    artifact_dir = mkdir_p(REPO_ROOT / "_artifacts" / issue_id / "cs_stochastic_phase3")
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "C&S game-card fidelity audit and corrected Euler/stochastic lane outputs.\n",
            encoding="utf-8",
        )
    npz_path = artifact_dir / "cs_stochastic_phase3_metrics.npz"
    np.savez_compressed(npz_path, **result.arrays)
    summary["tracked_note"] = f"results/{issue_id}/notes/cs_stochastic_phase3.md"
    summary["tracked_manifest"] = f"results/{issue_id}/notes/cs_stochastic_phase3_manifest.json"
    summary["artifact_npz"] = f"_artifacts/{issue_id}/cs_stochastic_phase3/{npz_path.name}"
    summary["artifact_npz_keys"] = sorted(result.arrays.keys())
    note_path = notes_dir / "cs_stochastic_phase3.md"
    manifest_path = notes_dir / "cs_stochastic_phase3_manifest.json"
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _phase3_noise_covariances(
    plant: PlantLinearization,
    output_config: OutputFeedbackConfig,
    config: Phase3StochasticConfig,
) -> CSNoiseCovariances:
    return default_cs_noise_covariances(
        plant,
        output_config,
        noise_config=config.noise_config,
    )


def _simulate_one(
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    key: Array,
    covariances: CSNoiseCovariances,
    output_config: OutputFeedbackConfig,
) -> CSStochasticRollout:
    draws = sample_forward_noise_draws(key, T=K.shape[0], covariances=covariances)
    return simulate_lqg_released_forward(
        plant,
        K,
        x0,
        draws=draws,
        covariances=covariances,
        config=output_config,
    )


def _stochastic_cost(schedule: CostSchedule, rollout: CSStochasticRollout) -> CostBreakdown:
    x = rollout.x.astype(jnp.float64)
    u = rollout.u_applied.astype(jnp.float64)
    state_terms = jnp.einsum("ti,tij,tj->t", x[:-1], schedule.Q, x[:-1])
    control_terms = jnp.einsum("ti,tij,tj->t", u, schedule.R, u)
    terminal = x[-1] @ schedule.Q_f @ x[-1]
    state_stage = float(jnp.sum(state_terms))
    control_stage = float(jnp.sum(control_terms))
    terminal_state = float(terminal)
    total = state_stage + control_stage + terminal_state
    return CostBreakdown(
        state_stage=state_stage,
        control_stage=control_stage,
        terminal_state=terminal_state,
        total_without_disturbance_penalty=total,
        disturbance_energy=0.0,
        h_infinity_objective=None,
    )


def _action_mismatch_ratio(
    u: Float[Array, "T m_u"],
    reference_u: Float[Array, "T m_u"],
    *,
    floor: float = 1e-12,
) -> float:
    numerator = jnp.linalg.norm(u - reference_u)
    denominator = jnp.maximum(jnp.linalg.norm(reference_u), floor)
    return float(numerator / denominator)


def _evaluation_summary(evaluation: Phase3StochasticEvaluation) -> dict[str, Any]:
    spec = evaluation.spec
    return {
        "label": spec.label,
        "source": spec.source,
        "deterministic_gain_relative_error": spec.deterministic_gain_relative_error,
        "deterministic_objective_ratio_to_reference": (
            spec.deterministic_objective_ratio_to_reference
        ),
        "cost_mean": evaluation.cost_mean,
        "cost_std": evaluation.cost_std,
        "cost_ratio_to_reference_mean": evaluation.cost_ratio_to_reference_mean,
        "cost_ratio_to_reference_std": evaluation.cost_ratio_to_reference_std,
        "peak_forward_velocity_mean": evaluation.peak_forward_velocity_mean,
        "peak_forward_velocity_std": evaluation.peak_forward_velocity_std,
        "terminal_error_mean": evaluation.terminal_error_mean,
        "terminal_error_std": evaluation.terminal_error_std,
        "control_effort_mean": evaluation.control_effort_mean,
        "control_effort_std": evaluation.control_effort_std,
        "action_mismatch_to_reference_mean": (evaluation.action_mismatch_to_reference_mean),
        "action_mismatch_to_reference_std": evaluation.action_mismatch_to_reference_std,
        "deterministic_exact_l2_cost": evaluation.deterministic_exact_l2_cost,
        "deterministic_exact_l2_cost_ratio_to_lqr": (
            evaluation.deterministic_exact_l2_cost_ratio_to_lqr
        ),
        "deterministic_exact_l2_cost_ratio_to_hinf": (
            evaluation.deterministic_exact_l2_cost_ratio_to_hinf
        ),
        "deterministic_gamma_penalized_feasible": (
            evaluation.deterministic_gamma_penalized_feasible
        ),
        "deterministic_lambda_over_gamma_squared": (
            evaluation.deterministic_lambda_over_gamma_squared
        ),
    }


def _deterministic_certificate_context(
    plant: PlantLinearization,
    schedule: CostSchedule,
    gamma: float,
    solution,
    x0: Float[Array, " n"],
    output_config: OutputFeedbackConfig,
) -> dict[str, float]:
    """Return the shared deterministic certificate budget and reference costs."""

    covs = robust_estimator_covariances(plant, schedule, gamma, output_config)
    robust_gains = robust_output_feedback_gains(
        plant,
        schedule,
        solution,
        covs,
        output_config,
    )
    robust_policy = robust_estimator_fixed_adversary_policy(
        plant,
        schedule,
        solution,
        robust_gains,
        covs,
        output_config,
    )
    robust_rollout = rollout_with_robust_estimator_policy(
        plant,
        schedule,
        solution,
        x0,
        robust_policy,
        gains=robust_gains,
        config=output_config,
    )
    budget = float(jnp.sum(robust_rollout.epsilon**2))
    lqr_exact = exact_output_feedback_adversary_audit(
        label="analytical_lqr_kalman_for_stochastic_phase3",
        plant=plant,
        schedule=schedule,
        controller_gains=materialize_reference(
            gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,)
        ).lqr_solution.K,
        x0=x0,
        budget=budget,
        estimator_kind="kalman",
        penalty_gamma=gamma,
        config=output_config,
    )
    hinf_exact = exact_output_feedback_adversary_audit(
        label="analytical_hinf_robust_for_stochastic_phase3",
        plant=plant,
        schedule=schedule,
        controller_gains=robust_gains,
        x0=x0,
        budget=budget,
        estimator_kind="robust",
        solution=solution,
        penalty_gamma=gamma,
        config=output_config,
    )
    return {
        "budget": budget,
        "lqr_exact_cost": lqr_exact["cost"].total_without_disturbance_penalty,
        "hinf_exact_cost": hinf_exact["cost"].total_without_disturbance_penalty,
    }


def _deterministic_certificate_for_controller(
    *,
    plant: PlantLinearization,
    schedule: CostSchedule,
    spec: Phase3ControllerSpec,
    x0: Float[Array, " n"],
    gamma: float,
    budget: float,
    lqr_exact_cost: float,
    hinf_exact_cost: float,
    output_config: OutputFeedbackConfig,
) -> dict[str, float | bool]:
    """Run the deterministic exact-L2 and finite-gamma audit for one gain tensor."""

    audit = exact_output_feedback_adversary_audit(
        label=f"stochastic_phase3_{spec.label}",
        plant=plant,
        schedule=schedule,
        controller_gains=spec.K,
        x0=x0,
        budget=budget,
        estimator_kind="kalman",
        penalty_gamma=gamma,
        config=output_config,
    )
    exact_l2_cost = audit["cost"].total_without_disturbance_penalty
    return {
        "exact_l2_cost": exact_l2_cost,
        "exact_l2_cost_ratio_to_lqr": exact_l2_cost / lqr_exact_cost,
        "exact_l2_cost_ratio_to_hinf": exact_l2_cost / hinf_exact_cost,
        "gamma_penalized_feasible": bool(audit["gamma_penalized"]["feasible"]),
        "lambda_over_gamma_squared": float(
            audit["gamma_penalized"]["max_eigenvalue_over_gamma_squared"]
        ),
    }


def _fit_metadata_by_label(manifest_path: Path) -> dict[str, dict[str, float]]:
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        row["label"]: {
            "gain_relative_error": row.get("gain_relative_error"),
            "objective_ratio_to_reference": row.get("objective_ratio_to_reference"),
        }
        for row in manifest.get("fits", [])
    }


def _verdict(result: Phase3StochasticResult) -> str:
    rows = {evaluation.spec.label: evaluation for evaluation in result.evaluations}
    scratch_rows = [
        evaluation
        for evaluation in result.evaluations
        if evaluation.spec.source == "deterministic_phase3_scratch_fit"
    ]
    preservation_rows = [
        evaluation
        for evaluation in result.evaluations
        if evaluation.spec.source == "deterministic_phase3_preservation_init_fit"
    ]
    best_scratch = min(scratch_rows, key=lambda row: row.cost_ratio_to_reference_mean)
    best_preservation = min(
        preservation_rows,
        key=lambda row: row.cost_ratio_to_reference_mean,
    )
    reference = rows["analytical_lqr_reference"]
    lines = [
        "The released-stochastic evaluation keeps the deterministic Phase 3 "
        "interpretation: scratch-like fitted controllers remain behaviorally "
        "near in cost but still have substantial action mismatch relative to the "
        "analytical reference under the same sampled noise.",
        (
            f"Best scratch stochastic cost ratio is "
            f"{best_scratch.cost_ratio_to_reference_mean:.8g} "
            f"({best_scratch.spec.label}), with action mismatch "
            f"{best_scratch.action_mismatch_to_reference_mean:.8g}."
        ),
        (
            f"The preservation-init fit remains indistinguishable from the "
            f"reference at this Monte Carlo scale: cost ratio "
            f"{best_preservation.cost_ratio_to_reference_mean:.8g}, action "
            f"mismatch {best_preservation.action_mismatch_to_reference_mean:.8g}."
        ),
        (
            f"The analytical reference mean cost is {reference.cost_mean:.8g} "
            f"with peak forward velocity {reference.peak_forward_velocity_mean:.8g}."
        ),
        "This lane evaluates forward-simulation fidelity only; it does not add a "
        "stochastic Bellman objective or parity result.",
    ]
    return "\n".join(lines)


def _safe_key(label: str) -> str:
    return label.replace("/", "_").replace("-", "_")


def _mean(values: Float[Array, " n"]) -> float:
    return float(jnp.mean(values))


def _std(values: Float[Array, " n"]) -> float:
    return float(jnp.std(values, ddof=1)) if values.shape[0] > 1 else 0.0


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.8g}"


__all__ = [
    "ISSUE_ID",
    "PHASE3_ISSUE_ID",
    "Phase3ControllerSpec",
    "Phase3ProcessNoiseSweepCell",
    "Phase3ProcessNoiseSweepResult",
    "Phase3StochasticConfig",
    "Phase3StochasticEvaluation",
    "Phase3StochasticResult",
    "load_default_controller_specs",
    "process_noise_sweep_summary",
    "render_markdown",
    "result_summary",
    "run_phase3_process_noise_sweep",
    "run_phase3_stochastic_evaluation",
    "write_outputs",
]
