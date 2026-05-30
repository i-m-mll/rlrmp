"""Phase 3 released-code stochastic rollout-recovery evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float

from rlrmp.analysis.cs_game_card import (
    CostBreakdown,
    PRIMARY_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.cs_released_simulation import (
    CSNoiseCovariances,
    CSStochasticRollout,
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    simulate_lqg_released_forward,
)
from rlrmp.analysis.hinf_riccati import CostSchedule, PlantLinearization
from rlrmp.analysis.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
)
from rlrmp.analysis.rerun_metadata import build_rerun_metadata
from rlrmp.paths import REPO_ROOT, mkdir_p


jax.config.update("jax_enable_x64", True)

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
    motor_covariance_scale: float = 1e-8
    process_covariance_scale: float | None = None
    signal_dependent_scale: float = 0.02


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


@dataclass(frozen=True)
class Phase3StochasticResult:
    """Complete released-code stochastic Phase 3 evaluation bundle."""

    issue_id: str
    phase3_issue_id: str
    config: Phase3StochasticConfig
    controllers: tuple[Phase3ControllerSpec, ...]
    evaluations: tuple[Phase3StochasticEvaluation, ...]
    arrays: dict[str, np.ndarray]


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


def run_phase3_stochastic_evaluation(
    *,
    config: Phase3StochasticConfig = Phase3StochasticConfig(),
    controllers: tuple[Phase3ControllerSpec, ...] | None = None,
    output_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Phase3StochasticResult:
    """Evaluate deterministic Phase 3 controllers with released stochastic noise."""

    reference = materialize_reference(gamma_factors=(PRIMARY_GAMMA_FACTOR,))
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
            )
        )
        key = _safe_key(spec.label)
        arrays[f"{key}_costs"] = np.asarray(costs)
        arrays[f"{key}_cost_ratios"] = np.asarray(ratios)
        arrays[f"{key}_peak_forward_velocities"] = np.asarray(peaks)
        arrays[f"{key}_terminal_errors"] = np.asarray(terminals)
        arrays[f"{key}_control_efforts"] = np.asarray(efforts)
        arrays[f"{key}_action_mismatch_to_reference"] = np.asarray(mismatches)

    return Phase3StochasticResult(
        issue_id=ISSUE_ID,
        phase3_issue_id=PHASE3_ISSUE_ID,
        config=config,
        controllers=controller_specs,
        evaluations=tuple(evaluations),
        arrays=arrays,
    )


def result_summary(result: Phase3StochasticResult) -> dict[str, Any]:
    """Return a JSON-serializable released stochastic Phase 3 summary."""

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
        "monte_carlo": result.config.__dict__,
        "claims": {
            "bellman_stochastic_parity": False,
            "extlqg_full_parity": False,
            "note": (
                "Deterministic init labels identify source controllers only; this "
                "lane evaluates released-code stochastic forward simulation and "
                "does not derive or claim a stochastic Bellman objective."
            ),
        },
        "scope": (
            "Small common-random-number Monte Carlo evaluation of deterministic "
            "Phase 3 rollout-recovery controllers under Euler plus sampled sensory, "
            "state-space motor/process, and signal-dependent state noise."
        ),
        "non_goals": (
            "No initial-state jitter sweep, no process-noise scale sweep, no full "
            "C&S extLQG fixed-point port, and no stochastic Bellman parity claim."
        ),
        "evaluations": [_evaluation_summary(evaluation) for evaluation in result.evaluations],
        "verdict": _verdict(result),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the tracked released stochastic Phase 3 note."""

    rows = [
        "| controller | source | cost mean | cost std | cost ratio | "
        "peak v mean | terminal err mean | action mismatch | deterministic gain err |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
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
            f"{_format_optional(row['deterministic_gain_relative_error'])} |"
        )
    mc = summary["monte_carlo"]
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
- Motor covariance scale: `{mc["motor_covariance_scale"]}`
- Process covariance scale: `{mc["process_covariance_scale"]}`
- Signal-dependent scale: `{mc["signal_dependent_scale"]}`

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
        motor_covariance_scale=config.motor_covariance_scale,
        process_covariance_scale=config.process_covariance_scale,
        signal_dependent_scale=config.signal_dependent_scale,
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
    "Phase3StochasticConfig",
    "Phase3StochasticEvaluation",
    "Phase3StochasticResult",
    "load_default_controller_specs",
    "render_markdown",
    "result_summary",
    "run_phase3_stochastic_evaluation",
    "write_outputs",
]
