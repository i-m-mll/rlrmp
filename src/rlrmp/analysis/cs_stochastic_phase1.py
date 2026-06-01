"""Phase 1 released-code stochastic evaluation for the C&S game card.

This lane evaluates the Euler C&S game card under sampled released-code-style
forward noise. It is intentionally a Monte Carlo evaluation/materialization
lane, not a Bellman or full extLQG parity claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float

from rlrmp.analysis.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    PRIMARY_GAMMA_FACTOR,
    TARGET_POS,
    materialize_reference,
    reference_summary,
)
from rlrmp.analysis.cs_released_simulation import (
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    CSForwardNoiseDraws,
    CSNoiseCovariances,
    CSReleasedStochasticNoiseConfig,
    CSStochasticRollout,
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    simulate_lqg_released_forward,
    simulate_robust_released_forward,
)
from rlrmp.analysis.hinf_riccati import CostSchedule, PlantLinearization
from rlrmp.analysis.output_feedback import (
    OutputFeedbackConfig,
    exact_output_feedback_adversary_audit,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_estimator_fixed_adversary_policy,
    robust_output_feedback_gains,
    rollout_with_robust_estimator_policy,
)
from rlrmp.analysis.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    build_rerun_metadata,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE_ID = "dd232cd"
UMBRELLA_ID = "43e8728"
DETERMINISTIC_PHASE1_ISSUE_ID = "a7dad8a"
LANE = "released_stochastic"
DEFAULT_SEEDS = tuple(range(12))
MOTOR_COVARIANCE_SCALE = DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG.motor_covariance_scale
PROCESS_COVARIANCE_SCALE = DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG.process_covariance_scale
SIGNAL_DEPENDENT_SCALE = DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG.signal_dependent_scale


@dataclass(frozen=True)
class CSFullStateStochasticRollout:
    """Full-state controller rollout under released-code-style plant noise."""

    x: Float[Array, "T_plus_1 n"]
    u_command: Float[Array, "T m_u"]
    u_applied: Float[Array, "T m_u"]
    motor_noise: Float[Array, "T n"]
    signal_dependent_standard: Float[Array, "T m_u"]
    signal_dependent_noise: Float[Array, "T n"]
    process_noise: Float[Array, "T n"]
    adversary_epsilon: Float[Array, "T m_w"]
    peak_forward_velocity: float
    peak_forward_velocity_idx: int
    terminal_position_error: float
    control_effort: float


@dataclass(frozen=True)
class Phase1StochasticTrial:
    """One shared-noise Monte Carlo trial across all Phase 1 arms."""

    seed: int
    draws: CSForwardNoiseDraws
    full_state_lqr: CSFullStateStochasticRollout
    full_state_hinf: CSFullStateStochasticRollout
    output_feedback_lqg: CSStochasticRollout
    output_feedback_hinf: CSStochasticRollout


@dataclass(frozen=True)
class Phase1StochasticResult:
    """Materialized Phase 1 stochastic-lane evaluation."""

    seeds: tuple[int, ...]
    noise_config: CSReleasedStochasticNoiseConfig
    covariances: CSNoiseCovariances
    trials: tuple[Phase1StochasticTrial, ...]
    extlqg_parity_status: str
    extlqg_n_iterations: int | None
    extlqg_expected_cost: float | None
    robust_gamma_factor: float
    robust_gamma: float
    deterministic_certificate_sidecar: dict[str, dict[str, float | bool | str]]


def simulate_full_state_released_forward(
    plant: PlantLinearization,
    controller_gains: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    *,
    draws: CSForwardNoiseDraws,
    covariances: CSNoiseCovariances,
    adversary_epsilon: Float[Array, "T m_w"] | None = None,
) -> CSFullStateStochasticRollout:
    """Roll ``u_t = -K_t x_t`` through the stochastic released-code plant."""

    T = int(controller_gains.shape[0])
    eps = (
        jnp.zeros((T, plant.m_w), dtype=jnp.float64)
        if adversary_epsilon is None
        else adversary_epsilon.astype(jnp.float64)
    )
    x_seq = [x0.astype(jnp.float64)]
    u_seq = []
    signal_dependent_seq = []
    for t in range(T):
        x_t = x_seq[-1]
        u_t = -controller_gains[t] @ x_t
        signal_dependent = jnp.einsum(
            "j,nmj,m->n",
            draws.signal_dependent_standard[t],
            covariances.signal_dependent_state,
            u_t,
        )
        x_next = (
            plant.A @ x_t
            + plant.B @ u_t
            + plant.Bw @ eps[t]
            + draws.motor[t]
            + signal_dependent
            + draws.process[t]
        )
        u_seq.append(u_t)
        signal_dependent_seq.append(signal_dependent)
        x_seq.append(x_next)

    x = jnp.stack(x_seq, axis=0)
    u = jnp.stack(u_seq, axis=0)
    peak, peak_idx, terminal, effort = _summary_fields(plant, x, u)
    return CSFullStateStochasticRollout(
        x=x,
        u_command=u,
        u_applied=u,
        motor_noise=draws.motor,
        signal_dependent_standard=draws.signal_dependent_standard,
        signal_dependent_noise=jnp.stack(signal_dependent_seq, axis=0),
        process_noise=draws.process,
        adversary_epsilon=eps,
        peak_forward_velocity=peak,
        peak_forward_velocity_idx=peak_idx,
        terminal_position_error=terminal,
        control_effort=effort,
    )


def analyze_phase1_stochastic(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    gamma_factor: float = OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    noise_config: CSReleasedStochasticNoiseConfig = DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> Phase1StochasticResult:
    """Evaluate Phase 1 controller families in the released stochastic lane."""

    reference = materialize_reference(gamma_factors=(gamma_factor,))
    gamma_ref = reference.gamma_references[0]
    plant = reference.plant
    schedule = reference.schedule
    x0 = make_cs_output_feedback_initial_state(plant, config)
    covariances = default_cs_noise_covariances(
        plant,
        config,
        noise_config=noise_config,
    )
    comparator = _build_extlqg_comparator_path(
        plant,
        reference.lqr_solution.K,
        covariances,
        schedule=schedule,
        config=config,
    )
    estimator_covariances = robust_estimator_covariances(
        plant,
        schedule,
        gamma_ref.gamma,
        config,
    )
    robust_gains = robust_output_feedback_gains(
        plant,
        schedule,
        gamma_ref.solution,
        estimator_covariances,
        config,
    )
    certificate_sidecar = _deterministic_certificate_sidecar(
        plant=plant,
        schedule=schedule,
        x0=x0,
        gamma=gamma_ref.gamma,
        lqr_value_matrices=reference.lqr_solution.P,
        hinf_solution=gamma_ref.solution,
        extlqg_gains=comparator.controller_gains,
        output_feedback_hinf_gains=robust_gains,
        config=config,
    )

    trials = []
    for seed in seeds:
        draws = sample_forward_noise_draws(
            jr.PRNGKey(int(seed)),
            T=schedule.T,
            covariances=covariances,
        )
        trials.append(
            Phase1StochasticTrial(
                seed=int(seed),
                draws=draws,
                full_state_lqr=simulate_full_state_released_forward(
                    plant,
                    reference.lqr_solution.K,
                    x0,
                    draws=draws,
                    covariances=covariances,
                ),
                full_state_hinf=simulate_full_state_released_forward(
                    plant,
                    gamma_ref.solution.K,
                    x0,
                    draws=draws,
                    covariances=covariances,
                ),
                output_feedback_lqg=simulate_lqg_released_forward(
                    plant,
                    comparator.controller_gains,
                    x0,
                    draws=draws,
                    covariances=covariances,
                    estimator_gains=comparator.estimator_gains,
                    config=config,
                ),
                output_feedback_hinf=simulate_robust_released_forward(
                    plant,
                    schedule,
                    gamma_ref.solution,
                    x0,
                    draws=draws,
                    covariances=covariances,
                    gains=robust_gains,
                    config=config,
                ),
            )
        )
    return Phase1StochasticResult(
        seeds=tuple(int(seed) for seed in seeds),
        noise_config=noise_config,
        covariances=covariances,
        trials=tuple(trials),
        extlqg_parity_status=comparator.parity_status,
        extlqg_n_iterations=getattr(comparator, "n_iterations", None),
        extlqg_expected_cost=(
            None
            if getattr(comparator, "expected_cost", None) is None
            else float(getattr(comparator, "expected_cost"))
        ),
        robust_gamma_factor=gamma_ref.factor,
        robust_gamma=gamma_ref.gamma,
        deterministic_certificate_sidecar=certificate_sidecar,
    )


def result_summary(
    result: Phase1StochasticResult,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
) -> dict[str, Any]:
    """Return a JSON-serializable Phase 1 stochastic-lane summary."""

    schedule = materialize_reference(
        gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,)
    ).schedule
    arms = {
        "full_state_lqr": {
            "controller_family": "LQR",
            "information_structure": "full_state",
            "comparator_status": "exact deterministic LQR gains under sampled stochastic plant",
            "rollouts": [trial.full_state_lqr for trial in result.trials],
        },
        "full_state_hinf": {
            "controller_family": "H-infinity",
            "information_structure": "full_state",
            "comparator_status": "exact deterministic H-infinity gains under sampled stochastic plant",
            "rollouts": [trial.full_state_hinf for trial in result.trials],
        },
        "output_feedback_lqg_extlqg": {
            "controller_family": "LQG",
            "information_structure": "output_feedback",
            "comparator_status": result.extlqg_parity_status,
            "rollouts": [trial.output_feedback_lqg for trial in result.trials],
        },
        "output_feedback_hinf": {
            "controller_family": "H-infinity",
            "information_structure": "output_feedback",
            "comparator_status": "C&S-style robust output-feedback gains under sampled stochastic plant",
            "rollouts": [trial.output_feedback_hinf for trial in result.trials],
        },
    }
    noise_contract = {
        **result.noise_config.summary(),
        "sensory_covariance_shape": list(result.covariances.sensory.shape),
        "motor_covariance_shape": list(result.covariances.motor.shape),
        "process_covariance_shape": list(result.covariances.process.shape),
        "signal_dependent_state_shape": list(result.covariances.signal_dependent_state.shape),
        "shared_noise_policy": (
            "Each seed samples one draw bundle and reuses it for full-state LQR, "
            "full-state H-infinity, output-feedback extLQG, and "
            "output-feedback H-infinity arms."
        ),
    }
    return {
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "deterministic_phase1_issue": DETERMINISTIC_PHASE1_ISSUE_ID,
        "rerun_metadata": build_rerun_metadata(
            discretization=discretization,
            lane=LANE,
            materializer="cs_stochastic_phase1",
        ),
        "no_bellman_claim": True,
        "bellman_claim": "none; stochastic Bellman parity is explicitly out of scope",
        "primary_gamma_factor": PRIMARY_GAMMA_FACTOR,
        "output_feedback_certificate_gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "robust_gamma_factor": result.robust_gamma_factor,
        "robust_gamma": result.robust_gamma,
        "n_trials": len(result.trials),
        "seeds": list(result.seeds),
        "noise_contract": noise_contract,
        "deterministic_certificate_sidecar": result.deterministic_certificate_sidecar,
        "extlqg_comparator": {
            "label": "local_extlqg_fixed_point",
            "matlab_function_chain": ["extLQG", "computeOFC", "computeExtKalman"],
            "parity_status": result.extlqg_parity_status,
            "n_iterations": result.extlqg_n_iterations,
            "expected_cost": result.extlqg_expected_cost,
        },
        "arms": {
            label: _arm_summary(
                rollouts=spec["rollouts"],
                schedule=schedule,
                controller_family=spec["controller_family"],
                information_structure=spec["information_structure"],
                comparator_status=spec["comparator_status"],
            )
            for label, spec in arms.items()
        },
        "trial_metrics": {
            label: [_rollout_metric_dict(rollout, schedule) for rollout in spec["rollouts"]]
            for label, spec in arms.items()
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a tracked Phase 1 stochastic-lane note."""

    rows = [
        "| Arm | Structure | Comparator status | Mean cost | Cost std | Peak v mean | Terminal error mean | Estimator RMS mean |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, arm in summary["arms"].items():
        estimator = arm["estimator_rms_error"]["mean"]
        estimator_text = "n/a" if estimator is None else f"{estimator:.8g}"
        rows.append(
            "| "
            f"`{label}` | "
            f"{arm['information_structure']} | "
            f"{arm['comparator_status']} | "
            f"{arm['task_cost']['mean']:.8g} | "
            f"{arm['task_cost']['std']:.8g} | "
            f"{arm['peak_forward_velocity']['mean']:.8g} | "
            f"{arm['terminal_position_error_m']['mean']:.8g} | "
            f"{estimator_text} |"
        )

    certificate_rows = [
        "| Arm | Certificate type | lambda/gamma^2 | finite-gamma feasible | Notes |",
        "|---|---|---:|---|---|",
    ]
    for label, cert in summary["deterministic_certificate_sidecar"].items():
        certificate_rows.append(
            "| "
            f"`{label}` | "
            f"{cert['certificate_type']} | "
            f"{cert['lambda_over_gamma_squared']:.8g} | "
            f"{cert['gamma_penalized_feasible']} | "
            f"{cert['notes']} |"
        )

    return f"""# Phase 1 Released Stochastic Evaluation

Issue: `{summary["issue"]}`. Umbrella: `{summary["umbrella"]}`. Deterministic
Phase 1 comparator: `{summary["deterministic_phase1_issue"]}`.

Rerun metadata:

- Discretization: `{summary["rerun_metadata"]["discretization"]}`.
- Lane: `{summary["rerun_metadata"]["lane"]}`.
- Lane scope: {summary["rerun_metadata"]["lane_description"]}
- Bellman claim: `{summary["bellman_claim"]}`.

This note materializes the Phase 1 released-code stochastic lane. All arms use
the Euler plant and sampled sensory, input-image additive motor noise,
signal-dependent input-image motor noise, and separate process/load noise.
Each seed reuses the same noise bundle across
arms, so output-feedback LQG and robust comparisons use common random numbers.

## Comparator Scope

The output-feedback LQG arm uses the local port of the C&S
`extLQG -> computeOFC -> computeExtKalman` fixed-point comparator. The robust
arm uses the local C&S-style output-feedback H-infinity gains. No stochastic
Bellman objective or Bellman parity is claimed in this lane.

## Summary Metrics

Trials: `{summary["n_trials"]}`. Seeds: `{summary["seeds"]}`.
Output-feedback certificate gamma factor:
`{summary["output_feedback_certificate_gamma_factor"]}`.

{"\n".join(rows)}

## Deterministic Certificate Sidecar

These values audit the exact controller gains used by the stochastic forward
simulation under the deterministic finite-gamma quadratic checks. They are not
Monte Carlo stochastic induced-gain certificates.

{"\n".join(certificate_rows)}

## Noise Contract

- Contract: `{summary["noise_contract"]["contract"]}`.
- Additive motor covariance: `{summary["noise_contract"]["additive_motor_covariance"]}`.
- Motor covariance scale: `{summary["noise_contract"]["motor_covariance_scale"]}`.
- Process/load noise: `{summary["noise_contract"]["process_noise"]}`.
- Process covariance scale: `{summary["noise_contract"]["process_covariance_scale"]}`.
- Signal-dependent motor noise: `{summary["noise_contract"]["signal_dependent_motor_noise"]}`.
- Signal-dependent tensor scale: `{summary["noise_contract"]["signal_dependent_scale"]}`.
- Shared-noise policy: {summary["noise_contract"]["shared_noise_policy"]}

## Interpretation

This is a released forward-simulation check for exact controller families where
local exact arrays exist. It should be read beside the deterministic analytical
Phase 1 result, not as a replacement for it. The output-feedback LQG row now
uses the local extLQG fixed-point path; remaining fidelity questions should be
treated as numerical/audit questions against the MATLAB code, not as a missing
comparator implementation.
"""


def write_outputs(
    issue_id: str = ISSUE_ID,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    noise_config: CSReleasedStochasticNoiseConfig = DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
) -> dict[str, Any]:
    """Write tracked Phase 1 stochastic summary outputs and bulk arrays."""

    reference = materialize_reference(
        gamma_factors=(PRIMARY_GAMMA_FACTOR, OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR)
    )
    result = analyze_phase1_stochastic(seeds=seeds, noise_config=noise_config)
    summary = {
        **result_summary(result, discretization=discretization),
        "game_card_summary": reference_summary(
            reference,
            discretization=discretization,
            lane=LANE,
        ),
    }
    results_dir = mkdir_p(REPO_ROOT / "results" / issue_id)
    notes_dir = mkdir_p(results_dir / "notes")
    artifact_dir = mkdir_p(REPO_ROOT / "_artifacts" / issue_id / "cs_stochastic_phase1")
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "C&S fidelity correction artifacts for issue dd232cd. See notes for the "
            "Euler rerun plan and released stochastic Phase 1 evaluation.\n",
            encoding="utf-8",
        )
    arrays = _npz_arrays(result)
    npz_path = artifact_dir / "cs_stochastic_phase1.npz"
    np.savez_compressed(npz_path, **arrays)
    summary["tracked_note"] = f"results/{issue_id}/notes/cs_stochastic_phase1.md"
    summary["tracked_manifest"] = f"results/{issue_id}/notes/cs_stochastic_phase1_manifest.json"
    summary["artifact_npz"] = f"_artifacts/{issue_id}/cs_stochastic_phase1/{npz_path.name}"
    summary["artifact_npz_keys"] = sorted(arrays.keys())
    note_path = notes_dir / "cs_stochastic_phase1.md"
    manifest_path = notes_dir / "cs_stochastic_phase1_manifest.json"
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _arm_summary(
    *,
    rollouts: list[CSFullStateStochasticRollout | CSStochasticRollout],
    schedule: CostSchedule,
    controller_family: str,
    information_structure: str,
    comparator_status: str,
) -> dict[str, Any]:
    metrics = [_rollout_metric_dict(rollout, schedule) for rollout in rollouts]
    return {
        "controller_family": controller_family,
        "information_structure": information_structure,
        "comparator_status": comparator_status,
        "n_trials": len(metrics),
        "task_cost": _mean_std([metric["task_cost"] for metric in metrics]),
        "peak_forward_velocity": _mean_std([metric["peak_forward_velocity"] for metric in metrics]),
        "terminal_position_error_m": _mean_std(
            [metric["terminal_position_error_m"] for metric in metrics]
        ),
        "control_effort": _mean_std([metric["control_effort"] for metric in metrics]),
        "estimator_rms_error": _optional_mean_std(
            [metric["estimator_rms_error"] for metric in metrics]
        ),
    }


def _build_extlqg_comparator_path(
    plant: PlantLinearization,
    controller_gains: Float[Array, "T m_u n"],
    covariances: CSNoiseCovariances,
    *,
    schedule: CostSchedule,
    config: OutputFeedbackConfig,
):
    """Build the extLQG comparator, using the fixed-point API when present."""

    try:
        return build_extlqg_comparator_path(
            plant,
            controller_gains,
            covariances,
            schedule=schedule,
            config=config,
        )
    except TypeError as exc:
        if "schedule" not in str(exc):
            raise
        return build_extlqg_comparator_path(plant, controller_gains, covariances, config)


def _deterministic_certificate_sidecar(
    *,
    plant: PlantLinearization,
    schedule: CostSchedule,
    x0: Float[Array, " n"],
    gamma: float,
    lqr_value_matrices: Float[Array, "T_plus_1 n n"],
    hinf_solution,
    extlqg_gains: Float[Array, "T m_u n"],
    output_feedback_hinf_gains: Float[Array, "T m_u n"],
    config: OutputFeedbackConfig,
) -> dict[str, dict[str, float | bool | str]]:
    """Audit the stochastic-lane exact gains with deterministic gamma checks."""

    robust_policy = robust_estimator_fixed_adversary_policy(
        plant,
        schedule,
        hinf_solution,
        output_feedback_hinf_gains,
        robust_estimator_covariances(plant, schedule, gamma, config),
        config,
    )
    robust_rollout = rollout_with_robust_estimator_policy(
        plant,
        schedule,
        hinf_solution,
        x0,
        robust_policy,
        gains=output_feedback_hinf_gains,
        config=config,
    )
    budget = float(jnp.sum(robust_rollout.epsilon**2))
    lqg_audit = exact_output_feedback_adversary_audit(
        label="phase1_stochastic_extlqg_sidecar",
        plant=plant,
        schedule=schedule,
        controller_gains=extlqg_gains,
        x0=x0,
        budget=budget,
        estimator_kind="kalman",
        penalty_gamma=gamma,
        config=config,
    )
    output_hinf_audit = exact_output_feedback_adversary_audit(
        label="phase1_stochastic_output_feedback_hinf_sidecar",
        plant=plant,
        schedule=schedule,
        controller_gains=output_feedback_hinf_gains,
        x0=x0,
        budget=budget,
        estimator_kind="robust",
        solution=hinf_solution,
        penalty_gamma=gamma,
        config=config,
    )
    full_lqr_ratio = _full_state_lambda_over_gamma_squared(
        plant,
        lqr_value_matrices,
        gamma,
    )
    full_hinf_ratio = float(jnp.max(hinf_solution.spectral_radii))
    return {
        "full_state_lqr": {
            "certificate_type": "full_state_riccati_value_sidecar",
            "lambda_over_gamma_squared": full_lqr_ratio,
            "gamma_penalized_feasible": full_lqr_ratio < 1.0,
            "notes": "computed from LQR value matrices at the finite H-infinity gamma",
        },
        "full_state_hinf": {
            "certificate_type": "full_state_hinf_riccati_admissibility",
            "lambda_over_gamma_squared": full_hinf_ratio,
            "gamma_penalized_feasible": bool(hinf_solution.admissible and full_hinf_ratio < 1.0),
            "notes": "max stored Riccati spectral radius for the H-infinity solution",
        },
        "output_feedback_lqg_extlqg": _output_feedback_certificate_row(
            lqg_audit,
            notes="Kalman/extLQG fixed-gain deterministic flattened-epsilon audit",
        ),
        "output_feedback_hinf": _output_feedback_certificate_row(
            output_hinf_audit,
            notes="robust-estimator deterministic flattened-epsilon audit",
        ),
    }


def _full_state_lambda_over_gamma_squared(
    plant: PlantLinearization,
    value_matrices: Float[Array, "T_plus_1 n n"],
    gamma: float,
) -> float:
    gamma2 = float(gamma * gamma)
    ratios = []
    for P_next in value_matrices[1:]:
        block = plant.Bw.T @ P_next @ plant.Bw
        ratios.append(float(jnp.linalg.eigvalsh(0.5 * (block + block.T))[-1]) / gamma2)
    return max(ratios)


def _output_feedback_certificate_row(
    audit: dict[str, Any],
    *,
    notes: str,
) -> dict[str, float | bool | str]:
    return {
        "certificate_type": "output_feedback_flattened_epsilon_sidecar",
        "lambda_over_gamma_squared": float(
            audit["gamma_penalized"]["max_eigenvalue_over_gamma_squared"]
        ),
        "gamma_penalized_feasible": bool(audit["gamma_penalized"]["feasible"]),
        "notes": notes,
    }


def _rollout_metric_dict(
    rollout: CSFullStateStochasticRollout | CSStochasticRollout,
    schedule: CostSchedule,
) -> dict[str, float | int | None]:
    x = rollout.x.astype(jnp.float64)
    u = rollout.u_applied.astype(jnp.float64)
    task_cost = _task_cost(schedule, x, u)
    estimator_rms = None
    if isinstance(rollout, CSStochasticRollout):
        estimator_rms = float(jnp.sqrt(jnp.mean((rollout.x - rollout.x_hat) ** 2)))
    return {
        "task_cost": task_cost,
        "peak_forward_velocity": rollout.peak_forward_velocity,
        "peak_forward_velocity_idx": rollout.peak_forward_velocity_idx,
        "terminal_position_error_m": rollout.terminal_position_error,
        "control_effort": rollout.control_effort,
        "estimator_rms_error": estimator_rms,
    }


def _task_cost(
    schedule: CostSchedule,
    x: Float[Array, "T_plus_1 n"],
    u: Float[Array, "T m_u"],
) -> float:
    state_terms = jnp.einsum("ti,tij,tj->t", x[:-1], schedule.Q, x[:-1])
    control_terms = jnp.einsum("ti,tij,tj->t", u, schedule.R, u)
    terminal = x[-1] @ schedule.Q_f @ x[-1]
    return float(jnp.sum(state_terms) + jnp.sum(control_terms) + terminal)


def _mean_std(values: list[float | int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
    }


def _optional_mean_std(values: list[float | None]) -> dict[str, float | None]:
    finite = [value for value in values if value is not None]
    if not finite:
        return {"mean": None, "std": None}
    return _mean_std(finite)


def _summary_fields(
    plant: PlantLinearization,
    x: Float[Array, "T_plus_1 n"],
    u: Float[Array, "T m_u"],
) -> tuple[float, int, float, float]:
    pos = x[:, plant.pos_slice[0] : plant.pos_slice[1]]
    vel = x[:, plant.vel_slice[0] : plant.vel_slice[1]]
    forward = vel @ jnp.array([1.0, 0.0], dtype=jnp.float64)
    pos_abs = pos + TARGET_POS[None, :]
    terminal = jnp.linalg.norm(pos_abs[-1] - TARGET_POS)
    return (
        float(jnp.max(forward)),
        int(jnp.argmax(forward)),
        float(terminal),
        float(jnp.sum(jnp.linalg.norm(u, axis=-1) ** 2) * plant.dt),
    )


def _npz_arrays(result: Phase1StochasticResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "seeds": np.asarray(result.seeds, dtype=np.int64),
    }
    arms = {
        "full_state_lqr": [trial.full_state_lqr for trial in result.trials],
        "full_state_hinf": [trial.full_state_hinf for trial in result.trials],
        "output_feedback_lqg": [trial.output_feedback_lqg for trial in result.trials],
        "output_feedback_hinf": [trial.output_feedback_hinf for trial in result.trials],
    }
    for label, rollouts in arms.items():
        arrays[f"{label}_x"] = np.stack([np.asarray(rollout.x) for rollout in rollouts], axis=0)
        arrays[f"{label}_u"] = np.stack(
            [np.asarray(rollout.u_applied) for rollout in rollouts],
            axis=0,
        )
    return arrays


__all__ = [
    "CSFullStateStochasticRollout",
    "DEFAULT_SEEDS",
    "Phase1StochasticResult",
    "Phase1StochasticTrial",
    "analyze_phase1_stochastic",
    "render_markdown",
    "result_summary",
    "simulate_full_state_released_forward",
    "write_outputs",
]
