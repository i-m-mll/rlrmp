"""State-weighted Phase 3 linear equivalence certificate.

This module consumes the Phase 3 linear round-trip result and asks whether the
objective-trained LQR controllers are equivalent to analytical LQR in the
state/action directions that matter for clean and disturbance-relevant behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from rlrmp.analysis.math.adversary_equivalence import rollout_arrays_with_open_loop_epsilon
from rlrmp.analysis.math.cs_game_card import GameCardReference
from rlrmp.analysis.math.linear_round_trip import (
    LinearOptimizationConfig,
    Phase3LinearRoundTripResult,
    canonical_initial_state,
    ensemble_clean_objective,
    ensemble_initial_states,
    result_summary as round_trip_summary,
    run_phase3_linear_round_trip,
)
from rlrmp.analysis.math.hinf_riccati import CostSchedule, PlantLinearization
from rlrmp.analysis.math.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    build_rerun_metadata,
)
from rlrmp.analysis.math import require_jax_x64

ISSUE_ID = "d01c35a"
PHASE3_ISSUE_ID = "6f5c79e"
UMBRELLA_ID = "43e8728"


@dataclass(frozen=True)
class CertificateConfig:
    """Configuration for state distributions and numeric thresholds."""

    validation_seed: int = 20260528
    validation_basis_scale: float = 0.01
    validation_random_state_scale: float = 0.02
    n_validation_random_states: int = 64
    covariance_rank_rtol: float = 1e-8
    denominator_floor: float = 1e-12
    behavior_cost_ratio_tol: float = 1.01
    clean_action_mismatch_tol: float = 1e-2
    adversary_action_mismatch_tol: float = 5e-2
    heldout_cost_ratio_tol: float = 1.02


@dataclass(frozen=True)
class StateDistribution:
    """Per-time state distribution used for weighted certificate metrics."""

    label: str
    x: Float[Array, "T_plus_1 batch n"]


@dataclass(frozen=True)
class DistributionMetrics:
    """Metrics for one controller on one state distribution."""

    label: str
    action_delta_rms: float
    action_reference_rms: float
    action_mismatch_ratio_mean: float
    action_mismatch_ratio_max: float
    transition_delta_rms: float
    transition_reference_rms: float
    transition_mismatch_ratio_mean: float
    transition_mismatch_ratio_max: float
    bellman_action_delta_rms: float
    bellman_action_reference_rms: float
    bellman_action_mismatch_ratio_mean: float
    bellman_action_mismatch_ratio_max: float
    mean_effective_rank: float
    min_effective_rank: float
    mean_identifiable_rank: float
    min_identifiable_rank: float
    gain_error_parallel_fraction_mean: float
    gain_error_parallel_fraction_max: float
    gain_error_orthogonal_fraction_mean: float
    gain_error_orthogonal_fraction_max: float


@dataclass(frozen=True)
class ControllerCertificate:
    """State/value/disturbance certificate for one objective-trained controller."""

    label: str
    raw_gain_relative_error: float
    clean_cost: float
    clean_cost_ratio: float
    heldout_cost: float
    heldout_cost_ratio: float
    objective_ratio: float
    final_gradient_norm: float
    reference_gradient_norm: float
    value_gap_clean_reference_cov_ratio: float
    value_gap_training_cov_ratio: float
    interpolation_objective_ratios: tuple[float, ...]
    distribution_metrics: tuple[DistributionMetrics, ...]
    classification: str


@dataclass(frozen=True)
class LinearEquivalenceCertificateResult:
    """Complete d01c35a certificate result."""

    phase3: Phase3LinearRoundTripResult
    config: CertificateConfig
    controllers: tuple[ControllerCertificate, ...]


def _outer_covariance(states: Float[Array, "T_plus_1 batch n"]) -> Float[Array, "T_plus_1 n n"]:
    return jnp.einsum("tbi,tbj->tij", states, states) / states.shape[1]


def _rollout_states_for_initials(
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    states: Float[Array, "batch n"],
) -> Float[Array, "T_plus_1 batch n"]:
    epsilon = jnp.zeros((K.shape[0], plant.m_w), dtype=jnp.float64)

    def rollout_one(x0: Float[Array, " n"]) -> Float[Array, "T_plus_1 n"]:
        x, _u = rollout_arrays_with_open_loop_epsilon(plant, K, x0, epsilon)
        return x

    return jax.vmap(rollout_one)(states).swapaxes(0, 1)


def _validation_initial_states(
    plant: PlantLinearization,
    config: CertificateConfig,
) -> Float[Array, "batch n"]:
    x0 = canonical_initial_state(plant)
    basis = jnp.eye(plant.n, dtype=jnp.float64)
    random_states = jr.normal(
        jr.PRNGKey(config.validation_seed),
        (config.n_validation_random_states, plant.n),
        dtype=jnp.float64,
    )
    return jnp.concatenate(
        [
            x0[None],
            config.validation_basis_scale * basis,
            -config.validation_basis_scale * basis,
            config.validation_random_state_scale * random_states,
        ],
        axis=0,
    )


def _state_distributions(
    result: Phase3LinearRoundTripResult,
    controller_label: str,
    K: Float[Array, "T m_u n"],
    config: CertificateConfig,
) -> tuple[StateDistribution, ...]:
    plant = result.reference.plant
    K_ref = result.reference.lqr_solution.K
    training_states, _weights = ensemble_initial_states(plant, LinearOptimizationConfig())
    validation_states = _validation_initial_states(plant, config)
    audit = next(audit for audit in result.audits if audit.label == controller_label)
    analytical_lqr_audit = next(
        audit for audit in result.audits if audit.label == "analytical_lqr_reference"
    )
    return (
        StateDistribution(
            label="canonical_clean_reference",
            x=result.reference.lqr_rollout.x[:, None, :],
        ),
        StateDistribution(
            label="training_ensemble_reference_rollouts",
            x=_rollout_states_for_initials(plant, K_ref, training_states),
        ),
        StateDistribution(
            label="validation_ensemble_reference_rollouts",
            x=_rollout_states_for_initials(plant, K_ref, validation_states),
        ),
        StateDistribution(
            label="candidate_heldout_adversary_states",
            x=audit.heldout.rollout.x[:, None, :],
        ),
        StateDistribution(
            label="analytical_lqr_heldout_adversary_states",
            x=analytical_lqr_audit.heldout.rollout.x[:, None, :],
        ),
    )


def policy_evaluation_matrices(
    plant: PlantLinearization,
    schedule: CostSchedule,
    K: Float[Array, "T m_u n"],
) -> Float[Array, "T_plus_1 n n"]:
    """Return finite-horizon value matrices for a fixed linear policy."""

    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    P_next = schedule.Q_f.astype(jnp.float64)
    matrices = [P_next]
    for t in range(schedule.T - 1, -1, -1):
        K_t = K[t].astype(jnp.float64)
        M_t = A - B @ K_t
        P_t = schedule.Q[t] + K_t.T @ schedule.R[t] @ K_t + M_t.T @ P_next @ M_t
        P_t = 0.5 * (P_t + P_t.T)
        matrices.append(P_t)
        P_next = P_t
    return jnp.stack(list(reversed(matrices)), axis=0)


def _safe_ratio(
    numerator: Float[Array, ""], denominator: Float[Array, ""], floor: float
) -> Float[Array, ""]:
    return numerator / jnp.maximum(denominator, floor)


def _distribution_metrics(
    *,
    plant: PlantLinearization,
    schedule: CostSchedule,
    K: Float[Array, "T m_u n"],
    K_ref: Float[Array, "T m_u n"],
    P_ref: Float[Array, "T_plus_1 n n"],
    distribution: StateDistribution,
    config: CertificateConfig,
) -> DistributionMetrics:
    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    delta_K = K - K_ref
    states = distribution.x[:-1]
    covariances = _outer_covariance(distribution.x)[:-1]
    M_ref = A[None, :, :] - jnp.einsum("iu,tuj->tij", B, K_ref)
    delta_M = -jnp.einsum("iu,tuj->tij", B, delta_K)
    H = schedule.R + jnp.einsum("na,tnm,mb->tab", B, P_ref[1:], B)

    action_delta = jnp.einsum("tuj,tbj->tbu", delta_K, states)
    action_ref = jnp.einsum("tuj,tbj->tbu", K_ref, states)
    transition_delta = jnp.einsum("tij,tbj->tbi", delta_M, states)
    transition_ref = jnp.einsum("tij,tbj->tbi", M_ref, states)

    action_num = jnp.mean(
        jnp.einsum("tbi,tij,tbj->tb", action_delta, schedule.R, action_delta), axis=1
    )
    action_den = jnp.mean(jnp.einsum("tbi,tij,tbj->tb", action_ref, schedule.R, action_ref), axis=1)
    transition_num = jnp.mean(jnp.sum(transition_delta**2, axis=-1), axis=1)
    transition_den = jnp.mean(jnp.sum(transition_ref**2, axis=-1), axis=1)
    bellman_num = jnp.einsum("tui,tuv,tvj,tij->t", delta_K, H, delta_K, covariances)
    bellman_den = jnp.einsum("tui,tuv,tvj,tij->t", K_ref, H, K_ref, covariances)

    action_ratio = _safe_ratio(action_num, action_den, config.denominator_floor)
    transition_ratio = _safe_ratio(transition_num, transition_den, config.denominator_floor)
    bellman_ratio = _safe_ratio(bellman_num, bellman_den, config.denominator_floor)

    singular_values = jax.vmap(lambda cov: jnp.linalg.svd(cov, compute_uv=False))(covariances)
    max_s = singular_values[:, :1]
    identifiable = singular_values > jnp.maximum(
        max_s * config.covariance_rank_rtol, config.denominator_floor
    )
    identifiable_rank = jnp.sum(identifiable, axis=1)
    normalized_s = singular_values / jnp.maximum(
        jnp.sum(singular_values, axis=1, keepdims=True), config.denominator_floor
    )
    entropy = -jnp.sum(
        jnp.where(normalized_s > 0, normalized_s * jnp.log(normalized_s), 0.0), axis=1
    )
    effective_rank = jnp.exp(entropy)

    def error_fractions(
        cov: Float[Array, "n n"], dK: Float[Array, "m_u n"]
    ) -> Float[Array, " two"]:
        u, s, _vh = jnp.linalg.svd(cov, full_matrices=False)
        keep = s > jnp.maximum(jnp.max(s) * config.covariance_rank_rtol, config.denominator_floor)
        basis = u * keep[None, :]
        projection = basis @ basis.T
        parallel = dK @ projection
        total = jnp.sum(dK**2)
        parallel_fraction = _safe_ratio(jnp.sum(parallel**2), total, config.denominator_floor)
        orthogonal_fraction = jnp.maximum(0.0, 1.0 - parallel_fraction)
        return jnp.array([parallel_fraction, orthogonal_fraction], dtype=jnp.float64)

    fractions = jax.vmap(error_fractions)(covariances, delta_K)
    return DistributionMetrics(
        label=distribution.label,
        action_delta_rms=float(jnp.sqrt(jnp.mean(action_num))),
        action_reference_rms=float(jnp.sqrt(jnp.mean(action_den))),
        action_mismatch_ratio_mean=float(jnp.mean(action_ratio)),
        action_mismatch_ratio_max=float(jnp.max(action_ratio)),
        transition_delta_rms=float(jnp.sqrt(jnp.mean(transition_num))),
        transition_reference_rms=float(jnp.sqrt(jnp.mean(transition_den))),
        transition_mismatch_ratio_mean=float(jnp.mean(transition_ratio)),
        transition_mismatch_ratio_max=float(jnp.max(transition_ratio)),
        bellman_action_delta_rms=float(jnp.sqrt(jnp.mean(bellman_num))),
        bellman_action_reference_rms=float(jnp.sqrt(jnp.mean(bellman_den))),
        bellman_action_mismatch_ratio_mean=float(jnp.mean(bellman_ratio)),
        bellman_action_mismatch_ratio_max=float(jnp.max(bellman_ratio)),
        mean_effective_rank=float(jnp.mean(effective_rank)),
        min_effective_rank=float(jnp.min(effective_rank)),
        mean_identifiable_rank=float(jnp.mean(identifiable_rank)),
        min_identifiable_rank=float(jnp.min(identifiable_rank)),
        gain_error_parallel_fraction_mean=float(jnp.mean(fractions[:, 0])),
        gain_error_parallel_fraction_max=float(jnp.max(fractions[:, 0])),
        gain_error_orthogonal_fraction_mean=float(jnp.mean(fractions[:, 1])),
        gain_error_orthogonal_fraction_max=float(jnp.max(fractions[:, 1])),
    )


def _objective_gradient_norm(
    reference: GameCardReference,
    K: Float[Array, "T m_u n"],
) -> float:
    states, weights = ensemble_initial_states(reference.plant, LinearOptimizationConfig())

    def objective(gains: Float[Array, "T m_u n"]) -> Float[Array, ""]:
        return ensemble_clean_objective(reference.plant, reference.schedule, gains, states, weights)

    grads = jax.grad(objective)(K)
    return float(jnp.linalg.norm(grads))


def _interpolation_objective_ratios(
    reference: GameCardReference,
    K: Float[Array, "T m_u n"],
    K_ref: Float[Array, "T m_u n"],
) -> tuple[float, ...]:
    states, weights = ensemble_initial_states(reference.plant, LinearOptimizationConfig())
    ref_objective = ensemble_clean_objective(
        reference.plant, reference.schedule, K_ref, states, weights
    )
    ratios = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        K_alpha = (1.0 - alpha) * K + alpha * K_ref
        value = ensemble_clean_objective(
            reference.plant, reference.schedule, K_alpha, states, weights
        )
        ratios.append(float(value / ref_objective))
    return tuple(ratios)


def _value_gap_ratio(
    P: Float[Array, "T_plus_1 n n"],
    P_ref: Float[Array, "T_plus_1 n n"],
    sigma0: Float[Array, "n n"],
    floor: float,
) -> float:
    numerator = jnp.trace((P[0] - P_ref[0]) @ sigma0)
    denominator = jnp.trace(P_ref[0] @ sigma0)
    return float(_safe_ratio(numerator, denominator, floor))


def _classify_controller(
    *,
    clean_cost_ratio: float,
    heldout_cost_ratio: float,
    clean_action_mismatch: float,
    adversary_action_mismatch: float,
    gradient_norm: float,
    config: CertificateConfig,
) -> str:
    if (
        heldout_cost_ratio <= config.heldout_cost_ratio_tol
        and adversary_action_mismatch <= config.adversary_action_mismatch_tol
    ):
        if (
            clean_cost_ratio <= config.behavior_cost_ratio_tol
            and clean_action_mismatch <= config.clean_action_mismatch_tol
        ):
            return "disturbance_equivalent"
    if gradient_norm > 1e-2:
        return "optimizer_uncertain_not_disturbance_equivalent"
    return "not_disturbance_equivalent"


def _controller_certificate(
    result: Phase3LinearRoundTripResult,
    label: str,
    K: Float[Array, "T m_u n"],
    raw_gain_error: float,
    objective_ratio: float,
    config: CertificateConfig,
) -> ControllerCertificate:
    reference = result.reference
    plant = reference.plant
    schedule = reference.schedule
    K_ref = reference.lqr_solution.K
    P_ref = reference.lqr_solution.P
    P = policy_evaluation_matrices(plant, schedule, K)
    clean = next(audit for audit in result.audits if audit.label == label)
    lqr_clean = next(audit for audit in result.audits if audit.label == "analytical_lqr_reference")
    clean_cost_ratio = clean.clean_cost / lqr_clean.clean_cost
    heldout_cost_ratio = (
        clean.heldout.cost.total_without_disturbance_penalty
        / lqr_clean.heldout.cost.total_without_disturbance_penalty
    )
    distributions = _state_distributions(result, label, K, config)
    distribution_metrics = tuple(
        _distribution_metrics(
            plant=plant,
            schedule=schedule,
            K=K,
            K_ref=K_ref,
            P_ref=P_ref,
            distribution=distribution,
            config=config,
        )
        for distribution in distributions
    )
    clean_metric = next(
        metric for metric in distribution_metrics if metric.label == "canonical_clean_reference"
    )
    adversary_metric = next(
        metric
        for metric in distribution_metrics
        if metric.label == "candidate_heldout_adversary_states"
    )
    training_states, _weights = ensemble_initial_states(plant, LinearOptimizationConfig())
    training_sigma0 = training_states.T @ training_states / training_states.shape[0]
    clean_x0 = reference.lqr_rollout.x[0]
    clean_sigma0 = jnp.outer(clean_x0, clean_x0)
    final_gradient_norm = _objective_gradient_norm(reference, K)
    reference_gradient_norm = _objective_gradient_norm(reference, K_ref)
    return ControllerCertificate(
        label=label,
        raw_gain_relative_error=raw_gain_error,
        clean_cost=clean.clean_cost,
        clean_cost_ratio=float(clean_cost_ratio),
        heldout_cost=clean.heldout.cost.total_without_disturbance_penalty,
        heldout_cost_ratio=float(heldout_cost_ratio),
        objective_ratio=objective_ratio,
        final_gradient_norm=final_gradient_norm,
        reference_gradient_norm=reference_gradient_norm,
        value_gap_clean_reference_cov_ratio=_value_gap_ratio(
            P, P_ref, clean_sigma0, config.denominator_floor
        ),
        value_gap_training_cov_ratio=_value_gap_ratio(
            P, P_ref, training_sigma0, config.denominator_floor
        ),
        interpolation_objective_ratios=_interpolation_objective_ratios(reference, K, K_ref),
        distribution_metrics=distribution_metrics,
        classification=_classify_controller(
            clean_cost_ratio=float(clean_cost_ratio),
            heldout_cost_ratio=float(heldout_cost_ratio),
            clean_action_mismatch=clean_metric.action_mismatch_ratio_mean,
            adversary_action_mismatch=adversary_metric.action_mismatch_ratio_mean,
            gradient_norm=final_gradient_norm,
            config=config,
        ),
    )


def run_linear_equivalence_certificate(
    *,
    config: CertificateConfig = CertificateConfig(),
    training_config: LinearOptimizationConfig = LinearOptimizationConfig(),
    quasi_newton_config: LinearOptimizationConfig = LinearOptimizationConfig(n_steps=500),
    heldout_step_sweep: tuple[int, ...] = (50, 200),
    heldout_restarts: int = 4,
) -> LinearEquivalenceCertificateResult:
    """Run the d01c35a certificate against the current Phase 3 controllers."""

    require_jax_x64("linear equivalence certificate analysis")
    phase3 = run_phase3_linear_round_trip(
        training_config=training_config,
        quasi_newton_config=quasi_newton_config,
        heldout_step_sweep=heldout_step_sweep,
        heldout_restarts=heldout_restarts,
    )
    summary = round_trip_summary(phase3)
    adam = summary["objective_trainings"][phase3.lqr_training.label]
    quasi = summary["objective_trainings"][phase3.lqr_quasi_newton_training.label]
    controllers = (
        _controller_certificate(
            phase3,
            phase3.lqr_training.label,
            phase3.lqr_training.K,
            phase3.lqr_training.gain_relative_error,
            adam["objective_ratio_to_reference"],
            config,
        ),
        _controller_certificate(
            phase3,
            phase3.lqr_quasi_newton_training.label,
            phase3.lqr_quasi_newton_training.K,
            phase3.lqr_quasi_newton_training.gain_relative_error,
            quasi["objective_ratio_to_reference"],
            config,
        ),
    )
    return LinearEquivalenceCertificateResult(
        phase3=phase3,
        config=config,
        controllers=controllers,
    )


def _distribution_metrics_summary(metric: DistributionMetrics) -> dict[str, Any]:
    return metric.__dict__


def _controller_summary(controller: ControllerCertificate) -> dict[str, Any]:
    return {
        "label": controller.label,
        "classification": controller.classification,
        "raw_gain_relative_error": controller.raw_gain_relative_error,
        "objective_ratio": controller.objective_ratio,
        "clean_cost": controller.clean_cost,
        "clean_cost_ratio": controller.clean_cost_ratio,
        "heldout_cost": controller.heldout_cost,
        "heldout_cost_ratio": controller.heldout_cost_ratio,
        "final_gradient_norm": controller.final_gradient_norm,
        "reference_gradient_norm": controller.reference_gradient_norm,
        "value_gap_clean_reference_cov_ratio": controller.value_gap_clean_reference_cov_ratio,
        "value_gap_training_cov_ratio": controller.value_gap_training_cov_ratio,
        "interpolation_objective_ratios": list(controller.interpolation_objective_ratios),
        "distribution_metrics": [
            _distribution_metrics_summary(metric) for metric in controller.distribution_metrics
        ],
    }


def result_summary(
    result: LinearEquivalenceCertificateResult,
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
) -> dict[str, Any]:
    """Return JSON-serializable certificate summary."""

    phase3_summary = round_trip_summary(
        result.phase3,
        discretization=discretization,
        lane=lane,
    )
    return {
        "issue": ISSUE_ID,
        "phase3_issue": PHASE3_ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "regeneration_command": "PYTHONPATH=src python scripts/materialize_linear_equivalence_certificate.py",
        "rerun_metadata": build_rerun_metadata(
            discretization=discretization,
            lane=lane,
            materializer="linear_equivalence_certificate",
        ),
        "phase3_status_before_certificate": phase3_summary["phase3_status"],
        "phase3_best_objective_training": phase3_summary["best_objective_training"],
        "certificate_config": result.config.__dict__,
        "controllers": [_controller_summary(controller) for controller in result.controllers],
        "overall_status": (
            "blocked_not_disturbance_equivalent"
            if any(
                controller.classification != "disturbance_equivalent"
                for controller in result.controllers
            )
            else "passed"
        ),
        "interpretation": (
            "The richer certificate treats clean behavior as insufficient. "
            "Controllers must also match analytical LQR on state-weighted action, "
            "closed-loop transition, value, and held-out disturbance-relevant metrics."
        ),
    }


__all__ = [
    "CertificateConfig",
    "ControllerCertificate",
    "DistributionMetrics",
    "ISSUE_ID",
    "LinearEquivalenceCertificateResult",
    "PHASE3_ISSUE_ID",
    "StateDistribution",
    "policy_evaluation_matrices",
    "result_summary",
    "run_linear_equivalence_certificate",
]
