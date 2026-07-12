"""Reusable standard-certificate analysis science."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.math.cs_game_card import OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    kalman_estimator_joint_matrices,
)
from rlrmp.analysis.bridge_certificates import build_standard_certificate_components
from rlrmp.analysis.bridge_results import (
    BridgeArchitecture,
    BridgeCertificateComponent,
    BridgeCertificateMode,
    BridgeAnalysisResult,
    BridgeRunSpec,
    make_bridge_run_id,
)
from rlrmp.paths import REPO_ROOT


BEHAVIORAL_ACTION_SIDECAR = "behavioral_action_sidecar"
DETERMINISTIC_AUDIT_SIDECAR = "deterministic_exact_l2_and_gamma_sidecar"
GAIN_DIAGNOSTIC_SIDECAR = "gain_diagnostic_sidecar"
ROLLOUT_BEHAVIOR_SIDECAR = "rollout_behavior_sidecar"


@dataclass(frozen=True)
class StandardCertificateRowRequest:
    """In-memory request for one standard-certificate row.

    ``component_kwargs`` is deliberately not part of the serialized manifest.
    It contains arrays and metadata passed through to
    :func:`build_standard_certificate_components`, which remains the umbrella
    standard-certificate entry point for all modes.
    """

    spec: BridgeRunSpec
    architecture: BridgeArchitecture
    status: str
    certificate_mode: BridgeCertificateMode | None = None
    component_kwargs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def build_standard_certificate_manifest(
    request: StandardCertificateRowRequest,
) -> BridgeAnalysisResult:
    """Build one serialized manifest row from a standard-certificate request."""

    components = build_standard_certificate_components(
        architecture=request.architecture,
        certificate_mode=request.certificate_mode,
        **request.component_kwargs,
    )
    return BridgeAnalysisResult(
        spec=request.spec,
        status=request.status,
        metrics=request.metrics,
        artifacts=request.artifacts,
        certificate_components=components,
    )


def component_by_name(
    row: BridgeAnalysisResult | dict[str, Any],
) -> dict[str, BridgeCertificateComponent | dict[str, Any]]:
    """Return certificate components keyed by component name."""

    components: Any
    if isinstance(row, BridgeAnalysisResult):
        components = row.certificate_components
    else:
        components = row.get("certificate_components", ())
    return {
        component.name
        if isinstance(component, BridgeCertificateComponent)
        else component["name"]: component
        for component in components
    }


def component_status_counts(
    rows: list[BridgeAnalysisResult] | list[dict[str, Any]],
) -> dict[str, int]:
    """Return stable ``component:status`` counts for materialized rows."""

    counts: Counter[str] = Counter()
    for row in rows:
        components = (
            row.certificate_components
            if isinstance(row, BridgeAnalysisResult)
            else row.get("certificate_components", ())
        )
        for component in components:
            if isinstance(component, BridgeCertificateComponent):
                name = component.name
                status = component.status
            else:
                name = component["name"]
                status = component["status"]
            counts[f"{name}:{status}"] += 1
    return dict(sorted(counts.items()))


def materialization_summary(rows: list[BridgeAnalysisResult]) -> dict[str, Any]:
    """Return a compact JSON summary for a standard-certificate result."""

    status_counts = Counter(row.status for row in rows)
    return {
        "n_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "component_status_counts": component_status_counts(rows),
    }


def repo_relative(path: Path, *, repo_root: Path) -> str:
    """Return a repo-relative path when possible."""

    try:
        return str(path.absolute().relative_to(repo_root))
    except ValueError:
        return str(path)


def deterministic_output_feedback_rows(
    *,
    fit: dict[str, Any],
    arrays: dict[str, np.ndarray],
    reference: Any,
    output_config: OutputFeedbackConfig,
    family: str,
    run_parts: tuple[str, ...],
    training_distribution: str,
    source_manifest: Path,
    extra_parameters: dict[str, Any],
    notes: str,
    issue_id: str,
    array_prefix: str | None = None,
    architecture: str = "time_constrained_free_gain",
    optimizer_label: str = "lbfgsb_strong_optimizer_whitened",
    gamma_factor: float = OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    reference_controller: str = "analytical_lqr_kalman",
    row_metrics: dict[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[BridgeAnalysisResult]:
    """Build deterministic output-feedback certificate rows from saved arrays."""

    rows = []
    prefix = array_prefix or fit["label"]
    evaluation_lenses = (
        ("nominal_clean", "clean", f"{family} nominal-clean"),
        (
            "riccati_epsilon_response",
            "under_eps",
            f"{family} Riccati-epsilon response",
        ),
    )
    source = repo_relative(source_manifest, repo_root=repo_root)
    for lens_label, array_suffix, evaluation_distribution in evaluation_lenses:
        components = list(
            _output_feedback_standard_components(
                arrays=arrays,
                reference=reference,
                output_config=output_config,
                array_prefix=f"{prefix}_{array_suffix}",
                candidate_gain=np.asarray(arrays[f"{prefix}_K"]),
                optimizer_metadata=_optimizer_metadata(fit),
                state_label=f"{evaluation_distribution}_coupled_state",
                action_state_label=f"{evaluation_distribution}_estimated_state",
                architecture=architecture,
            )
        )
        components.extend(_fit_diagnostic_sidecars(fit))
        spec = BridgeRunSpec(
            issue_id=issue_id,
            run_id=make_bridge_run_id(*run_parts, lens_label),
            objective="optimal",
            architecture=architecture,  # type: ignore[arg-type]
            controller_label=fit["label"],
            optimizer_label=optimizer_label,
            training_distribution=training_distribution,
            evaluation_lane="deterministic",
            reference_controller=reference_controller,
            gamma_factor=gamma_factor,
            parameters={
                "evaluation_distribution": evaluation_distribution,
                "distribution_family": family,
                "source_manifest": source,
            }
            | extra_parameters,
            notes=notes,
        )
        rows.append(
            BridgeAnalysisResult(
                spec=spec,
                status="full_standard_certificate",
                metrics=_fit_metrics(fit)
                | (row_metrics or {})
                | {"certificate_evaluation_lens": lens_label},
                artifacts={"source_manifest": source},
                certificate_components=tuple(components),
            )
        )
    return rows


def deterministic_standard_rows_from_manifest_entries(
    *,
    entries: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    reference: Any,
    output_config: OutputFeedbackConfig,
    issue_id: str,
    source_manifest: Path,
    default_family: str,
    default_source_group: str = "saved",
    default_training_distribution: str = "mixed",
    default_optimizer_label: str = "lbfgsb_strong_optimizer_whitened",
    default_architecture: str = "time_constrained_free_gain",
    default_gamma_factor: float = OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    default_reference_controller: str = "analytical_lqr_kalman",
    default_notes: str = "Full standard certificate computed from saved deterministic arrays.",
) -> list[dict[str, Any]]:
    """Build standard rows from reusable deterministic row descriptors."""

    rows: list[dict[str, Any]] = []
    for entry in entries:
        fit = entry.get("fit", entry)
        label = fit["label"]
        source_group = entry.get("source_group", default_source_group)
        run_parts = tuple(entry.get("run_parts", (source_group, label)))
        row_manifests = deterministic_output_feedback_rows(
            fit=fit,
            arrays=arrays,
            reference=reference,
            output_config=output_config,
            family=entry.get("family", default_family),
            run_parts=run_parts,
            training_distribution=entry.get(
                "training_distribution",
                default_training_distribution,
            ),
            source_manifest=source_manifest,
            extra_parameters=entry.get("parameters", entry.get("row_parameters", {})),
            notes=entry.get("notes", default_notes),
            issue_id=issue_id,
            array_prefix=entry.get("array_prefix", label),
            architecture=entry.get("architecture", default_architecture),
            optimizer_label=entry.get("optimizer_label", default_optimizer_label),
            gamma_factor=entry.get("gamma_factor", default_gamma_factor),
            reference_controller=entry.get(
                "reference_controller",
                default_reference_controller,
            ),
            row_metrics=entry.get("metrics"),
        )
        rows.extend(row.to_payload() for row in row_manifests)
    return rows


def _output_feedback_standard_components(
    *,
    arrays: dict[str, np.ndarray],
    reference: Any,
    output_config: OutputFeedbackConfig,
    array_prefix: str,
    candidate_gain: np.ndarray,
    optimizer_metadata: dict[str, Any],
    state_label: str,
    action_state_label: str,
    architecture: str,
) -> tuple[BridgeCertificateComponent, ...]:
    plant = reference.plant
    schedule = reference.schedule
    reference_gain = np.asarray(reference.lqr_solution.K)
    x = arrays[f"{array_prefix}_x"]
    x_hat = arrays[f"{array_prefix}_x_hat"]
    coupled = np.concatenate([x, x_hat], axis=-1)
    coupled_states = coupled[None, :, :] if coupled.ndim == 2 else coupled
    action_states = x_hat[None, :, :] if x_hat.ndim == 2 else x_hat
    candidate_transition = np.asarray(
        kalman_estimator_joint_matrices(plant, jnp.asarray(candidate_gain), output_config)[0]
    )
    reference_transition = np.asarray(
        kalman_estimator_joint_matrices(plant, jnp.asarray(reference_gain), output_config)[0]
    )
    candidate_values = _policy_value_matrices(schedule, candidate_gain, candidate_transition)
    reference_values = _policy_value_matrices(schedule, reference_gain, reference_transition)
    bellman_hessian = _bellman_hessian(schedule, plant, reference.lqr_solution.P)
    return build_standard_certificate_components(
        architecture=architecture,  # type: ignore[arg-type]
        states=coupled_states,
        action_states=action_states,
        candidate_gain=candidate_gain,
        reference_gain=reference_gain,
        action_weight=np.asarray(schedule.R),
        candidate_transition=candidate_transition,
        reference_transition=reference_transition,
        candidate_value_matrices=candidate_values,
        reference_value_matrices=reference_values,
        bellman_hessian=bellman_hessian,
        optimizer_metadata=optimizer_metadata,
        state_label=state_label,
        action_state_label=action_state_label,
        action_label="control",
    )


def _policy_value_matrices(
    schedule: Any,
    gains: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    n = int(schedule.Q.shape[-1])
    zeros = np.zeros((n, n), dtype=float)
    stage = []
    for q_t, r_t, k_t in zip(schedule.Q, schedule.R, gains, strict=True):
        state_block = np.asarray(q_t, dtype=float)
        control_block = (
            np.asarray(k_t, dtype=float).T
            @ np.asarray(r_t, dtype=float)
            @ np.asarray(k_t, dtype=float)
        )
        stage.append(np.block([[state_block, zeros], [zeros, control_block]]))
    terminal = np.block([[np.asarray(schedule.Q_f, dtype=float), zeros], [zeros, zeros]])
    values = [terminal]
    next_value = terminal
    for a_t, q_t in zip(transition[::-1], np.asarray(stage)[::-1], strict=True):
        next_value = q_t + a_t.T @ next_value @ a_t
        next_value = 0.5 * (next_value + next_value.T)
        values.append(next_value)
    return np.asarray(list(reversed(values)))


def _bellman_hessian(schedule: Any, plant: Any, p_values: np.ndarray) -> np.ndarray:
    p_next = np.asarray(p_values[1:], dtype=float)
    b = np.asarray(plant.B, dtype=float)
    return np.asarray(schedule.R, dtype=float) + np.einsum("iu,tij,jv->tuv", b, p_next, b)


def _fit_diagnostic_sidecars(fit: dict[str, Any]) -> list[BridgeCertificateComponent]:
    return [
        BridgeCertificateComponent.available(
            BEHAVIORAL_ACTION_SIDECAR,
            clean_action_mismatch_ratio=fit.get("clean_action_mismatch_ratio"),
            under_epsilon_action_mismatch_ratio=fit.get("under_epsilon_action_mismatch_ratio"),
            clean_cost=fit.get("clean_cost"),
            under_epsilon_cost_ratio_to_lqr=fit.get("under_epsilon_cost_ratio_to_lqr"),
            evaluation_distribution="deterministic clean plus Riccati-epsilon sidecar",
        ),
        BridgeCertificateComponent.available(
            DETERMINISTIC_AUDIT_SIDECAR,
            exact_l2_cost_ratio_to_lqr=fit.get("exact_l2_cost_ratio_to_lqr"),
            exact_l2_cost_ratio_to_hinf=fit.get("exact_l2_cost_ratio_to_hinf"),
            lambda_over_gamma_squared=fit.get("gamma_penalized_lambda_over_gamma_squared"),
            gamma_penalized_feasible=fit.get("gamma_penalized_feasible"),
        ),
        BridgeCertificateComponent.available(
            GAIN_DIAGNOSTIC_SIDECAR,
            gain_relative_error=fit.get("gain_relative_error"),
            diagnostic_only=True,
            gate="not_used_as_certificate_gate",
        ),
        BridgeCertificateComponent.available(
            ROLLOUT_BEHAVIOR_SIDECAR,
            peak_forward_velocity=fit.get("clean_rollout", {}).get("peak_forward_velocity"),
            terminal_position_error_m=fit.get("clean_rollout", {}).get("terminal_position_error_m"),
            control_effort=fit.get("clean_rollout", {}).get("control_effort"),
        ),
    ]


def _optimizer_metadata(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_status": fit.get("optimizer_status"),
        "optimizer_success": fit.get("optimizer_success"),
        "n_iterations": fit.get("n_iterations"),
        "n_function_evaluations": fit.get("n_function_evaluations"),
        "gradient_norm_initial": fit.get("gradient_norm_initial"),
        "gradient_norm_final": fit.get("gradient_norm_final"),
        "projected_gradient_norm_final": fit.get("projected_gradient_norm_final"),
    }


def _fit_metrics(fit: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "label",
        "objective_ratio_to_reference",
        "gain_relative_error",
        "clean_action_mismatch_ratio",
        "under_epsilon_action_mismatch_ratio",
        "exact_l2_cost_ratio_to_lqr",
        "exact_l2_cost_ratio_to_hinf",
        "gamma_penalized_lambda_over_gamma_squared",
        "gamma_penalized_feasible",
        "n_iterations",
        "optimizer_success",
        "clean_rollout",
    )
    return {key: fit.get(key) for key in keys}


__all__ = [
    "StandardCertificateRowRequest",
    "build_standard_certificate_manifest",
    "component_by_name",
    "component_status_counts",
    "deterministic_output_feedback_rows",
    "deterministic_standard_rows_from_manifest_entries",
    "materialization_summary",
    "repo_relative",
]
