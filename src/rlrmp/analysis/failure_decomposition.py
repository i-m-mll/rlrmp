"""Failure-decomposition helpers for bridge certificate rows.

The standard certificate answers whether a learned controller behaves like a
reference controller on the chosen bridge row.  This module provides the
companion diagnostics for failed rows: objective/gradient comparison,
learned-to-reference interpolation, visited-state gain-error decomposition, and
a compact failure classification.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

FailureClass = Literal[
    "not_failure",
    "under_identification",
    "optimizer_basin",
    "objective_mismatch",
    "sidecar_improving_non_equivalent",
    "io_map_mismatch",
    "representation_failure",
    "mixed",
    "uncertain",
]

SIDECAR_IMPROVING_NON_EQUIVALENT = "sidecar_improving_non_equivalent"
IO_MAP_MISMATCH = "io_map_mismatch"
REPRESENTATION_FAILURE = "representation_failure"

ObjectiveFn = Callable[[np.ndarray], float]
GradientFn = Callable[[np.ndarray], np.ndarray]
MetricFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class FailureDecompositionNumerics:
    """Numerical tolerances for failure-decomposition summaries."""

    denominator_floor: float = 1e-12
    strong_covariance_rtol: float = 1e-8
    weak_covariance_rtol: float = 1e-12
    objective_close_rtol: float = 1e-3
    objective_failure_rtol: float = 1e-2
    gradient_stationary_norm: float = 1e-3
    gradient_large_norm: float = 1.0
    certificate_failure_ratio: float = 1e-2
    weak_or_unvisited_fraction: float = 0.5


def objective_gradient_summary(
    *,
    learned: np.ndarray,
    reference: np.ndarray,
    objective_fn: ObjectiveFn,
    gradient_fn: GradientFn | None = None,
    projected_gradient_fn: GradientFn | None = None,
    numerics: FailureDecompositionNumerics = FailureDecompositionNumerics(),
) -> dict[str, Any]:
    """Summarize objective and gradient diagnostics at learned/reference points.

    Args:
        learned: Learned parameter vector or array in the objective's parameterization.
        reference: Reference parameter vector or array in the same parameterization.
        objective_fn: Function returning a scalar objective for one parameter array.
        gradient_fn: Optional function returning the unconstrained gradient.
        projected_gradient_fn: Optional projected-gradient function. If omitted,
            projected-gradient fields are reported as ``None``.
        numerics: Denominator floor for objective ratios.

    Returns:
        JSON-compatible objective and gradient diagnostics.
    """

    learned_values = np.asarray(learned, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    learned_objective = float(objective_fn(learned_values))
    reference_objective = float(objective_fn(reference_values))
    summary: dict[str, Any] = {
        "learned_objective": learned_objective,
        "reference_objective": reference_objective,
        "learned_to_reference_objective_ratio": _ratio(
            learned_objective,
            reference_objective,
            numerics.denominator_floor,
        ),
    }
    if gradient_fn is not None:
        learned_gradient = np.asarray(gradient_fn(learned_values), dtype=float)
        reference_gradient = np.asarray(gradient_fn(reference_values), dtype=float)
        summary.update(
            {
                "learned_gradient_norm": float(np.linalg.norm(learned_gradient)),
                "reference_gradient_norm": float(np.linalg.norm(reference_gradient)),
            }
        )
    else:
        summary.update(
            {
                "learned_gradient_norm": None,
                "reference_gradient_norm": None,
            }
        )
    if projected_gradient_fn is not None:
        learned_projected = np.asarray(projected_gradient_fn(learned_values), dtype=float)
        reference_projected = np.asarray(projected_gradient_fn(reference_values), dtype=float)
        summary.update(
            {
                "learned_projected_gradient_norm": float(np.linalg.norm(learned_projected)),
                "reference_projected_gradient_norm": float(np.linalg.norm(reference_projected)),
            }
        )
    else:
        summary.update(
            {
                "learned_projected_gradient_norm": None,
                "reference_projected_gradient_norm": None,
            }
        )
    return summary


def interpolation_curve(
    *,
    learned: np.ndarray,
    reference: np.ndarray,
    metric_fns: Mapping[str, MetricFn],
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[dict[str, Any]]:
    """Evaluate metrics on the straight-line path from learned to reference.

    ``alpha=0`` is the learned point and ``alpha=1`` is the reference point.
    """

    learned_values = np.asarray(learned, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    records = []
    for alpha in alphas:
        alpha_float = float(alpha)
        point = (1.0 - alpha_float) * learned_values + alpha_float * reference_values
        metrics = {name: float(fn(point)) for name, fn in metric_fns.items()}
        records.append({"alpha": alpha_float, **metrics})
    return records


def covariances_from_states(
    states: np.ndarray,
    *,
    layout: Literal["batch_time_state", "time_batch_state"] = "batch_time_state",
) -> np.ndarray:
    """Return uncentered per-time state covariances from sample states.

    Args:
        states: State samples with shape ``(batch, time, state)`` or
            ``(time, batch, state)``.
        layout: Input array layout.
    """

    values = np.asarray(states, dtype=float)
    if values.ndim != 3:
        raise ValueError("states must have shape (batch, time, state) or (time, batch, state)")
    if layout == "batch_time_state":
        values = np.swapaxes(values, 0, 1)
    return np.einsum("tbi,tbj->tij", values, values) / values.shape[1]


def gain_error_subspace_decomposition(
    *,
    gain_delta: np.ndarray,
    state_covariances: np.ndarray,
    numerics: FailureDecompositionNumerics = FailureDecompositionNumerics(),
) -> dict[str, Any]:
    """Decompose gain error into strongly, weakly, and unvisited state subspaces.

    The decomposition is performed independently at each time step using the
    eigenspaces of the available state covariance.  Strong dimensions have
    eigenvalues above ``strong_covariance_rtol`` times the largest eigenvalue;
    weak dimensions are below the strong threshold but above
    ``weak_covariance_rtol`` times the largest eigenvalue; the rest are treated
    as unvisited for this diagnostic.
    """

    delta = np.asarray(gain_delta, dtype=float)
    covariances = _match_covariances(
        np.asarray(state_covariances, dtype=float),
        horizon=delta.shape[0],
        state_dim=delta.shape[-1],
    )
    if delta.ndim != 3:
        raise ValueError("gain_delta must have shape (time, action, state)")

    rows = []
    for cov_t, delta_t in zip(covariances, delta, strict=True):
        eigvals, eigvecs = np.linalg.eigh(0.5 * (cov_t + cov_t.T))
        eigvals = np.maximum(eigvals, 0.0)
        max_eval = float(np.max(eigvals)) if eigvals.size else 0.0
        strong_threshold = max(
            max_eval * numerics.strong_covariance_rtol,
            numerics.denominator_floor,
        )
        weak_threshold = max(
            max_eval * numerics.weak_covariance_rtol,
            numerics.denominator_floor,
        )
        strong = eigvals > strong_threshold
        weak = (eigvals > weak_threshold) & ~strong
        unvisited = ~strong & ~weak
        total = float(np.sum(delta_t**2))
        rows.append(
            {
                "strong_fraction": _projected_fraction(
                    delta_t, eigvecs[:, strong], total, numerics
                ),
                "weak_fraction": _projected_fraction(delta_t, eigvecs[:, weak], total, numerics),
                "unvisited_fraction": _projected_fraction(
                    delta_t,
                    eigvecs[:, unvisited],
                    total,
                    numerics,
                ),
                "strong_rank": int(np.sum(strong)),
                "weak_rank": int(np.sum(weak)),
                "unvisited_rank": int(np.sum(unvisited)),
                "max_covariance_eigenvalue": max_eval,
                "total_gain_error_energy": total,
            }
        )

    return {
        "strong_fraction_mean": _mean(rows, "strong_fraction"),
        "strong_fraction_max": _max(rows, "strong_fraction"),
        "weak_fraction_mean": _mean(rows, "weak_fraction"),
        "weak_fraction_max": _max(rows, "weak_fraction"),
        "unvisited_fraction_mean": _mean(rows, "unvisited_fraction"),
        "unvisited_fraction_max": _max(rows, "unvisited_fraction"),
        "weak_or_unvisited_fraction_mean": _mean_sum(
            rows,
            ("weak_fraction", "unvisited_fraction"),
        ),
        "strong_rank_min": _min(rows, "strong_rank"),
        "strong_rank_mean": _mean(rows, "strong_rank"),
        "weak_rank_mean": _mean(rows, "weak_rank"),
        "unvisited_rank_mean": _mean(rows, "unvisited_rank"),
        "total_gain_error_rms": float(
            np.sqrt(np.mean([row["total_gain_error_energy"] for row in rows]))
        ),
        "time_steps": len(rows),
    }


def classify_failure(
    *,
    objective_ratio: float | None,
    learned_gradient_norm: float | None,
    reference_gradient_norm: float | None,
    certificate_mismatch_ratio: float | None,
    io_map_mismatch_ratio: float | None = None,
    representation_failed: bool = False,
    subspace_decomposition: Mapping[str, Any] | None = None,
    sidecar_improved: bool = False,
    equivalence_metrics_failed: bool = False,
    numerics: FailureDecompositionNumerics = FailureDecompositionNumerics(),
) -> dict[str, Any]:
    """Classify a bridge row failure from objective, gradient, and subspace signals."""

    reasons = []
    weak_or_unvisited = (
        None
        if subspace_decomposition is None
        else subspace_decomposition.get("weak_or_unvisited_fraction_mean")
    )
    objective_close = (
        objective_ratio is not None and objective_ratio <= 1.0 + numerics.objective_close_rtol
    )
    objective_bad = (
        objective_ratio is not None and objective_ratio >= 1.0 + numerics.objective_failure_rtol
    )
    certificate_bad = (
        certificate_mismatch_ratio is not None
        and certificate_mismatch_ratio >= numerics.certificate_failure_ratio
    )
    io_map_bad = (
        io_map_mismatch_ratio is not None
        and io_map_mismatch_ratio >= numerics.certificate_failure_ratio
    )
    any_certificate_bad = bool(certificate_bad or io_map_bad or representation_failed)
    learned_stationary = (
        learned_gradient_norm is not None
        and learned_gradient_norm <= numerics.gradient_stationary_norm
    )
    learned_gradient_bad = (
        learned_gradient_norm is not None and learned_gradient_norm >= numerics.gradient_large_norm
    )
    reference_gradient_bad = (
        reference_gradient_norm is not None
        and reference_gradient_norm >= numerics.gradient_large_norm
    )
    weak_subspace_dominates = (
        weak_or_unvisited is not None and weak_or_unvisited >= numerics.weak_or_unvisited_fraction
    )

    if not any_certificate_bad and objective_close:
        return {
            "classification": "not_failure",
            "reasons": ["standard mismatch and objective ratio are both within tolerance"],
        }
    if is_sidecar_improving_non_equivalent(
        sidecar_improved=sidecar_improved,
        equivalence_metrics_failed=equivalence_metrics_failed or any_certificate_bad,
    ):
        reasons.append(SIDECAR_IMPROVING_NON_EQUIVALENT)
    if representation_failed:
        reasons.append(REPRESENTATION_FAILURE)
    if io_map_bad:
        reasons.append(IO_MAP_MISMATCH)
    if certificate_bad and objective_close and weak_subspace_dominates:
        reasons.append("under_identification")
    if objective_bad or learned_gradient_bad:
        reasons.append("optimizer_basin")
    if certificate_bad and learned_stationary and reference_gradient_bad:
        reasons.append("objective_mismatch")

    unique_reasons = list(dict.fromkeys(reasons))
    if len(unique_reasons) == 1:
        classification: FailureClass = unique_reasons[0]  # type: ignore[assignment]
    elif len(unique_reasons) > 1:
        classification = "mixed"
    else:
        classification = "uncertain" if any_certificate_bad else "not_failure"
    return {
        "classification": classification,
        "reasons": unique_reasons,
        "signals": {
            "objective_close": objective_close,
            "objective_bad": objective_bad,
            "certificate_bad": certificate_bad,
            "io_map_bad": io_map_bad,
            "representation_failed": representation_failed,
            "learned_gradient_bad": learned_gradient_bad,
            "reference_gradient_bad": reference_gradient_bad,
            "weak_subspace_dominates": weak_subspace_dominates,
            "sidecar_improved": sidecar_improved,
            "equivalence_metrics_failed": equivalence_metrics_failed,
        },
    }


def is_sidecar_improving_non_equivalent(
    *,
    sidecar_improved: bool,
    equivalence_metrics_failed: bool,
) -> bool:
    """Return whether exact-L2/gamma sidecars improved without equivalence.

    This helper labels rows where sidecar diagnostics improve but the formal
    bridge gates still fail action, value, transition, or reference-equivalence
    metrics. The label is deliberately not a pass condition.
    """

    return bool(sidecar_improved and equivalence_metrics_failed)


def _match_covariances(covariances: np.ndarray, horizon: int, state_dim: int) -> np.ndarray:
    if covariances.shape == (horizon + 1, state_dim, state_dim):
        return covariances[:-1]
    if covariances.shape == (horizon, state_dim, state_dim):
        return covariances
    raise ValueError("state covariances must have shape (time, state, state)")


def _projected_fraction(
    delta: np.ndarray,
    basis: np.ndarray,
    total: float,
    numerics: FailureDecompositionNumerics,
) -> float:
    if basis.size == 0:
        return 0.0
    projected = delta @ basis @ basis.T
    return float(np.sum(projected**2) / max(total, numerics.denominator_floor))


def _ratio(numerator: float, denominator: float, floor: float) -> float:
    return float(numerator / max(denominator, floor))


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def _max(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.max([row[key] for row in rows]))


def _min(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.min([row[key] for row in rows]))


def _mean_sum(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float:
    return float(np.mean([sum(float(row[key]) for key in keys) for row in rows]))


__all__ = [
    "FailureClass",
    "FailureDecompositionNumerics",
    "IO_MAP_MISMATCH",
    "REPRESENTATION_FAILURE",
    "SIDECAR_IMPROVING_NON_EQUIVALENT",
    "classify_failure",
    "covariances_from_states",
    "gain_error_subspace_decomposition",
    "is_sidecar_improving_non_equivalent",
    "interpolation_curve",
    "objective_gradient_summary",
]
