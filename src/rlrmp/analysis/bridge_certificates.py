"""Reusable adapters for bridge standard-certificate component rows.

These helpers keep the bridge manifest contract small: callers provide common
NumPy-like arrays or metadata, and this module returns deterministic
``BridgeCertificateComponent`` rows with explicit status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from rlrmp.analysis.bridge_contracts import (
    BridgeArchitecture,
    BridgeCertificateMode,
    BridgeCertificateComponent,
)

ArrayLayout = Literal["batch_time_state", "time_batch_state"]

STATE_WEIGHTED_ACTION_MISMATCH = "state_weighted_action_mismatch"
CLOSED_LOOP_TRANSITION_MISMATCH = "closed_loop_transition_mismatch"
VALUE_POLICY_GAP = "value_policy_gap"
BELLMAN_HESSIAN_RESIDUAL = "bellman_hessian_residual"
VISITED_SUBSPACE_DIAGNOSTICS = "visited_subspace_diagnostics"
OPTIMIZER_METADATA = "optimizer_metadata"
RECURRENCE_GRU_DIAGNOSTICS = "recurrence_gru_diagnostics"

_RECURRENCE_ARCHITECTURES = {"linear_recurrence", "gru"}


@dataclass(frozen=True)
class CertificateNumerics:
    """Numerical tolerances used by bridge certificate adapters."""

    denominator_floor: float = 1e-12
    covariance_rank_rtol: float = 1e-8


def action_energy_mismatch_summary(
    *,
    candidate_actions: np.ndarray,
    reference_actions: np.ndarray,
    action_weight: np.ndarray | None = None,
    numerics: CertificateNumerics = CertificateNumerics(),
) -> dict[str, Any]:
    """Return timewise and aggregate action-energy mismatch summaries.

    Args:
        candidate_actions: Candidate actions with shape ``(time, batch, action)``.
        reference_actions: Reference actions with shape ``(time, batch, action)``.
        action_weight: Optional action metric with shape ``(action, action)`` or
            ``(time, action, action)``.
        numerics: Denominator floor used in ratios.
    """

    candidate = _as_float_array(candidate_actions)
    reference = _as_float_array(reference_actions)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference actions must have the same shape")
    if candidate.ndim != 3:
        raise ValueError("actions must have shape (time, batch, action)")

    delta = candidate - reference
    weights = _time_weight(action_weight, candidate.shape[0], candidate.shape[-1])
    delta_energy = _weighted_energy(delta, weights)
    reference_energy = _weighted_energy(reference, weights)
    ratios = delta_energy / np.maximum(reference_energy, numerics.denominator_floor)
    aggregate_delta_energy = float(np.sum(delta_energy))
    aggregate_reference_energy = float(np.sum(reference_energy))
    return {
        "delta_rms": _json_float(np.sqrt(np.mean(delta_energy))),
        "reference_rms": _json_float(np.sqrt(np.mean(reference_energy))),
        "mismatch_ratio_mean": _json_float(np.mean(ratios)),
        "mismatch_ratio_max": _json_float(np.max(ratios)),
        "aggregate_mismatch_ratio": _json_float(
            aggregate_delta_energy / max(aggregate_reference_energy, numerics.denominator_floor)
        ),
        "aggregate_delta_energy": _json_float(aggregate_delta_energy),
        "aggregate_reference_energy": _json_float(aggregate_reference_energy),
    }


def missing_component(name: str, reason: str) -> BridgeCertificateComponent:
    """Create an explicit missing certificate-component row."""

    return BridgeCertificateComponent(name=name, status="missing", reason=reason)


def state_weighted_action_mismatch_component(
    *,
    states: np.ndarray | None = None,
    candidate_actions: np.ndarray | None = None,
    reference_actions: np.ndarray | None = None,
    candidate_gain: np.ndarray | None = None,
    reference_gain: np.ndarray | None = None,
    action_weight: np.ndarray | None = None,
    layout: ArrayLayout = "batch_time_state",
    numerics: CertificateNumerics = CertificateNumerics(),
    name: str = STATE_WEIGHTED_ACTION_MISMATCH,
    state_label: str = "state",
    action_label: str = "action",
) -> BridgeCertificateComponent:
    """Return the state-weighted action mismatch row.

    Args:
        states: State samples used with gains. Shape is either
            ``(batch, horizon + 1, state)`` or ``(horizon + 1, batch, state)``.
        candidate_actions: Candidate action samples with the same layout as
            ``states`` but horizon-length time axis.
        reference_actions: Reference action samples.
        candidate_gain: Candidate gain matrices with shape ``(horizon, action, state)``.
        reference_gain: Reference gain matrices with shape ``(horizon, action, state)``.
        action_weight: Optional ``(action, action)`` or
            ``(horizon, action, action)`` positive-semidefinite weight.
        layout: Whether sample arrays are batch-first or time-first.
        numerics: Denominator floor and rank tolerances.
        name: Component row name.
        state_label: Human-readable state basis used for the summary.
        action_label: Human-readable action basis used for the summary.
    """

    candidate, reference = _action_pair(
        states=states,
        candidate_actions=candidate_actions,
        reference_actions=reference_actions,
        candidate_gain=candidate_gain,
        reference_gain=reference_gain,
        layout=layout,
    )
    if candidate is None or reference is None:
        return missing_component(
            name,
            "requires either candidate/reference actions or states plus candidate/reference gains",
        )
    summary = action_energy_mismatch_summary(
        candidate_actions=candidate,
        reference_actions=reference,
        action_weight=action_weight,
        numerics=numerics,
    )
    return BridgeCertificateComponent.available(
        name,
        **summary,
        n_samples=int(np.prod(candidate.shape[:2])),
        state_label=state_label,
        action_label=action_label,
    )


def closed_loop_transition_mismatch_component(
    *,
    states: np.ndarray | None,
    candidate_transition: np.ndarray | None,
    reference_transition: np.ndarray | None,
    layout: ArrayLayout = "batch_time_state",
    numerics: CertificateNumerics = CertificateNumerics(),
    name: str = CLOSED_LOOP_TRANSITION_MISMATCH,
    state_label: str = "state",
) -> BridgeCertificateComponent:
    """Return closed-loop transition mismatch over visited states."""

    if states is None or candidate_transition is None or reference_transition is None:
        return missing_component(
            name,
            "requires states plus candidate/reference closed-loop transition matrices",
        )
    x = _time_batch_states(states, layout)[:-1]
    candidate_matrix = _time_matrix(candidate_transition, x.shape[0], x.shape[-1], x.shape[-1])
    reference_matrix = _time_matrix(reference_transition, x.shape[0], x.shape[-1], x.shape[-1])
    delta_matrix = candidate_matrix - reference_matrix
    delta = np.einsum("tij,tbj->tbi", delta_matrix, x)
    reference = np.einsum("tij,tbj->tbi", reference_matrix, x)
    delta_energy = np.mean(np.sum(delta**2, axis=-1), axis=1)
    reference_energy = np.mean(np.sum(reference**2, axis=-1), axis=1)
    ratios = delta_energy / np.maximum(reference_energy, numerics.denominator_floor)
    return BridgeCertificateComponent.available(
        name,
        delta_rms=_json_float(np.sqrt(np.mean(delta_energy))),
        reference_rms=_json_float(np.sqrt(np.mean(reference_energy))),
        mismatch_ratio_mean=_json_float(np.mean(ratios)),
        mismatch_ratio_max=_json_float(np.max(ratios)),
        state_label=state_label,
    )


def value_policy_gap_component(
    *,
    candidate_value_matrices: np.ndarray | None,
    reference_value_matrices: np.ndarray | None,
    state_covariances: np.ndarray | None,
    numerics: CertificateNumerics = CertificateNumerics(),
    name: str = VALUE_POLICY_GAP,
    state_label: str = "state",
) -> BridgeCertificateComponent:
    """Return finite-horizon policy-evaluation value-gap ratios."""

    if (
        candidate_value_matrices is None
        or reference_value_matrices is None
        or state_covariances is None
    ):
        return missing_component(
            name,
            "requires candidate/reference value matrices and state covariances",
        )
    candidate = _as_float_array(candidate_value_matrices)
    reference = _as_float_array(reference_value_matrices)
    covariances = _match_covariances(state_covariances, candidate.shape[0], candidate.shape[-1])
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference value matrices must have the same shape")
    if candidate.ndim != 3 or candidate.shape[-1] != candidate.shape[-2]:
        raise ValueError("value matrices must have shape (time, state, state)")

    numerator = np.einsum("tij,tji->t", candidate - reference, covariances)
    denominator = np.einsum("tij,tji->t", reference, covariances)
    ratios = numerator / np.maximum(denominator, numerics.denominator_floor)
    return BridgeCertificateComponent.available(
        name,
        gap_ratio_mean=_json_float(np.mean(ratios)),
        gap_ratio_max=_json_float(np.max(ratios)),
        gap_ratio_max_abs=_json_float(np.max(np.abs(ratios))),
        state_label=state_label,
    )


def bellman_hessian_residual_component(
    *,
    gain_delta: np.ndarray | None,
    bellman_hessian: np.ndarray | None,
    state_covariances: np.ndarray | None,
    reference_gain: np.ndarray | None = None,
    numerics: CertificateNumerics = CertificateNumerics(),
    name: str = BELLMAN_HESSIAN_RESIDUAL,
    state_label: str = "state",
) -> BridgeCertificateComponent:
    """Return Bellman-Hessian-weighted gain/action residuals."""

    if gain_delta is None or bellman_hessian is None or state_covariances is None:
        return missing_component(
            name,
            "requires gain delta, Bellman action Hessians, and state covariances",
        )
    delta = _as_float_array(gain_delta)
    hessian = _time_matrix(bellman_hessian, delta.shape[0], delta.shape[1], delta.shape[1])
    covariances = _match_covariances(state_covariances, delta.shape[0], delta.shape[-1])
    if delta.ndim != 3:
        raise ValueError("gain_delta must have shape (time, action, state)")

    residual = np.einsum("tui,tuv,tvj,tij->t", delta, hessian, delta, covariances)
    summary: dict[str, Any] = {
        "residual_rms": _json_float(np.sqrt(np.mean(np.maximum(residual, 0.0)))),
        "residual_mean": _json_float(np.mean(residual)),
        "residual_max": _json_float(np.max(residual)),
        "state_label": state_label,
    }
    if reference_gain is not None:
        reference = _as_float_array(reference_gain)
        if reference.shape != delta.shape:
            raise ValueError("reference_gain must match gain_delta shape")
        reference_energy = np.einsum(
            "tui,tuv,tvj,tij->t", reference, hessian, reference, covariances
        )
        ratios = residual / np.maximum(reference_energy, numerics.denominator_floor)
        summary["reference_rms"] = _json_float(np.sqrt(np.mean(np.maximum(reference_energy, 0.0))))
        summary["residual_ratio_mean"] = _json_float(np.mean(ratios))
        summary["residual_ratio_max"] = _json_float(np.max(ratios))
    return BridgeCertificateComponent.available(name, **summary)


def visited_subspace_diagnostics_component(
    *,
    states: np.ndarray | None = None,
    state_covariances: np.ndarray | None = None,
    gain_delta: np.ndarray | None = None,
    layout: ArrayLayout = "batch_time_state",
    numerics: CertificateNumerics = CertificateNumerics(),
    name: str = VISITED_SUBSPACE_DIAGNOSTICS,
    state_label: str = "state",
) -> BridgeCertificateComponent:
    """Return effective-rank and visited-subspace diagnostics."""

    covariances = _covariances_or_none(states, state_covariances, layout)
    if covariances is None:
        return missing_component(name, "requires states or state covariances")
    singular_values = np.linalg.svd(covariances, compute_uv=False)
    max_s = singular_values[:, :1]
    identifiable = singular_values > np.maximum(
        max_s * numerics.covariance_rank_rtol,
        numerics.denominator_floor,
    )
    normalized = singular_values / np.maximum(
        np.sum(singular_values, axis=1, keepdims=True),
        numerics.denominator_floor,
    )
    normalized_for_log = np.where(normalized > 0.0, normalized, 1.0)
    entropy = -np.sum(
        np.where(normalized > 0.0, normalized * np.log(normalized_for_log), 0.0),
        axis=1,
    )
    effective_rank = np.exp(entropy)
    summary: dict[str, Any] = {
        "mean_effective_rank": _json_float(np.mean(effective_rank)),
        "min_effective_rank": _json_float(np.min(effective_rank)),
        "mean_identifiable_rank": _json_float(np.mean(np.sum(identifiable, axis=1))),
        "min_identifiable_rank": _json_float(np.min(np.sum(identifiable, axis=1))),
        "time_steps": int(covariances.shape[0]),
        "state_label": state_label,
    }
    if gain_delta is not None:
        fractions = _gain_error_fractions(gain_delta, covariances, numerics)
        summary.update(
            {
                "gain_error_parallel_fraction_mean": _json_float(np.mean(fractions[:, 0])),
                "gain_error_parallel_fraction_max": _json_float(np.max(fractions[:, 0])),
                "gain_error_orthogonal_fraction_mean": _json_float(np.mean(fractions[:, 1])),
                "gain_error_orthogonal_fraction_max": _json_float(np.max(fractions[:, 1])),
            }
        )
    return BridgeCertificateComponent.available(name, **summary)


def optimizer_metadata_component(
    metadata: dict[str, Any] | None,
    *,
    name: str = OPTIMIZER_METADATA,
) -> BridgeCertificateComponent:
    """Return optimizer/convergence metadata when available."""

    if not metadata:
        return missing_component(name, "no optimizer metadata was supplied")
    return BridgeCertificateComponent.available(name, **_json_summary(metadata))


def recurrence_safe_components(
    *,
    architecture: BridgeArchitecture,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[BridgeCertificateComponent, ...]:
    """Return honest rows for recurrence/GRU cases without formal linear claims."""

    if architecture not in _RECURRENCE_ARCHITECTURES:
        return ()
    reason = (
        f"{architecture} controllers do not define a global linear certificate "
        "without an explicit linearization or recurrence-state model"
    )
    diagnostic = (
        BridgeCertificateComponent.available(
            RECURRENCE_GRU_DIAGNOSTICS,
            architecture=architecture,
            **_json_summary(diagnostics),
        )
        if diagnostics
        else missing_component(
            RECURRENCE_GRU_DIAGNOSTICS,
            "no recurrence/GRU diagnostic metadata was supplied",
        )
    )
    return (
        BridgeCertificateComponent.not_applicable(CLOSED_LOOP_TRANSITION_MISMATCH, reason),
        BridgeCertificateComponent.not_applicable(VALUE_POLICY_GAP, reason),
        BridgeCertificateComponent.not_applicable(BELLMAN_HESSIAN_RESIDUAL, reason),
        diagnostic,
    )


def augmented_linear_recurrent_components(
    *,
    augmented_states: np.ndarray | None,
    candidate_action_sensitivity: np.ndarray | None = None,
    reference_action_sensitivity: np.ndarray | None = None,
    candidate_actions: np.ndarray | None = None,
    reference_actions: np.ndarray | None = None,
    action_weight: np.ndarray | None = None,
    candidate_transition: np.ndarray | None = None,
    reference_transition: np.ndarray | None = None,
    candidate_value_matrices: np.ndarray | None = None,
    reference_value_matrices: np.ndarray | None = None,
    bellman_hessian: np.ndarray | None = None,
    augmented_state_covariances: np.ndarray | None = None,
    optimizer_metadata: dict[str, Any] | None = None,
    recurrence_diagnostics: dict[str, Any] | None = None,
    layout: ArrayLayout = "batch_time_state",
    numerics: CertificateNumerics = CertificateNumerics(),
    state_label: str = "augmented_state",
    action_label: str = "action",
) -> tuple[BridgeCertificateComponent, ...]:
    """Build formal certificate rows for a linear recurrent augmented state.

    ``augmented_states`` is the state used by the linear recurrence certificate,
    typically ``z_t = [x_t; h_t]`` or a plant/estimator/hidden-state stack.
    Candidate/reference action sensitivities are linear maps from that
    augmented state to action with shape ``(time, action, augmented_state)``.
    """

    covariances = _covariances_or_none(augmented_states, augmented_state_covariances, layout)
    gain_delta = (
        None
        if candidate_action_sensitivity is None or reference_action_sensitivity is None
        else _as_float_array(candidate_action_sensitivity)
        - _as_float_array(reference_action_sensitivity)
    )
    return (
        state_weighted_action_mismatch_component(
            states=augmented_states,
            candidate_actions=candidate_actions,
            reference_actions=reference_actions,
            candidate_gain=candidate_action_sensitivity,
            reference_gain=reference_action_sensitivity,
            action_weight=action_weight,
            layout=layout,
            numerics=numerics,
            state_label=state_label,
            action_label=action_label,
        ),
        visited_subspace_diagnostics_component(
            states=augmented_states,
            state_covariances=covariances,
            gain_delta=gain_delta,
            layout=layout,
            numerics=numerics,
            state_label=state_label,
        ),
        optimizer_metadata_component(optimizer_metadata),
        closed_loop_transition_mismatch_component(
            states=augmented_states,
            candidate_transition=candidate_transition,
            reference_transition=reference_transition,
            layout=layout,
            numerics=numerics,
            state_label=state_label,
        ),
        value_policy_gap_component(
            candidate_value_matrices=candidate_value_matrices,
            reference_value_matrices=reference_value_matrices,
            state_covariances=covariances,
            numerics=numerics,
            state_label=state_label,
        ),
        bellman_hessian_residual_component(
            gain_delta=gain_delta,
            bellman_hessian=bellman_hessian,
            state_covariances=covariances,
            reference_gain=reference_action_sensitivity,
            numerics=numerics,
            state_label=state_label,
        ),
        (
            BridgeCertificateComponent.available(
                RECURRENCE_GRU_DIAGNOSTICS,
                certificate_mode="augmented_linear",
                **_json_summary(recurrence_diagnostics),
            )
            if recurrence_diagnostics
            else missing_component(
                RECURRENCE_GRU_DIAGNOSTICS,
                "no augmented linear recurrence diagnostic metadata was supplied",
            )
        ),
    )


def build_standard_certificate_components(
    *,
    architecture: BridgeArchitecture,
    certificate_mode: BridgeCertificateMode | None = None,
    states: np.ndarray | None = None,
    action_states: np.ndarray | None = None,
    augmented_states: np.ndarray | None = None,
    candidate_actions: np.ndarray | None = None,
    reference_actions: np.ndarray | None = None,
    candidate_gain: np.ndarray | None = None,
    reference_gain: np.ndarray | None = None,
    candidate_augmented_action_sensitivity: np.ndarray | None = None,
    reference_augmented_action_sensitivity: np.ndarray | None = None,
    action_weight: np.ndarray | None = None,
    candidate_transition: np.ndarray | None = None,
    reference_transition: np.ndarray | None = None,
    candidate_value_matrices: np.ndarray | None = None,
    reference_value_matrices: np.ndarray | None = None,
    bellman_hessian: np.ndarray | None = None,
    state_covariances: np.ndarray | None = None,
    augmented_state_covariances: np.ndarray | None = None,
    optimizer_metadata: dict[str, Any] | None = None,
    recurrence_diagnostics: dict[str, Any] | None = None,
    layout: ArrayLayout = "batch_time_state",
    numerics: CertificateNumerics = CertificateNumerics(),
    state_label: str = "state",
    action_state_label: str = "state",
    action_label: str = "action",
) -> tuple[BridgeCertificateComponent, ...]:
    """Build the standard bridge certificate component bundle.

    The same entry point covers full-state, output-feedback linear, and
    augmented-linear recurrent cases. For output-feedback linear controllers,
    pass ``states`` as the coupled closed-loop state (for transition/value/rank
    rows) and ``action_states`` as the estimated state used by the controller
    gain. For linear recurrent controllers with a formal recurrence-state model,
    set ``certificate_mode="augmented_linear"`` and provide ``augmented_states``
    plus augmented action/transition sensitivities.
    """

    mode = _resolve_certificate_mode(
        architecture=architecture,
        certificate_mode=certificate_mode,
        augmented_states=augmented_states,
        candidate_augmented_action_sensitivity=candidate_augmented_action_sensitivity,
        reference_augmented_action_sensitivity=reference_augmented_action_sensitivity,
    )
    if mode == "augmented_linear":
        return augmented_linear_recurrent_components(
            augmented_states=augmented_states,
            candidate_action_sensitivity=candidate_augmented_action_sensitivity,
            reference_action_sensitivity=reference_augmented_action_sensitivity,
            candidate_actions=candidate_actions,
            reference_actions=reference_actions,
            action_weight=action_weight,
            candidate_transition=candidate_transition,
            reference_transition=reference_transition,
            candidate_value_matrices=candidate_value_matrices,
            reference_value_matrices=reference_value_matrices,
            bellman_hessian=bellman_hessian,
            augmented_state_covariances=augmented_state_covariances,
            optimizer_metadata=optimizer_metadata,
            recurrence_diagnostics=recurrence_diagnostics,
            layout=layout,
            numerics=numerics,
            state_label="augmented_state" if state_label == "state" else state_label,
            action_label=action_label,
        )

    covariances = _covariances_or_none(states, state_covariances, layout)
    action_covariances = (
        _covariances_or_none(action_states, None, layout)
        if action_states is not None
        else covariances
    )
    gain_delta = (
        None
        if candidate_gain is None or reference_gain is None
        else _as_float_array(candidate_gain) - _as_float_array(reference_gain)
    )
    visited_gain_delta = (
        gain_delta
        if gain_delta is not None
        and covariances is not None
        and covariances.shape[-1] == gain_delta.shape[-1]
        else None
    )
    components: list[BridgeCertificateComponent] = [
        state_weighted_action_mismatch_component(
            states=action_states if action_states is not None else states,
            candidate_actions=candidate_actions,
            reference_actions=reference_actions,
            candidate_gain=candidate_gain,
            reference_gain=reference_gain,
            action_weight=action_weight,
            layout=layout,
            numerics=numerics,
            state_label=action_state_label,
            action_label=action_label,
        ),
        visited_subspace_diagnostics_component(
            states=states,
            state_covariances=covariances,
            gain_delta=visited_gain_delta,
            layout=layout,
            numerics=numerics,
            state_label=state_label,
        ),
        optimizer_metadata_component(optimizer_metadata),
    ]
    if architecture in _RECURRENCE_ARCHITECTURES:
        components.extend(
            recurrence_safe_components(
                architecture=architecture,
                diagnostics=recurrence_diagnostics,
            )
        )
        return tuple(components)

    components.extend(
        [
            closed_loop_transition_mismatch_component(
                states=states,
                candidate_transition=candidate_transition,
                reference_transition=reference_transition,
                layout=layout,
                numerics=numerics,
                state_label=state_label,
            ),
            value_policy_gap_component(
                candidate_value_matrices=candidate_value_matrices,
                reference_value_matrices=reference_value_matrices,
                state_covariances=covariances,
                numerics=numerics,
                state_label=state_label,
            ),
            bellman_hessian_residual_component(
                gain_delta=gain_delta,
                bellman_hessian=bellman_hessian,
                state_covariances=action_covariances,
                reference_gain=reference_gain,
                numerics=numerics,
                state_label=action_state_label,
            ),
        ]
    )
    return tuple(components)


def _resolve_certificate_mode(
    *,
    architecture: BridgeArchitecture,
    certificate_mode: BridgeCertificateMode | None,
    augmented_states: np.ndarray | None,
    candidate_augmented_action_sensitivity: np.ndarray | None,
    reference_augmented_action_sensitivity: np.ndarray | None,
) -> BridgeCertificateMode:
    if certificate_mode is not None:
        if certificate_mode == "augmented_linear" and architecture == "gru":
            raise ValueError("GRU rows cannot claim augmented_linear certificate mode")
        return certificate_mode
    if (
        architecture == "linear_recurrence"
        and augmented_states is not None
        and candidate_augmented_action_sensitivity is not None
        and reference_augmented_action_sensitivity is not None
    ):
        return "augmented_linear"
    if architecture in _RECURRENCE_ARCHITECTURES:
        return "empirical_nonlinear"
    return "static_gain"


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _json_float(value: float | np.floating[Any]) -> float:
    return float(np.asarray(value))


def _json_summary(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {str(key): _json_value(value) for key, value in metadata.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return _json_summary(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _time_batch_states(states: np.ndarray, layout: ArrayLayout) -> np.ndarray:
    values = _as_float_array(states)
    if values.ndim != 3:
        raise ValueError("states must have shape (batch, time, state) or (time, batch, state)")
    if layout == "batch_time_state":
        return np.swapaxes(values, 0, 1)
    return values


def _time_batch_actions(actions: np.ndarray, layout: ArrayLayout) -> np.ndarray:
    values = _as_float_array(actions)
    if values.ndim != 3:
        raise ValueError(
            "actions must have shape (batch, horizon, action) or (time, batch, action)"
        )
    if layout == "batch_time_state":
        return np.swapaxes(values, 0, 1)
    return values


def _action_pair(
    *,
    states: np.ndarray | None,
    candidate_actions: np.ndarray | None,
    reference_actions: np.ndarray | None,
    candidate_gain: np.ndarray | None,
    reference_gain: np.ndarray | None,
    layout: ArrayLayout,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if candidate_actions is not None and reference_actions is not None:
        return (
            _time_batch_actions(candidate_actions, layout),
            _time_batch_actions(reference_actions, layout),
        )
    if states is None or candidate_gain is None or reference_gain is None:
        return None, None
    x = _time_batch_states(states, layout)
    candidate_matrix = _as_float_array(candidate_gain)
    reference_matrix = _as_float_array(reference_gain)
    if candidate_matrix.shape != reference_matrix.shape:
        raise ValueError("candidate and reference gains must have the same shape")
    if candidate_matrix.ndim != 3:
        raise ValueError("gains must have shape (time, action, state)")
    if x.shape[0] != candidate_matrix.shape[0] + 1:
        raise ValueError("states must have exactly one more time sample than gains")
    if x.shape[-1] != candidate_matrix.shape[-1]:
        raise ValueError("state dimension must match gain input dimension")
    return (
        np.einsum("tuj,tbj->tbu", candidate_matrix, x[:-1]),
        np.einsum("tuj,tbj->tbu", reference_matrix, x[:-1]),
    )


def _time_weight(weight: np.ndarray | None, horizon: int, dim: int) -> np.ndarray:
    if weight is None:
        return np.broadcast_to(np.eye(dim, dtype=float), (horizon, dim, dim))
    values = _as_float_array(weight)
    if values.shape == (dim, dim):
        return np.broadcast_to(values, (horizon, dim, dim))
    if values.shape == (horizon, dim, dim):
        return values
    raise ValueError("weight must have shape (dim, dim) or (horizon, dim, dim)")


def _time_matrix(matrix: np.ndarray, horizon: int, rows: int, cols: int) -> np.ndarray:
    values = _as_float_array(matrix)
    if values.shape == (rows, cols):
        return np.broadcast_to(values, (horizon, rows, cols))
    if values.shape == (horizon, rows, cols):
        return values
    raise ValueError("matrix must have shape (rows, cols) or (horizon, rows, cols)")


def _weighted_energy(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.mean(np.einsum("tbi,tij,tbj->tb", values, weights, values), axis=1)


def _covariances_or_none(
    states: np.ndarray | None,
    state_covariances: np.ndarray | None,
    layout: ArrayLayout,
) -> np.ndarray | None:
    if state_covariances is not None:
        covariances = _as_float_array(state_covariances)
        if covariances.ndim != 3 or covariances.shape[-1] != covariances.shape[-2]:
            raise ValueError("state_covariances must have shape (time, state, state)")
        return covariances
    if states is None:
        return None
    x = _time_batch_states(states, layout)
    return np.einsum("tbi,tbj->tij", x, x) / x.shape[1]


def _match_covariances(covariances: np.ndarray, horizon: int, state_dim: int) -> np.ndarray:
    values = _as_float_array(covariances)
    if values.shape == (horizon + 1, state_dim, state_dim):
        return values[:-1]
    if values.shape == (horizon, state_dim, state_dim):
        return values
    raise ValueError("state covariances must have shape (time, state, state)")


def _gain_error_fractions(
    gain_delta: np.ndarray,
    covariances: np.ndarray,
    numerics: CertificateNumerics,
) -> np.ndarray:
    delta = _as_float_array(gain_delta)
    cov = _match_covariances(covariances, delta.shape[0], delta.shape[-1])
    fractions = []
    for cov_t, delta_t in zip(cov, delta, strict=True):
        u, s, _vh = np.linalg.svd(cov_t, full_matrices=False)
        keep = s > np.maximum(np.max(s) * numerics.covariance_rank_rtol, numerics.denominator_floor)
        basis = u * keep[None, :]
        projection = basis @ basis.T
        parallel = delta_t @ projection
        total = np.sum(delta_t**2)
        parallel_fraction = np.sum(parallel**2) / max(total, numerics.denominator_floor)
        fractions.append(
            [
                float(parallel_fraction),
                float(max(0.0, 1.0 - parallel_fraction)),
            ]
        )
    return np.asarray(fractions)


__all__ = [
    "BELLMAN_HESSIAN_RESIDUAL",
    "CLOSED_LOOP_TRANSITION_MISMATCH",
    "OPTIMIZER_METADATA",
    "RECURRENCE_GRU_DIAGNOSTICS",
    "STATE_WEIGHTED_ACTION_MISMATCH",
    "VALUE_POLICY_GAP",
    "VISITED_SUBSPACE_DIAGNOSTICS",
    "ArrayLayout",
    "CertificateNumerics",
    "action_energy_mismatch_summary",
    "augmented_linear_recurrent_components",
    "bellman_hessian_residual_component",
    "build_standard_certificate_components",
    "closed_loop_transition_mismatch_component",
    "missing_component",
    "optimizer_metadata_component",
    "recurrence_safe_components",
    "state_weighted_action_mismatch_component",
    "value_policy_gap_component",
    "visited_subspace_diagnostics_component",
]
