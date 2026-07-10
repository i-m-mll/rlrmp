"""LEGACY (frozen 2026-07-03, issue 64d5f13).

This materializer is not contract-native: it predates the feedbax recipe,
bundle, and manifest contracts. It may not run without deliberate realignment.
Do not copy it as a pattern for new analyses. The port-or-delete decision is
deferred to the report-stage era (feedbax 132f98c) / publication.

Reusable delayed-reach direction-split and peak-decay diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.analysis.pipelines.diagnostic_provenance import write_regeneration_spec
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.runtime.spec_migrations import (
    DELAYED_DIAGNOSTIC_BUNDLE_KIND,
    DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
    stamp_current_schema,
)


SCHEMA_VERSION = DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION
_ANALYSIS_PRESET = load_analysis_parameter_preset("delayed_diagnostic_bundle").parameters
DEFAULT_DECAY_THRESHOLDS = tuple(_ANALYSIS_PRESET["decay_thresholds"])
DEFAULT_SUPPORT_WINDOWS = tuple(tuple(window) for window in _ANALYSIS_PRESET["support_windows"])
SignalRole = Literal["command", "force_filter", "efferent", "acceleration", "velocity", "other"]


@dataclass(frozen=True)
class DirectionGroupSpec:
    """Named direction-index group for delayed-bank split summaries."""

    name: str
    direction_indices: tuple[int, ...]
    label: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return JSON-compatible group metadata."""

        return {
            "name": self.name,
            "label": self.label or self.name,
            "direction_indices": [int(index) for index in self.direction_indices],
        }


@dataclass(frozen=True)
class DecaySignalSpec:
    """Signal metadata for peak/support-decay summaries."""

    name: str
    role: SignalRole
    source: str
    units: str
    baseline_window: tuple[int, int] = (0, 5)
    threshold_start_step: int | None = None

    def to_json(self) -> dict[str, Any]:
        """Return JSON-compatible signal metadata."""

        return {
            "name": self.name,
            "role": self.role,
            "source": self.source,
            "units": self.units,
            "baseline_window": [int(self.baseline_window[0]), int(self.baseline_window[1])],
            "threshold_start_step": (
                None if self.threshold_start_step is None else int(self.threshold_start_step)
            ),
        }


def build_delayed_diagnostic_bundle(
    *,
    issue: str,
    scope: str,
    direction_split: Mapping[str, Any] | None = None,
    peak_decay: Mapping[str, Any] | None = None,
    checkpoint_policy: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a schema-stamped delayed diagnostic bundle payload.

    The bundle is intentionally diagnostic-only. It does not select checkpoints,
    does not imply a standard-certificate pass/fail result, and does not own the
    fixed delayed-bank scorer that produces the arrays it consumes.
    """

    direction_payload = (
        _not_materialized("direction_split", "no_direction_split_payload_supplied")
        if direction_split is None
        else dict(direction_split)
    )
    peak_payload = (
        _not_materialized("peak_decay", "no_peak_decay_payload_supplied")
        if peak_decay is None
        else dict(peak_decay)
    )
    return stamp_current_schema(
        DELAYED_DIAGNOSTIC_BUNDLE_KIND,
        {
            "issue": issue,
            "scope": scope,
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
            "checkpoint_policy": checkpoint_policy or "external",
            "definitions": delayed_diagnostic_definitions(),
            "context": dict(context or {}),
            "direction_split": direction_payload,
            "peak_decay": peak_payload,
        },
    )


def materialize_delayed_diagnostic_bundle(
    *,
    issue: str,
    scope: str,
    output_path: Path,
    direction_split: Mapping[str, Any] | None = None,
    peak_decay: Mapping[str, Any] | None = None,
    checkpoint_policy: str | None = None,
    context: Mapping[str, Any] | None = None,
    regeneration_spec_path: Path | None = None,
    source_inputs: Sequence[Mapping[str, Any]] = (),
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Write a delayed diagnostic bundle manifest and regeneration spec."""

    payload = build_delayed_diagnostic_bundle(
        issue=issue,
        scope=scope,
        direction_split=direction_split,
        peak_decay=peak_decay,
        checkpoint_policy=checkpoint_policy,
        context=context,
    )
    mkdir_p(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    regeneration_spec_path = regeneration_spec_path or output_path.with_name(
        f"{output_path.stem}_regeneration_spec.json"
    )
    write_regeneration_spec(
        spec_path=regeneration_spec_path,
        diagnostic_name="delayed_diagnostic_bundle",
        materializer="rlrmp.analysis.pipelines.delayed_diagnostic_bundle.materialize_delayed_diagnostic_bundle",
        command=None,
        parameters={
            "issue": issue,
            "scope": scope,
            "checkpoint_policy": checkpoint_policy,
            "context": dict(context or {}),
        },
        inputs=list(source_inputs),
        outputs=[{"role": "delayed_diagnostic_bundle_manifest", "path": output_path}],
        source_files=[
            "src/rlrmp/analysis/pipelines/delayed_diagnostic_bundle.py",
            "src/rlrmp/runtime/spec_migrations.py",
        ],
        notes=[
            "Direction split is only meaningful for multi-direction delayed banks.",
            "Peak/support decay is a diagnostic sidecar, not a certificate gate.",
        ],
        repo_root=repo_root,
    )
    return payload


def summarize_direction_split(
    *,
    velocity: Any,
    direction_index: Any | None,
    direction_groups: Sequence[DirectionGroupSpec | Mapping[str, Any]],
    dt: float,
    reach_direction: Any | None = None,
    target_position: Any | None = None,
    initial_position: Any | None = None,
    time_zero_step: int = 0,
    bank_metadata: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize target-radial velocity by declared direction-index groups.

    Args:
        velocity: Rollout velocity with shape ``(..., trial, time, 2)``.
        direction_index: Direction index per trial, shape ``(trial,)`` or
            broadcastable to the leading sample axes plus trial.
        direction_groups: Named sets of direction indices.
        dt: Seconds per time step.
        reach_direction: Optional reach direction vectors. If omitted, they are
            inferred from ``target_position - initial_position``.
        target_position: Optional target positions used to infer directions.
        initial_position: Optional initial positions used to infer directions.
        time_zero_step: Step index treated as time zero in the returned axis.
        bank_metadata: Optional delayed-bank metadata.
        context: Optional caller provenance.

    Returns:
        JSON-compatible direction-split diagnostic. Single-direction or missing
        direction-index contexts return ``not_applicable``.
    """

    velocity_array = np.asarray(velocity, dtype=np.float64)
    if velocity_array.ndim < 3 or velocity_array.shape[-1] != 2:
        raise ValueError(
            f"velocity must have shape (..., trial, time, 2); got {velocity_array.shape}"
        )
    n_trials = int(velocity_array.shape[-3])
    n_steps = int(velocity_array.shape[-2])
    if direction_index is None:
        return _not_applicable(
            "direction_split",
            "missing_direction_index",
            "direction split requires a multi-direction delayed-bank direction index",
            context=context,
            bank_metadata=bank_metadata,
        )
    direction_index_array = _broadcast_trial_vector(
        direction_index,
        leading_shape=velocity_array.shape[:-3],
        n_trials=n_trials,
        name="direction_index",
    ).astype(np.int64)
    unique_directions = sorted({int(index) for index in direction_index_array.reshape(-1)})
    if len(unique_directions) < 2:
        return _not_applicable(
            "direction_split",
            "single_direction_context",
            "single-direction or fixed-target evaluations do not support direction splits",
            context=context,
            bank_metadata=bank_metadata,
        )
    groups = [_coerce_direction_group(group) for group in direction_groups]
    if not groups:
        raise ValueError("direction_groups must contain at least one group")

    directions = _resolve_reach_directions(
        velocity_array.shape[:-2],
        reach_direction=reach_direction,
        target_position=target_position,
        initial_position=initial_position,
    )
    unit = _unit_vectors(directions)
    forward_velocity = np.sum(velocity_array * unit[..., None, :], axis=-1)
    flat_forward = forward_velocity.reshape(-1, n_steps)
    flat_direction = direction_index_array.reshape(-1)
    time_s = (np.arange(n_steps, dtype=np.float64) - int(time_zero_step)) * float(dt)
    by_group = {}
    for group in groups:
        group_indices = {int(index) for index in group.direction_indices}
        mask = np.asarray([int(index) in group_indices for index in flat_direction])
        if not mask.any():
            by_group[group.name] = {
                **group.to_json(),
                "status": "not_applicable",
                "reason": "no_trials_for_direction_group",
                "n_samples": 0,
            }
            continue
        samples = flat_forward[mask]
        mean = np.mean(samples, axis=0)
        peak_idx = int(np.argmax(mean))
        by_group[group.name] = {
            **group.to_json(),
            "status": "available",
            "n_samples": int(samples.shape[0]),
            "time_s": time_s.tolist(),
            "mean_forward_velocity_m_s": mean.tolist(),
            "std_forward_velocity_m_s": np.std(samples, axis=0).tolist(),
            "peak_mean_forward_velocity_m_s": float(mean[peak_idx]),
            "time_of_peak_mean_forward_velocity_s": float(time_s[peak_idx]),
        }
    return {
        "status": "available",
        "scope": "multi_direction_target_radial_velocity_split",
        "basis": "target_radial_forward_velocity",
        "not_applicable_contexts": [
            "single_direction_context",
            "fixed_target_without_direction_index",
        ],
        "direction_index_unique": unique_directions,
        "n_trials": n_trials,
        "n_time_steps": n_steps,
        "dt_s": float(dt),
        "time_zero_step": int(time_zero_step),
        "bank_metadata": dict(bank_metadata or {}),
        "context": dict(context or {}),
        "groups": by_group,
    }


def summarize_peak_decay(
    *,
    signals: Mapping[str, Any],
    signal_specs: Sequence[DecaySignalSpec | Mapping[str, Any]],
    dt: float,
    thresholds: Sequence[float] = DEFAULT_DECAY_THRESHOLDS,
    support_windows: Sequence[tuple[int, int]] = DEFAULT_SUPPORT_WINDOWS,
    checkpoint_signals: Mapping[str, Mapping[str, Any]] | None = None,
    reference_signals: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize peak timing and post-peak support decay for delayed rollouts."""

    specs = [_coerce_signal_spec(spec) for spec in signal_specs]
    if not specs:
        raise ValueError("signal_specs must contain at least one signal")
    missing = [spec.name for spec in specs if spec.name not in signals]
    if missing:
        raise KeyError(f"missing signal profiles for peak decay: {missing}")
    threshold_tuple = tuple(float(value) for value in thresholds)
    support_window_tuple = tuple((int(start), int(stop)) for start, stop in support_windows)
    final_signals = {
        spec.name: summarize_decay_profile(
            signals[spec.name],
            spec=spec,
            dt=dt,
            thresholds=threshold_tuple,
            support_windows=support_window_tuple,
            reference_profile=None
            if reference_signals is None
            else reference_signals.get(spec.name),
        )
        for spec in specs
    }
    checkpoint_rows = []
    for checkpoint_label, checkpoint_payload in (checkpoint_signals or {}).items():
        row: dict[str, Any] = {"checkpoint": str(checkpoint_label), "signals": {}}
        for spec in specs:
            if spec.name not in checkpoint_payload:
                row["signals"][spec.name] = {
                    "status": "not_materialized",
                    "reason": "signal_absent_for_checkpoint",
                    "signal": spec.to_json(),
                }
                continue
            row["signals"][spec.name] = summarize_decay_profile(
                checkpoint_payload[spec.name],
                spec=spec,
                dt=dt,
                thresholds=threshold_tuple,
                support_windows=support_window_tuple,
                reference_profile=(
                    None if reference_signals is None else reference_signals.get(spec.name)
                ),
            )
        checkpoint_rows.append(row)
    return {
        "status": "available",
        "scope": "delayed_reach_peak_and_support_decay",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "definitions": {
            "peak": "maximum of the mean scalar signal profile over the evaluated window",
            "decay_crossing": (
                "first post-peak step at or after threshold_start_step whose mean signal "
                "falls below threshold * peak"
            ),
            "support_window": "mean positive support in the inclusive/exclusive step window",
            "checkpoint_sweep": (
                "same summaries over caller-supplied checkpoint profiles; the diagnostic "
                "does not select checkpoints"
            ),
        },
        "dt_s": float(dt),
        "thresholds": list(threshold_tuple),
        "support_windows": [
            {"start_step": int(start), "stop_step": int(stop)}
            for start, stop in support_window_tuple
        ],
        "context": dict(context or {}),
        "signals": final_signals,
        "checkpoint_sweep": {
            "status": "available" if checkpoint_rows else "not_materialized",
            "rows": checkpoint_rows,
        },
    }


def summarize_decay_profile(
    profile: Any,
    *,
    spec: DecaySignalSpec,
    dt: float,
    thresholds: Sequence[float] = DEFAULT_DECAY_THRESHOLDS,
    support_windows: Sequence[tuple[int, int]] = DEFAULT_SUPPORT_WINDOWS,
    reference_profile: Any | None = None,
) -> dict[str, Any]:
    """Summarize one scalar or vector signal profile."""

    samples = _as_sample_time_profile(profile, name=spec.name)
    mean = np.mean(samples, axis=0)
    peak_step = int(np.argmax(mean))
    peak_value = float(mean[peak_step])
    threshold_start = int(
        spec.threshold_start_step if spec.threshold_start_step is not None else peak_step
    )
    threshold_start = max(threshold_start, peak_step)
    decay_crossings = {}
    for threshold in thresholds:
        crossing_step = _first_decay_crossing(
            mean,
            peak_value=peak_value,
            threshold=float(threshold),
            start_step=threshold_start,
        )
        decay_crossings[str(float(threshold))] = {
            "threshold_fraction_of_peak": float(threshold),
            "step": crossing_step,
            "time_s": None if crossing_step is None else float(crossing_step * dt),
        }
    baseline = _window_mean(mean, spec.baseline_window)
    support = {}
    for start, stop in support_windows:
        support[f"{start}:{stop}"] = {
            "start_step": int(start),
            "stop_step": int(stop),
            "mean_positive_support": float(np.mean(np.maximum(mean[start:stop], 0.0))),
            "mean_abs_support": float(np.mean(np.abs(mean[start:stop]))),
            "fraction_of_baseline": _safe_ratio(_window_mean(mean, (start, stop)), baseline),
        }
    payload = {
        "status": "available",
        "signal": spec.to_json(),
        "n_samples": int(samples.shape[0]),
        "n_time_steps": int(samples.shape[1]),
        "mean_profile": mean.tolist(),
        "std_profile": np.std(samples, axis=0).tolist(),
        "peak_value": peak_value,
        "peak_step": peak_step,
        "peak_time_s": float(peak_step * dt),
        "baseline_mean": float(baseline),
        "decay_crossings": decay_crossings,
        "support_windows": support,
    }
    if reference_profile is not None:
        reference = np.mean(_as_sample_time_profile(reference_profile, name=spec.name), axis=0)
        common = min(reference.shape[0], mean.shape[0])
        payload["reference_comparison"] = {
            "status": "available",
            "rmse": float(np.sqrt(np.mean((mean[:common] - reference[:common]) ** 2))),
            "peak_value": float(reference[int(np.argmax(reference))]),
            "peak_step": int(np.argmax(reference)),
        }
    return payload


def delayed_diagnostic_definitions() -> dict[str, str]:
    """Return bundle-level definitions for delayed diagnostics."""

    return {
        "direction_split": (
            "Target-radial forward velocity profiles split by declared delayed-bank "
            "direction indices. It is not meaningful for a single target direction."
        ),
        "command": "Controller output before downstream motor/efferent filtering.",
        "force_filter": (
            "Plant-side force/filter state or equivalent caller-declared support signal; "
            "valid only when the source graph exposes the signal."
        ),
        "efferent": "Caller-declared efferent support signal when distinct from command.",
        "acceleration": (
            "Acceleration profile supplied by the caller or derived from velocity outside "
            "this bundle; source and filtering must be declared in the signal spec."
        ),
        "velocity": "Target-radial or scalar velocity profile supplied by the caller.",
        "checkpoint_sweep": (
            "Per-checkpoint diagnostic rows over the same fixed delayed-bank context; "
            "audit-only and separate from checkpoint-selection policy."
        ),
    }


def _not_materialized(component: str, reason: str) -> dict[str, Any]:
    return {
        "status": "not_materialized",
        "component": component,
        "reason": reason,
    }


def _not_applicable(
    component: str,
    reason: str,
    note: str,
    *,
    context: Mapping[str, Any] | None = None,
    bank_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "component": component,
        "reason": reason,
        "note": note,
        "context": dict(context or {}),
        "bank_metadata": dict(bank_metadata or {}),
    }


def _coerce_direction_group(group: DirectionGroupSpec | Mapping[str, Any]) -> DirectionGroupSpec:
    if isinstance(group, DirectionGroupSpec):
        return group
    return DirectionGroupSpec(
        name=str(group["name"]),
        label=None if group.get("label") is None else str(group["label"]),
        direction_indices=tuple(int(index) for index in group["direction_indices"]),
    )


def _coerce_signal_spec(spec: DecaySignalSpec | Mapping[str, Any]) -> DecaySignalSpec:
    if isinstance(spec, DecaySignalSpec):
        return spec
    baseline_window = spec.get("baseline_window", (0, 5))
    return DecaySignalSpec(
        name=str(spec["name"]),
        role=str(spec.get("role", "other")),  # type: ignore[arg-type]
        source=str(spec.get("source", "caller_supplied")),
        units=str(spec.get("units", "arbitrary")),
        baseline_window=(int(baseline_window[0]), int(baseline_window[1])),
        threshold_start_step=(
            None if spec.get("threshold_start_step") is None else int(spec["threshold_start_step"])
        ),
    )


def _broadcast_trial_vector(
    values: Any, *, leading_shape: tuple[int, ...], n_trials: int, name: str
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        if array.shape[0] != n_trials:
            raise ValueError(f"{name} length {array.shape[0]} does not match n_trials {n_trials}")
        return np.broadcast_to(array, leading_shape + (n_trials,))
    if array.shape[-1] != n_trials:
        raise ValueError(f"{name} trailing length {array.shape[-1]} does not match {n_trials}")
    return np.broadcast_to(array, leading_shape + (n_trials,))


def _resolve_reach_directions(
    sample_trial_shape: tuple[int, ...],
    *,
    reach_direction: Any | None,
    target_position: Any | None,
    initial_position: Any | None,
) -> np.ndarray:
    if reach_direction is not None:
        direction = np.asarray(reach_direction, dtype=np.float64)
    elif target_position is not None and initial_position is not None:
        target = np.asarray(target_position, dtype=np.float64)
        initial = np.asarray(initial_position, dtype=np.float64)
        direction = target - initial
    else:
        raise ValueError(
            "direction split requires reach_direction or target_position and initial_position"
        )
    if direction.shape[-1] != 2:
        raise ValueError(f"reach direction must end in dimension 2; got {direction.shape}")
    return np.broadcast_to(direction, sample_trial_shape + (2,))


def _unit_vectors(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("reach direction vectors must be nonzero")
    return vectors / norm


def _as_sample_time_profile(profile: Any, *, name: str) -> np.ndarray:
    array = np.asarray(profile, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError(f"{name} profile must have at least one dimension")
    if array.ndim >= 2 and array.shape[-1] in {2, 8}:
        array = np.linalg.norm(array, axis=-1)
    if array.ndim == 1:
        return array[None, :]
    return array.reshape(-1, array.shape[-1])


def _first_decay_crossing(
    mean: np.ndarray,
    *,
    peak_value: float,
    threshold: float,
    start_step: int,
) -> int | None:
    if peak_value <= 0.0:
        return None
    cutoff = threshold * peak_value
    for step in range(max(0, start_step), mean.shape[0]):
        if float(mean[step]) <= cutoff:
            return int(step)
    return None


def _window_mean(profile: np.ndarray, window: tuple[int, int]) -> float:
    start, stop = int(window[0]), int(window[1])
    if start < 0 or stop <= start or start >= profile.shape[0]:
        raise ValueError(f"invalid support window {window} for profile length {profile.shape[0]}")
    return float(np.mean(profile[start : min(stop, profile.shape[0])]))


def _safe_ratio(value: float, denominator: float) -> float | None:
    if abs(float(denominator)) <= 1e-12:
        return None
    return float(value / denominator)


__all__ = [
    "DEFAULT_DECAY_THRESHOLDS",
    "DEFAULT_SUPPORT_WINDOWS",
    "DecaySignalSpec",
    "DirectionGroupSpec",
    "SCHEMA_VERSION",
    "build_delayed_diagnostic_bundle",
    "delayed_diagnostic_definitions",
    "materialize_delayed_diagnostic_bundle",
    "summarize_decay_profile",
    "summarize_direction_split",
    "summarize_peak_decay",
]
