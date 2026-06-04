"""GRU observation-history-to-action map error decomposition sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rlrmp.paths import REPO_ROOT

OBSERVATION_CHANNELS = ("px", "py", "vx", "vy")
ACTION_CHANNELS = ("ux", "uy")
FORMAT_VERSION = "rlrmp.gru_map_error_decomposition.v1"
ISSUE_ID = "ddf7f43"
SOURCE_ISSUE_ID = "aacb9ed"
DEFAULT_LABEL = "fixed_target_random_perturb_validation_selected"
DEFAULT_STANDARD_MANIFEST = (
    REPO_ROOT
    / "results"
    / SOURCE_ISSUE_ID
    / "notes"
    / f"gru_standard_certificates_{DEFAULT_LABEL}_manifest.json"
)


def materialize_gru_map_error_decomposition(
    *,
    standard_manifest_path: Path = DEFAULT_STANDARD_MANIFEST,
    experiment: str = SOURCE_ISSUE_ID,
    run_ids: tuple[str, ...] | None = None,
    use_validation_selected_checkpoints: bool = True,
    top_k: int = 5,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Recompute response maps and return compact decomposition rows."""

    from rlrmp.analysis.cs_gru_standard_materialization import (
        cs_output_feedback_observation_action_map,
        evaluate_gru_clean_actions,
    )

    manifest = _read_json(standard_manifest_path)
    selected_run_ids = run_ids or _source_run_ids_from_standard_manifest(manifest)
    reference_map, reference_metadata = cs_output_feedback_observation_action_map()
    rows = []
    for run_id in selected_run_ids:
        run_spec = _read_json(repo_root / "results" / experiment / "runs" / run_id / "run.json")
        _actions, candidate_map, evaluation_metadata = evaluate_gru_clean_actions(
            run_id,
            run_spec=run_spec,
            experiment=experiment,
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
            repo_root=repo_root,
        )
        covariance = evaluation_metadata.pop("_observation_history_covariance_array", None)
        covariance_metadata = evaluation_metadata.get("observation_history_covariance")
        reference_batch = np.broadcast_to(reference_map[None, :, :, :], candidate_map.shape)
        decomposition = decompose_gru_map_error(
            candidate_map=candidate_map,
            reference_map=reference_batch,
            observation_dim=int(reference_metadata["observation_dim"]),
            input_covariance=covariance,
            input_covariance_metadata=covariance_metadata,
            top_k=top_k,
        )
        rows.append(
            {
                "run_id": f"{run_id}__nominal_clean",
                "source_run_id": run_id,
                "standard_certificate_row": _find_standard_row(manifest, run_id),
                "reference_metadata": reference_metadata,
                "evaluation_metadata": evaluation_metadata,
                "decomposition": decomposition,
            }
        )
    return {
        "format": FORMAT_VERSION,
        "issue": ISSUE_ID,
        "source_issue": experiment,
        "source_standard_manifest": _repo_relative(standard_manifest_path, repo_root=repo_root),
        "checkpoint_policy": (
            "validation_selected_per_replicate"
            if use_validation_selected_checkpoints
            else "final_checkpoint"
        ),
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "rows": rows,
    }


def decompose_gru_map_error(
    *,
    candidate_map: np.ndarray,
    reference_map: np.ndarray,
    observation_dim: int = 4,
    observation_channel_names: tuple[str, ...] = OBSERVATION_CHANNELS,
    action_channel_names: tuple[str, ...] = ACTION_CHANNELS,
    input_covariance: np.ndarray | None = None,
    input_covariance_metadata: dict[str, Any] | None = None,
    top_k: int = 5,
    denominator_floor: float = 1e-12,
) -> dict[str, Any]:
    """Decompose finite-horizon GRU/reference observation-action map error.

    Args:
        candidate_map: GRU local response maps with shape
            ``(*samples, action_time, action, observation_time * observation)``.
        reference_map: extLQG response maps with the same shape.
        observation_dim: Number of observation channels per history time.
        observation_channel_names: Labels for the observation channels.
        action_channel_names: Labels for the action channels.
        input_covariance: Optional covariance in the flattened observation-history basis.
        input_covariance_metadata: JSON metadata for ``input_covariance``.
        top_k: Number of singular directions to report.
        denominator_floor: Floor for ratio denominators.

    Returns:
        JSON-compatible decomposition summary.
    """

    candidate = _as_float_array(candidate_map, name="candidate_map")
    reference = _as_float_array(reference_map, name="reference_map")
    if candidate.shape != reference.shape:
        raise ValueError("candidate_map and reference_map must have the same shape")
    if candidate.ndim < 3:
        raise ValueError("maps must have shape (*samples, action_time, action, history)")
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if len(observation_channel_names) != observation_dim:
        raise ValueError("observation_channel_names length must match observation_dim")

    action_time_count = int(candidate.shape[-3])
    action_dim = int(candidate.shape[-2])
    history_dim = int(candidate.shape[-1])
    if history_dim % observation_dim:
        raise ValueError("history dimension must be divisible by observation_dim")
    if len(action_channel_names) != action_dim:
        action_channel_names = tuple(f"u{idx}" for idx in range(action_dim))

    observation_time_count = history_dim // observation_dim
    delta = candidate - reference
    delta_energy = float(np.sum(delta**2))
    reference_energy = float(np.sum(reference**2))
    candidate_energy = float(np.sum(candidate**2))
    inner = float(np.sum(candidate * reference))
    candidate_norm = float(np.sqrt(candidate_energy))
    reference_norm = float(np.sqrt(reference_energy))
    scalar_gain = inner / max(reference_energy, denominator_floor)
    scalar_residual = candidate - scalar_gain * reference
    scalar_residual_energy = float(np.sum(scalar_residual**2))
    cosine = inner / max(candidate_norm * reference_norm, denominator_floor)

    sample_shape = tuple(int(dim) for dim in candidate.shape[:-3])
    delta_by_time_action_history = np.sum(delta**2, axis=tuple(range(delta.ndim - 3)))
    delta_by_time_action_obs = delta_by_time_action_history.reshape(
        action_time_count,
        action_dim,
        observation_time_count,
        observation_dim,
    )

    matrix = delta.reshape((-1, history_dim))
    singular_directions = _top_singular_directions(
        matrix,
        top_k=min(top_k, min(matrix.shape)),
        observation_dim=observation_dim,
        observation_channel_names=observation_channel_names,
        action_time_count=action_time_count,
        action_dim=action_dim,
        action_channel_names=action_channel_names,
        input_covariance=input_covariance,
        input_covariance_metadata=input_covariance_metadata,
        denominator_floor=denominator_floor,
    )
    annotations = _decision_rule_annotations(
        norm_ratio=candidate_norm / max(reference_norm, denominator_floor),
        cosine=cosine,
        scalar_gain=scalar_gain,
        residual_ratio=scalar_residual_energy / max(reference_energy, denominator_floor),
        singular_directions=singular_directions,
        covariance_available=input_covariance is not None,
    )

    return {
        "format": FORMAT_VERSION,
        "map_shape": [int(dim) for dim in candidate.shape],
        "sample_shape": list(sample_shape),
        "basis": {
            "input": "flattened_observation_history",
            "observation_dim": observation_dim,
            "observation_channels": list(observation_channel_names),
            "observation_time_count": observation_time_count,
            "output": "action_history",
            "action_dim": action_dim,
            "action_channels": list(action_channel_names),
            "action_time_count": action_time_count,
        },
        "summary": {
            "candidate_frobenius": _json_float(candidate_norm),
            "reference_frobenius": _json_float(reference_norm),
            "delta_frobenius": _json_float(np.sqrt(delta_energy)),
            "candidate_reference_norm_ratio": _json_float(
                candidate_norm / max(reference_norm, denominator_floor)
            ),
            "candidate_reference_cosine": _json_float(cosine),
            "best_scalar_gain": _json_float(scalar_gain),
            "best_scalar_residual_frobenius": _json_float(np.sqrt(scalar_residual_energy)),
            "best_scalar_residual_ratio": _json_float(
                scalar_residual_energy / max(reference_energy, denominator_floor)
            ),
            "aggregate_delta_ratio": _json_float(
                delta_energy / max(reference_energy, denominator_floor)
            ),
        },
        "energy_decomposition": {
            "by_action_time": _axis_energy(
                np.sum(delta_by_time_action_obs, axis=(1, 2, 3)),
                total=delta_energy,
                label_prefix="t",
            ),
            "by_observation_time": _axis_energy(
                np.sum(delta_by_time_action_obs, axis=(0, 1, 3)),
                total=delta_energy,
                label_prefix="obs_t",
            ),
            "by_observation_lag": _lag_energy(delta_by_time_action_obs, total=delta_energy),
            "by_observation_channel": _axis_energy(
                np.sum(delta_by_time_action_obs, axis=(0, 1, 2)),
                total=delta_energy,
                labels=observation_channel_names,
            ),
            "by_action_channel": _axis_energy(
                np.sum(delta_by_time_action_obs, axis=(0, 2, 3)),
                total=delta_energy,
                labels=action_channel_names,
            ),
        },
        "top_singular_directions": singular_directions,
        "decision_rule_annotations": annotations,
    }


def render_map_error_decomposition_markdown(result: dict[str, Any]) -> str:
    """Render compact markdown for one or more map-error decomposition rows."""

    rows = result["rows"]
    lines = [
        "# GRU Map-Error Decomposition",
        "",
        f"Issue: `{result['issue']}`. Source issue: `{result['source_issue']}`.",
        "",
        "This sidecar decomposes the GRU minus extLQG 4D observation-history-to-action "
        "response map. It is diagnostic-only; the standard certificate gate remains the "
        "standard response-map/action evidence.",
        "",
        "## Rows",
        "",
        (
            "| Row | norm ratio | cosine | scalar gain | scalar residual | top error | "
            "covariance | annotations |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        summary = row["decomposition"]["summary"]
        top = row["decomposition"]["top_singular_directions"]
        top_value = top[0]["singular_value"] if top else None
        projection = top[0].get("covariance_projection", {}) if top else {}
        annotations = ", ".join(row["decomposition"]["decision_rule_annotations"]) or "none"
        lines.append(
            "| "
            f"{row['run_id']} | "
            f"{_fmt(summary['candidate_reference_norm_ratio'])} | "
            f"{_fmt(summary['candidate_reference_cosine'])} | "
            f"{_fmt(summary['best_scalar_gain'])} | "
            f"{_fmt(summary['best_scalar_residual_ratio'])} | "
            f"{_fmt(top_value)} | "
            f"{projection.get('status', 'not_available')} | "
            f"{annotations} |"
        )

    lines.extend(["", "## Top Singular Directions", ""])
    for row in rows:
        lines.append(f"### `{row['run_id']}`")
        lines.append("")
        lines.append(
            "| rank | singular value | energy fraction | obs time | obs channel | action time | "
            "action channel | covariance projection |"
        )
        lines.append("|---:|---:|---:|---|---|---|---|---:|")
        for direction in row["decomposition"]["top_singular_directions"]:
            projection = direction.get("covariance_projection", {})
            lines.append(
                "| "
                f"{direction['rank']} | "
                f"{_fmt(direction['singular_value'])} | "
                f"{_fmt(direction['delta_energy_fraction'])} | "
                f"{direction['dominant_observation_time']} | "
                f"{direction['dominant_observation_channel']} | "
                f"{direction['dominant_action_time']} | "
                f"{direction['dominant_action_channel']} | "
                f"{_fmt(projection.get('variance'))} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_map_error_decomposition_result(
    result: dict[str, Any],
    *,
    markdown_path: Path,
    json_path: Path,
) -> None:
    """Write compact markdown and JSON sidecars."""

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_map_error_decomposition_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_run_ids_from_standard_manifest(manifest: dict[str, Any]) -> tuple[str, ...]:
    run_ids = []
    for row in manifest.get("rows", ()):
        source_run_id = row.get("spec", {}).get("parameters", {}).get("source_run_id")
        if source_run_id:
            run_ids.append(source_run_id)
    if not run_ids:
        raise ValueError("standard manifest does not contain spec.parameters.source_run_id rows")
    return tuple(run_ids)


def _find_standard_row(manifest: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    for row in manifest.get("rows", ()):
        if row.get("spec", {}).get("parameters", {}).get("source_run_id") == run_id:
            spec = row.get("spec", {})
            return {
                "run_id": spec.get("run_id"),
                "status": row.get("status"),
                "observation_history_to_action_map_mismatch": _component_summary(
                    row,
                    "observation_history_to_action_map_mismatch",
                ),
            }
    return None


def _component_summary(row: dict[str, Any], component_name: str) -> dict[str, Any] | None:
    for component in row.get("certificate_components", ()):
        if component.get("name") == component_name:
            return {
                "status": component.get("status"),
                "summary": component.get("summary"),
                "reason": component.get("reason"),
            }
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _top_singular_directions(
    matrix: np.ndarray,
    *,
    top_k: int,
    observation_dim: int,
    observation_channel_names: tuple[str, ...],
    action_time_count: int,
    action_dim: int,
    action_channel_names: tuple[str, ...],
    input_covariance: np.ndarray | None,
    input_covariance_metadata: dict[str, Any] | None,
    denominator_floor: float,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    total_singular_energy = float(np.sum(singular_values**2))
    directions = []
    for index in range(top_k):
        right = vh[index]
        left = u[:, index].reshape((-1, action_time_count, action_dim))
        right_energy = right.reshape((-1, observation_dim)) ** 2
        left_energy = np.sum(left**2, axis=0)
        obs_time_energy = np.sum(right_energy, axis=1)
        obs_channel_energy = np.sum(right_energy, axis=0)
        action_time_energy = np.sum(left_energy, axis=1)
        action_channel_energy = np.sum(left_energy, axis=0)
        projection = _covariance_projection(
            right,
            input_covariance=input_covariance,
            input_covariance_metadata=input_covariance_metadata,
            denominator_floor=denominator_floor,
        )
        directions.append(
            {
                "rank": index + 1,
                "singular_value": _json_float(singular_values[index]),
                "delta_energy_fraction": _json_float(
                    singular_values[index] ** 2 / max(total_singular_energy, denominator_floor)
                ),
                "dominant_observation_time": int(np.argmax(obs_time_energy)),
                "dominant_observation_channel": observation_channel_names[
                    int(np.argmax(obs_channel_energy))
                ],
                "dominant_action_time": int(np.argmax(action_time_energy)),
                "dominant_action_channel": action_channel_names[
                    int(np.argmax(action_channel_energy))
                ],
                "covariance_projection": projection,
            }
        )
    return directions


def _covariance_projection(
    direction: np.ndarray,
    *,
    input_covariance: np.ndarray | None,
    input_covariance_metadata: dict[str, Any] | None,
    denominator_floor: float,
) -> dict[str, Any]:
    if input_covariance is None:
        return {
            "status": "not_available",
            "missing_input": "input_covariance",
            "reason": (
                "training/eval perturbation covariance in observation-history basis was not "
                "supplied"
            ),
            "covariance_metadata": input_covariance_metadata or {"status": "missing"},
        }
    covariance = _as_float_array(input_covariance, name="input_covariance")
    if covariance.shape != (direction.shape[0], direction.shape[0]):
        return {
            "status": "not_available",
            "missing_input": "input_covariance_matching_history_basis",
            "reason": (
                "input_covariance shape does not match flattened observation-history dimension"
            ),
            "covariance_shape": [int(dim) for dim in covariance.shape],
            "history_dim": int(direction.shape[0]),
            "covariance_metadata": input_covariance_metadata or {"status": "shape_mismatch"},
        }
    variance = float(direction @ covariance @ direction)
    trace = float(np.trace(covariance))
    return {
        "status": "available",
        "variance": _json_float(variance),
        "trace_fraction": _json_float(variance / max(trace, denominator_floor)),
        "covariance_metadata": input_covariance_metadata or {"status": "available"},
    }


def _decision_rule_annotations(
    *,
    norm_ratio: float,
    cosine: float,
    scalar_gain: float,
    residual_ratio: float,
    singular_directions: list[dict[str, Any]],
    covariance_available: bool,
) -> list[str]:
    annotations = []
    if norm_ratio < 0.5:
        annotations.append("low_norm")
    if cosine >= 0.8 and scalar_gain < 0.7:
        annotations.append("weak_gain")
    if 0.5 <= norm_ratio <= 1.5 and cosine < 0.5:
        annotations.append("wrong_timing_or_channel")
    if singular_directions:
        top_projection = singular_directions[0].get("covariance_projection", {})
        trace_fraction = top_projection.get("trace_fraction")
        if covariance_available and trace_fraction is not None:
            if trace_fraction < 0.01:
                annotations.append("unexcited_directions")
            elif residual_ratio > 0.25:
                annotations.append("well_excited_residual")
        elif not covariance_available:
            annotations.append("excitation_unknown")
    return annotations or ["no_single_rule_dominates"]


def _lag_energy(values: np.ndarray, *, total: float) -> list[dict[str, Any]]:
    # values shape: action_time, action, observation_time, observation_channel
    lag_totals: dict[int, float] = {}
    for action_time in range(values.shape[0]):
        for observation_time in range(values.shape[2]):
            lag = action_time - observation_time
            lag_totals[lag] = lag_totals.get(lag, 0.0) + float(
                np.sum(values[action_time, :, observation_time, :])
            )
    return [
        {
            "lag": int(lag),
            "energy": _json_float(energy),
            "fraction": _json_float(energy / max(total, 1e-12)),
        }
        for lag, energy in sorted(lag_totals.items())
    ]


def _axis_energy(
    values: np.ndarray,
    *,
    total: float,
    labels: tuple[str, ...] | None = None,
    label_prefix: str = "",
) -> list[dict[str, Any]]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    entries = []
    for index, energy in enumerate(flat):
        label = labels[index] if labels is not None else f"{label_prefix}{index}"
        entries.append(
            {
                "index": index,
                "label": label,
                "energy": _json_float(energy),
                "fraction": _json_float(float(energy) / max(total, 1e-12)),
            }
        )
    return entries


def _as_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _json_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _fmt(value: Any) -> str:
    if value is None:
        return "not_available"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "not_available"
    return f"{numeric:.6g}"


__all__ = [
    "ACTION_CHANNELS",
    "DEFAULT_LABEL",
    "DEFAULT_STANDARD_MANIFEST",
    "FORMAT_VERSION",
    "ISSUE_ID",
    "OBSERVATION_CHANNELS",
    "SOURCE_ISSUE_ID",
    "decompose_gru_map_error",
    "materialize_gru_map_error_decomposition",
    "render_map_error_decomposition_markdown",
    "write_map_error_decomposition_result",
]
