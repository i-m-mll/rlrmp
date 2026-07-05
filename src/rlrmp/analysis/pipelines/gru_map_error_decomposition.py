"""GRU observation-history-to-action map error decomposition sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rlrmp.analysis.manifest_queries import (
    certificate_component_summary,
    standard_row_by_source_run_id,
)
from rlrmp.analysis.pipelines.gru_checkpoint_selection import load_materialized_fixed_bank_manifest
from rlrmp.io import read_json, update_marked_section
from rlrmp.paths import REPO_ROOT
from rlrmp.runtime.run_specs import resolve_run_record

OBSERVATION_CHANNELS = ("px", "py", "vx", "vy")
ACTION_CHANNELS = ("ux", "uy")
ALIGNED_OBSERVATION_CHANNELS = ("p_parallel", "p_lateral", "v_parallel", "v_lateral")
ALIGNED_ACTION_CHANNELS = ("u_parallel", "u_lateral")
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
    preferred_checkpoint_manifest_path: Path | None = None,
    alignment_basis: str = "raw_cartesian",
    reference_feedback_basis: str = "auto",
    top_k: int = 5,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Recompute response maps and return compact decomposition rows."""

    from rlrmp.analysis.pipelines.cs_gru_standard_materialization import (
        cs_output_feedback_observation_action_map,
        evaluate_gru_clean_actions,
    )

    manifest = read_json(standard_manifest_path)
    selected_run_ids = run_ids or _source_run_ids_from_standard_manifest(manifest)
    reference_map, reference_metadata = cs_output_feedback_observation_action_map()
    rows = []
    for run_id in selected_run_ids:
        run_spec = resolve_run_record(experiment, run_id, repo_root=repo_root)
        _actions, candidate_map, evaluation_metadata = evaluate_gru_clean_actions(
            run_id,
            run_spec=run_spec,
            experiment=experiment,
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
            preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
            repo_root=repo_root,
        )
        covariance = evaluation_metadata.pop("_observation_history_covariance_array", None)
        covariance_metadata = evaluation_metadata.get("observation_history_covariance")
        candidate_feedback_basis = _candidate_feedback_basis(run_spec)
        (
            candidate_map,
            covariance,
            covariance_metadata,
            candidate_feedback_basis,
            projection_metadata,
        ) = _project_candidate_map_to_decomposition_basis(
            candidate_map=candidate_map,
            covariance=covariance,
            covariance_metadata=covariance_metadata,
            candidate_feedback_basis=candidate_feedback_basis,
            run_spec=run_spec,
            reference_observation_dim=int(reference_metadata["observation_dim"]),
        )
        if projection_metadata is not None:
            evaluation_metadata["candidate_map_projection"] = projection_metadata
            evaluation_metadata["observation_history_covariance"] = covariance_metadata
        (
            candidate_map,
            reference_map_for_run,
            covariance,
            covariance_metadata,
            history_alignment_metadata,
        ) = _align_maps_to_common_observation_history(
            candidate_map=candidate_map,
            reference_map=reference_map,
            covariance=covariance,
            covariance_metadata=covariance_metadata,
            observation_dim=int(reference_metadata["observation_dim"]),
        )
        if history_alignment_metadata is not None:
            evaluation_metadata["map_history_alignment"] = history_alignment_metadata
            evaluation_metadata["observation_history_covariance"] = covariance_metadata
        reference_batch = np.broadcast_to(
            reference_map_for_run[None, :, :, :],
            candidate_map.shape,
        )
        reference_input_transform = _reference_observation_from_candidate_feedback_transform(
            candidate_feedback_basis=candidate_feedback_basis,
            reference_feedback_basis=reference_feedback_basis,
        )
        alignment_directions = _alignment_directions_from_run_spec(
            run_spec,
            alignment_basis=alignment_basis,
        )
        decomposition = decompose_gru_map_error(
            candidate_map=candidate_map,
            reference_map=reference_batch,
            observation_dim=int(reference_metadata["observation_dim"]),
            reference_observation_from_candidate_transform=reference_input_transform,
            candidate_feedback_basis=candidate_feedback_basis,
            reference_feedback_basis=(
                "raw_delayed_position_velocity"
                if reference_feedback_basis == "auto"
                else reference_feedback_basis
            ),
            alignment_directions=alignment_directions,
            alignment_frame_source=_alignment_frame_source(alignment_basis, run_spec),
            target_time_convention=_target_time_convention(alignment_basis, run_spec),
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
            _effective_checkpoint_policy_from_manifest(
                experiment,
                preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
                repo_root=repo_root,
            )
            if use_validation_selected_checkpoints
            else "final_checkpoint"
        ),
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "rows": rows,
    }


def _effective_checkpoint_policy_from_manifest(
    experiment: str,
    *,
    preferred_checkpoint_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Return the checkpoint policy represented by an optional preferred manifest."""

    manifest = load_materialized_fixed_bank_manifest(
        experiment=experiment,
        manifest_path=preferred_checkpoint_manifest_path,
        repo_root=repo_root,
    )
    if manifest is not None:
        return str(manifest.get("checkpoint_policy") or "fixed_bank_rescored_per_replicate")
    return "validation_selected_per_replicate"


def decompose_gru_map_error(
    *,
    candidate_map: np.ndarray,
    reference_map: np.ndarray,
    observation_dim: int = 4,
    observation_channel_names: tuple[str, ...] = OBSERVATION_CHANNELS,
    action_channel_names: tuple[str, ...] = ACTION_CHANNELS,
    reference_observation_from_candidate_transform: np.ndarray | None = None,
    candidate_feedback_basis: str = "raw_delayed_position_velocity",
    reference_feedback_basis: str = "raw_delayed_position_velocity",
    alignment_directions: np.ndarray | None = None,
    alignment_frame_source: str = "raw_cartesian_observation_action_basis",
    target_time_convention: str = "not_applicable_static_or_raw",
    moving_target_status: str = "not_applicable",
    moving_target_deferred_note: str | None = None,
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
        reference_observation_from_candidate_transform: Optional ``(observation,
            observation)`` matrix ``C`` such that ``reference_observation = C @
            candidate_feedback``. This converts the analytical reference map into the
            controller-visible input basis before any task-frame alignment. For the
            target-relative GRU contract, ``C = -I`` because the feedback is
            ``[target - position, -velocity]``.
        candidate_feedback_basis: Metadata label for the GRU/controller feedback basis.
        reference_feedback_basis: Metadata label for the analytical reference map basis.
        alignment_directions: Optional reach direction vectors with shape ``(2,)`` or
            ``(*samples, 2)``. When supplied, maps are transformed into the Feedbax-compatible
            parallel/lateral basis before decomposition.
        alignment_frame_source: Metadata describing where ``alignment_directions`` came from.
        target_time_convention: Metadata for static-target timing, or a deferred moving-target
            convention note.
        moving_target_status: ``not_applicable``/``deferred`` metadata for tracking tasks.
        moving_target_deferred_note: Optional note explaining the missing moving-target contract.
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

    candidate_observation_basis_metadata = _candidate_observation_basis_metadata(
        candidate_feedback_basis=candidate_feedback_basis,
        reference_feedback_basis=reference_feedback_basis,
        reference_observation_from_candidate_transform=(
            reference_observation_from_candidate_transform
        ),
    )
    if reference_observation_from_candidate_transform is not None:
        reference = _apply_reference_observation_basis_transform(
            reference,
            observation_dim=observation_dim,
            transform=reference_observation_from_candidate_transform,
        )

    alignment_metadata = _raw_alignment_metadata(
        frame_source=alignment_frame_source,
        target_time_convention=target_time_convention,
        moving_target_status=moving_target_status,
        moving_target_deferred_note=moving_target_deferred_note,
    )
    if alignment_directions is not None:
        (
            candidate,
            reference,
            observation_channel_names,
            action_channel_names,
            input_covariance,
            input_covariance_metadata,
            alignment_metadata,
        ) = _align_maps_to_reach_basis(
            candidate=candidate,
            reference=reference,
            observation_dim=observation_dim,
            action_channel_names=action_channel_names,
            directions=alignment_directions,
            frame_source=alignment_frame_source,
            target_time_convention=target_time_convention,
            moving_target_status=moving_target_status,
            moving_target_deferred_note=moving_target_deferred_note,
            input_covariance=input_covariance,
            input_covariance_metadata=input_covariance_metadata,
        )

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
            "candidate_observation_basis": candidate_observation_basis_metadata,
            "observation_time_count": observation_time_count,
            "output": "action_history",
            "action_dim": action_dim,
            "action_channels": list(action_channel_names),
            "action_time_count": action_time_count,
        },
        "alignment": alignment_metadata,
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
    update_marked_section(
        markdown_path,
        "gru_map_error_decomposition",
        render_map_error_decomposition_markdown(result),
    )
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
    row = standard_row_by_source_run_id(manifest, run_id)
    if row is None:
        return None
    spec = row.get("spec", {})
    return {
        "run_id": spec.get("run_id"),
        "status": row.get("status"),
        "observation_history_to_action_map_mismatch": certificate_component_summary(
            row,
            "observation_history_to_action_map_mismatch",
        ),
    }


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _raw_alignment_metadata(
    *,
    frame_source: str,
    target_time_convention: str,
    moving_target_status: str,
    moving_target_deferred_note: str | None,
) -> dict[str, Any]:
    return {
        "alignment_basis": "raw_cartesian",
        "frame_source": frame_source,
        "target_time_convention": target_time_convention,
        "sign_convention": "raw ux/uy and px/py/vx/vy Cartesian channels",
        "direction_vectors": None,
        "target_velocity_used": False,
        "moving_target": _moving_target_metadata(
            status=moving_target_status,
            deferred_note=moving_target_deferred_note,
        ),
    }


def _candidate_observation_basis_metadata(
    *,
    candidate_feedback_basis: str,
    reference_feedback_basis: str,
    reference_observation_from_candidate_transform: np.ndarray | None,
) -> dict[str, Any]:
    metadata = {
        "candidate_feedback_basis": candidate_feedback_basis,
        "reference_feedback_basis": reference_feedback_basis,
        "reference_converted_to_candidate_basis": (
            reference_observation_from_candidate_transform is not None
        ),
    }
    if reference_observation_from_candidate_transform is not None:
        transform = _as_float_array(
            reference_observation_from_candidate_transform,
            name="reference_observation_from_candidate_transform",
        )
        metadata["reference_observation_from_candidate_transform"] = [
            [_json_float(value) for value in row] for row in transform
        ]
        if np.allclose(transform, -np.eye(transform.shape[0])):
            metadata["sign_convention"] = (
                "candidate feedback is [target_x - delayed_x, target_y - delayed_y, "
                "-delayed_vx, -delayed_vy]; analytical reference input derivatives are "
                "therefore multiplied by -1 before comparison"
            )
    return metadata


def _apply_reference_observation_basis_transform(
    values: np.ndarray,
    *,
    observation_dim: int,
    transform: np.ndarray,
) -> np.ndarray:
    """Convert a reference map from its observation basis to the candidate basis."""

    matrix = _as_float_array(transform, name="reference_observation_from_candidate_transform")
    if matrix.shape != (observation_dim, observation_dim):
        raise ValueError(
            "reference_observation_from_candidate_transform must have shape "
            f"({observation_dim}, {observation_dim})"
        )
    if values.shape[-1] % observation_dim:
        raise ValueError("history dimension must be divisible by observation_dim")
    observation_time_count = values.shape[-1] // observation_dim
    maps = values.reshape(values.shape[:-1] + (observation_time_count, observation_dim))
    converted = np.einsum("...ti,ij->...tj", maps, matrix)
    return converted.reshape(values.shape)


def _align_maps_to_reach_basis(
    *,
    candidate: np.ndarray,
    reference: np.ndarray,
    observation_dim: int,
    action_channel_names: tuple[str, ...],
    directions: np.ndarray,
    frame_source: str,
    target_time_convention: str,
    moving_target_status: str,
    moving_target_deferred_note: str | None,
    input_covariance: np.ndarray | None,
    input_covariance_metadata: dict[str, Any] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray | None,
    dict[str, Any] | None,
    dict[str, Any],
]:
    if observation_dim != 4:
        raise ValueError("reach-aligned decomposition currently requires observation_dim=4")
    if candidate.shape[-2] != 2:
        raise ValueError("reach-aligned decomposition currently requires 2D action channels")

    sample_shape = tuple(int(dim) for dim in candidate.shape[:-3])
    direction_vectors = _broadcast_direction_vectors(directions, sample_shape=sample_shape)
    transform = _projection_matrices(direction_vectors)
    candidate_aligned = _apply_reach_alignment_to_map(candidate, transform)
    reference_aligned = _apply_reach_alignment_to_map(reference, transform)
    covariance_aligned, covariance_metadata = _align_input_covariance(
        input_covariance,
        input_covariance_metadata=input_covariance_metadata,
        transforms=transform,
        observation_time_count=candidate.shape[-1] // observation_dim,
    )
    # This mirrors Feedbax AlignedVars/project_onto_direction today. Keep it isolated so
    # future GraphSpec retained-observable selectors can replace only the frame source.
    metadata = {
        "alignment_basis": "reach_aligned_parallel_lateral",
        "frame_source": frame_source,
        "target_time_convention": target_time_convention,
        "sign_convention": (
            "parallel = dot(unit_reach_direction, vector); "
            "lateral = cross(unit_reach_direction, vector)"
        ),
        "direction_vectors": _direction_metadata(direction_vectors),
        "target_velocity_used": False,
        "moving_target": _moving_target_metadata(
            status=moving_target_status,
            deferred_note=moving_target_deferred_note,
        ),
        "feedbax_reference": "feedbax.analysis.aligned.project_onto_direction",
    }
    return (
        candidate_aligned,
        reference_aligned,
        ALIGNED_OBSERVATION_CHANNELS,
        ALIGNED_ACTION_CHANNELS if len(action_channel_names) == 2 else action_channel_names,
        covariance_aligned,
        covariance_metadata,
        metadata,
    )


def _candidate_feedback_basis(run_spec: dict[str, Any]) -> str:
    model_basis = (
        run_spec.get("model_summary", {})
        .get("feedback", {})
        .get("basis")
    )
    input_basis = (
        run_spec.get("task_timing", {})
        .get("target_relative_multitarget", {})
        .get("input_contract", {})
        .get("controller_feedback_basis")
    )
    return str(model_basis or input_basis or "raw_delayed_position_velocity")


def _candidate_feedback_dim(run_spec: dict[str, Any]) -> int:
    feedback = run_spec.get("model_summary", {}).get("feedback", {})
    if "dimension" in feedback:
        return int(feedback["dimension"])
    contract_shape = (
        run_spec.get("task_timing", {})
        .get("target_relative_multitarget", {})
        .get("input_contract", {})
        .get("shape", [4])
    )
    return int(contract_shape[0])


def _project_candidate_map_to_decomposition_basis(
    *,
    candidate_map: np.ndarray,
    covariance: np.ndarray | None,
    covariance_metadata: dict[str, Any] | None,
    candidate_feedback_basis: str,
    run_spec: dict[str, Any],
    reference_observation_dim: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any] | None, str, dict[str, Any] | None]:
    if candidate_feedback_basis != "target_relative_delayed_feedback_plus_force_filter":
        return candidate_map, covariance, covariance_metadata, candidate_feedback_basis, None
    feedback_dim = _candidate_feedback_dim(run_spec)
    if feedback_dim <= reference_observation_dim:
        return candidate_map, covariance, covariance_metadata, "target_relative_delayed_feedback", None
    if candidate_map.shape[-1] % feedback_dim:
        raise ValueError(
            "proprioceptive candidate map history dimension is not divisible by "
            f"feedback_dim={feedback_dim}"
        )
    observation_time_count = candidate_map.shape[-1] // feedback_dim
    maps = candidate_map.reshape(
        candidate_map.shape[:-1] + (observation_time_count, feedback_dim)
    )
    projected = maps[..., :reference_observation_dim].reshape(
        candidate_map.shape[:-1] + (observation_time_count * reference_observation_dim,)
    )
    projected_covariance = covariance
    projected_covariance_metadata = covariance_metadata
    if covariance is not None:
        keep_indices = np.concatenate(
            [
                np.arange(
                    time * feedback_dim,
                    time * feedback_dim + reference_observation_dim,
                )
                for time in range(observation_time_count)
            ]
        )
        projected_covariance = covariance[np.ix_(keep_indices, keep_indices)]
        projected_covariance_metadata = {
            **(covariance_metadata or {}),
            "projection": "first_four_pos_vel_channels_from_6d_proprioceptive_feedback",
            "original_covariance_shape": [int(dim) for dim in covariance.shape],
            "projected_covariance_shape": [
                int(dim) for dim in projected_covariance.shape
            ],
        }
    projection_metadata = {
        "status": "projected",
        "reason": (
            "The standard map-error decomposition compares to the extLQG 4D "
            "position/velocity observation-history map."
        ),
        "original_candidate_feedback_basis": candidate_feedback_basis,
        "projected_candidate_feedback_basis": "target_relative_delayed_feedback",
        "original_feedback_dim": int(feedback_dim),
        "projected_feedback_dim": int(reference_observation_dim),
        "excluded_channels": ["delayed_force_filter_x", "delayed_force_filter_y"],
    }
    return (
        projected,
        projected_covariance,
        projected_covariance_metadata,
        "target_relative_delayed_feedback",
        projection_metadata,
    )


def _reference_observation_from_candidate_feedback_transform(
    *,
    candidate_feedback_basis: str,
    reference_feedback_basis: str,
) -> np.ndarray | None:
    if reference_feedback_basis not in {"auto", "raw_delayed_position_velocity"}:
        raise ValueError(
            "reference_feedback_basis must be 'auto' or 'raw_delayed_position_velocity'"
        )
    if candidate_feedback_basis == "target_relative_delayed_feedback":
        return -np.eye(4, dtype=np.float64)
    if candidate_feedback_basis in {
        "raw_delayed_position_velocity",
        "4D delayed position/velocity feedback",
    }:
        return None
    raise ValueError(f"unsupported candidate feedback basis: {candidate_feedback_basis}")


def _align_maps_to_common_observation_history(
    *,
    candidate_map: np.ndarray,
    reference_map: np.ndarray,
    covariance: np.ndarray | None,
    covariance_metadata: dict[str, Any] | None,
    observation_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any] | None, dict[str, Any] | None]:
    candidate_obs = candidate_map.shape[-1] // observation_dim
    reference_obs = reference_map.shape[-1] // observation_dim
    if candidate_obs == reference_obs:
        return candidate_map, reference_map, covariance, covariance_metadata, None
    keep_obs = min(candidate_obs, reference_obs)
    candidate_start = candidate_obs - keep_obs
    reference_start = reference_obs - keep_obs
    aligned_candidate = _crop_map_observation_history(
        candidate_map,
        observation_dim=observation_dim,
        start=candidate_start,
        keep=keep_obs,
    )
    aligned_reference = _crop_map_observation_history(
        reference_map,
        observation_dim=observation_dim,
        start=reference_start,
        keep=keep_obs,
    )
    aligned_covariance = covariance
    aligned_covariance_metadata = covariance_metadata
    if covariance is not None:
        keep_indices = np.concatenate(
            [
                np.arange(
                    (candidate_start + time) * observation_dim,
                    (candidate_start + time + 1) * observation_dim,
                )
                for time in range(keep_obs)
            ]
        )
        aligned_covariance = covariance[np.ix_(keep_indices, keep_indices)]
        aligned_covariance_metadata = {
            **(covariance_metadata or {}),
            "history_alignment": "most_recent_common_observation_history",
            "original_covariance_shape": [int(dim) for dim in covariance.shape],
            "aligned_covariance_shape": [int(dim) for dim in aligned_covariance.shape],
        }
    metadata = {
        "status": "aligned",
        "method": "most_recent_common_observation_history",
        "candidate_observation_time_count": int(candidate_obs),
        "reference_observation_time_count": int(reference_obs),
        "retained_observation_time_count": int(keep_obs),
        "candidate_dropped_oldest_observation_steps": int(candidate_start),
        "reference_dropped_oldest_observation_steps": int(reference_start),
    }
    return (
        aligned_candidate,
        aligned_reference,
        aligned_covariance,
        aligned_covariance_metadata,
        metadata,
    )


def _crop_map_observation_history(
    values: np.ndarray,
    *,
    observation_dim: int,
    start: int,
    keep: int,
) -> np.ndarray:
    observation_time_count = values.shape[-1] // observation_dim
    maps = values.reshape(values.shape[:-1] + (observation_time_count, observation_dim))
    cropped = maps[..., start : start + keep, :]
    return cropped.reshape(values.shape[:-1] + (keep * observation_dim,))


def _alignment_directions_from_run_spec(
    run_spec: dict[str, Any],
    *,
    alignment_basis: str,
) -> np.ndarray | None:
    if alignment_basis == "raw_cartesian":
        return None
    if alignment_basis not in {"static_reach_aligned", "auto_static_reach_aligned"}:
        raise ValueError(
            "alignment_basis must be 'raw_cartesian', 'static_reach_aligned', "
            "or 'auto_static_reach_aligned'"
        )
    return np.asarray(_static_reach_vector_from_run_spec(run_spec), dtype=np.float64)


def _static_reach_vector_from_run_spec(run_spec: dict[str, Any]) -> list[float]:
    task_timing = run_spec.get("task_timing", {})
    target_cfg = task_timing.get("target_relative_multitarget", {})
    distribution = target_cfg.get("target_distribution", {})
    if target_cfg.get("enabled") and distribution.get("original_target_anchor_m"):
        target = distribution["original_target_anchor_m"]
        start = distribution.get("start_position_m", [0.0, 0.0])
        return [float(target[0]) - float(start[0]), float(target[1]) - float(start[1])]

    hps_task = run_spec.get("hps", {}).get("task", {})
    if hps_task.get("fixed_target_pos") is not None:
        target = hps_task["fixed_target_pos"]
        start = hps_task.get("fixed_init_pos", [0.0, 0.0])
        return [float(target[0]) - float(start[0]), float(target[1]) - float(start[1])]

    raise ValueError("run spec does not declare a static reach direction")


def _alignment_frame_source(alignment_basis: str, run_spec: dict[str, Any]) -> str:
    if alignment_basis == "raw_cartesian":
        return "raw_cartesian_observation_action_basis"
    if (
        run_spec.get("task_timing", {})
        .get("target_relative_multitarget", {})
        .get("enabled")
    ):
        return "declared_target_relative_original_target_anchor"
    return "declared_fixed_target_endpoint_minus_start"


def _target_time_convention(alignment_basis: str, run_spec: dict[str, Any]) -> str:
    if alignment_basis == "raw_cartesian":
        return "not_applicable_static_or_raw"
    if (
        run_spec.get("task_timing", {})
        .get("target_relative_multitarget", {})
        .get("enabled")
    ):
        return (
            "static_target_known_immediately; current local response-map bank repeats the "
            "first validation target for stochastic histories"
        )
    return "static_fixed_target_endpoint_minus_start"


def _broadcast_direction_vectors(directions: np.ndarray, *, sample_shape: tuple[int, ...]) -> np.ndarray:
    direction_vectors = _as_float_array(directions, name="alignment_directions")
    if direction_vectors.shape[-1:] != (2,):
        raise ValueError("alignment_directions must end with an xy dimension of length 2")
    if direction_vectors.shape == (2,):
        direction_vectors = np.broadcast_to(direction_vectors, sample_shape + (2,))
    else:
        direction_vectors = np.broadcast_to(direction_vectors, sample_shape + (2,))
    norms = np.linalg.norm(direction_vectors, axis=-1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("alignment_directions must contain nonzero reach vectors")
    return direction_vectors / norms


def _projection_matrices(unit_directions: np.ndarray) -> np.ndarray:
    matrices = np.empty(unit_directions.shape[:-1] + (2, 2), dtype=np.float64)
    matrices[..., 0, 0] = unit_directions[..., 0]
    matrices[..., 0, 1] = unit_directions[..., 1]
    matrices[..., 1, 0] = -unit_directions[..., 1]
    matrices[..., 1, 1] = unit_directions[..., 0]
    return matrices


def _apply_reach_alignment_to_map(values: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    sample_shape = values.shape[:-3]
    observation_time_count = values.shape[-1] // 4
    action_time_count = values.shape[-3]
    sample_count = int(np.prod(sample_shape, dtype=np.int64)) if sample_shape else 1
    maps = values.reshape((sample_count, action_time_count, 2, observation_time_count, 2, 2))
    transform = transforms.reshape((sample_count, 2, 2))
    input_inverse = np.swapaxes(transform, -1, -2)
    aligned = np.einsum("nab,ntbopc,ncd->ntaopd", transform, maps, input_inverse)
    return aligned.reshape(values.shape)


def _align_input_covariance(
    covariance: np.ndarray | None,
    *,
    input_covariance_metadata: dict[str, Any] | None,
    transforms: np.ndarray,
    observation_time_count: int,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if covariance is None:
        return None, input_covariance_metadata
    if not _all_transforms_equal(transforms):
        metadata = dict(input_covariance_metadata or {})
        metadata.update(
            {
                "status": "not_applicable",
                "reason": (
                    "condition-wise reach alignment has no single pooled flattened "
                    "observation-history covariance basis"
                ),
                "original_covariance_status": input_covariance_metadata or {"status": "available"},
            }
        )
        return None, metadata
    transform = transforms.reshape((-1, 2, 2))[0]
    per_observation = _block_diag(transform, transform)
    history_transform = _block_diag(*(per_observation for _ in range(observation_time_count)))
    cov = _as_float_array(covariance, name="input_covariance")
    if cov.shape != history_transform.shape:
        return cov, input_covariance_metadata
    return history_transform @ cov @ history_transform.T, input_covariance_metadata


def _all_transforms_equal(transforms: np.ndarray) -> bool:
    flat = transforms.reshape((-1, 2, 2))
    return bool(np.allclose(flat, flat[0]))


def _block_diag(*blocks: np.ndarray) -> np.ndarray:
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    result = np.zeros((rows, cols), dtype=np.float64)
    row_start = 0
    col_start = 0
    for block in blocks:
        row_end = row_start + block.shape[0]
        col_end = col_start + block.shape[1]
        result[row_start:row_end, col_start:col_end] = block
        row_start = row_end
        col_start = col_end
    return result


def _direction_metadata(direction_vectors: np.ndarray) -> dict[str, Any]:
    flat = direction_vectors.reshape((-1, 2))
    if _all_transforms_equal(_projection_matrices(direction_vectors)):
        return {
            "mode": "static",
            "unit_direction": [_json_float(value) for value in flat[0]],
        }
    return {
        "mode": "condition_wise",
        "sample_shape": [int(dim) for dim in direction_vectors.shape[:-1]],
        "unit_directions": [
            [_json_float(value) for value in direction]
            for direction in flat
        ],
    }


def _moving_target_metadata(
    *,
    status: str,
    deferred_note: str | None,
) -> dict[str, Any]:
    metadata = {"status": status}
    if status == "deferred" and deferred_note is None:
        deferred_note = (
            "Moving-target/tracking alignment is deferred until a reference trajectory, "
            "target-velocity, and information-delay contract exists."
        )
    if deferred_note is not None:
        metadata["deferred_note"] = deferred_note
    return metadata


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
    "ALIGNED_ACTION_CHANNELS",
    "ALIGNED_OBSERVATION_CHANNELS",
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
