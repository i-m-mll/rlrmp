"""Steady-state perturbation transforms and structured response reductions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rlrmp.analysis.perturbation_rows import PerturbationSpec
from rlrmp.analysis.pipelines.diagnostic_provenance import repo_relative
from rlrmp.eval.gru_diagnostics import RolloutEvaluation
from rlrmp.eval.steady_state import SteadyStatePerturbationBankConfig
from rlrmp.model.feedback_descriptors import (
    COMPONENT_FORCE_FILTER,
    COMPONENT_POSITION,
    COMPONENT_VELOCITY,
    resolve_controller_feedback_view,
)


SCHEMA_VERSION = "rlrmp.gru_steady_state_perturbation_bank.v1"

PerturbationFamily = Literal["position", "velocity", "force_filter"]
WashinStatus = Literal["steady_state_response", "washin_endpoint_response"]


@dataclass(frozen=True)
class FeedbackPerturbation:
    """One steady-state feedback offset row."""

    perturbation_id: str
    family: PerturbationFamily
    feedback_indices: tuple[int, int]
    direction: tuple[float, float]
    amplitude: float
    units: str
    sign: int

    def to_bank_row(
        self, *, feedback_dim: int, pulse_start: int, pulse_duration: int
    ) -> dict[str, Any]:
        """Return the graph-adapter perturbation row consumed by existing adapters."""

        payload_index = self.feedback_indices[0] if self.direction[0] else self.feedback_indices[1]
        axis = "x" if self.direction[0] else "y"
        return PerturbationSpec(
            perturbation_id=self.perturbation_id,
            channel="sensory_feedback",
            family=f"steady_state_{self.family}_feedback_offset",
            amplitude=float(self.amplitude),
            units=self.units,
            axis=axis,
            basis=f"feedback_{self.family}_xy",
            sign=int(self.sign),
            timing={
                "epoch": "steady_state_endpoint",
                "start_time_index": int(pulse_start),
                "duration_steps": int(pulse_duration),
            },
            adapter="named_graph_channel_offset",
            description=(
                f"Add a {self.units} {self.family} feedback offset after the shared "
                "steady-state wash-in prefix."
            ),
            timing_bin="steady_state_endpoint",
            semantic_family="steady_state_feedback_offset",
            channel_provenance={
                "feedback_dim": int(feedback_dim),
                "feedback_quantity": self.family,
                "feedback_payload_index": int(payload_index),
                "direction": [float(self.direction[0]), float(self.direction[1])],
            },
            feedback_payload_index=payload_index,
            feedback_quantity=self.family,
            force_filter_feedback_only=self.family == "force_filter",
        ).to_json()


def default_feedback_perturbations(
    *,
    feedback_dim: int,
    config: SteadyStatePerturbationBankConfig | None = None,
    position_scale_m: float | None = None,
    velocity_scale_m_s: float | None = None,
    force_filter_scale: float | None = None,
) -> tuple[FeedbackPerturbation, ...]:
    """Return symmetric position, velocity, and force/filter feedback offsets."""

    config = config or SteadyStatePerturbationBankConfig()
    position_scale_m = config.position_scale_m if position_scale_m is None else position_scale_m
    velocity_scale_m_s = (
        config.velocity_scale_m_s if velocity_scale_m_s is None else velocity_scale_m_s
    )
    force_filter_scale = (
        config.force_filter_scale if force_filter_scale is None else force_filter_scale
    )
    rows: list[FeedbackPerturbation] = []
    descriptor_view = resolve_controller_feedback_view(
        None,
        feedback_dim=feedback_dim,
        source="steady_state_feedback_perturbation_bank",
    )
    amplitudes = {
        COMPONENT_POSITION: position_scale_m,
        COMPONENT_VELOCITY: velocity_scale_m_s,
        COMPONENT_FORCE_FILTER: force_filter_scale,
    }
    for component in descriptor_view.iter_components():
        family = component.component_id
        indices = tuple(component.absolute_indices)
        amplitude = amplitudes[family]
        units = component.units or "model_feedback_units"
        for axis, direction in (("x", (1.0, 0.0)), ("y", (0.0, 1.0))):
            for sign, sign_label in ((1, "pos"), (-1, "neg")):
                signed = (sign * direction[0], sign * direction[1])
                rows.append(
                    FeedbackPerturbation(
                        perturbation_id=(
                            f"steady_state_{family}_feedback_offset__{axis}_{sign_label}"
                        ),
                        family=family,
                        feedback_indices=indices,
                        direction=signed,
                        amplitude=amplitude,
                        units=units,
                        sign=sign,
                    )
                )
    return tuple(rows)


def slim_steady_state_manifest(
    manifest: Mapping[str, Any],
    *,
    detail_manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Remove dense profiles and adapter detail from the tracked summary manifest."""

    slim = {key: value for key, value in manifest.items() if key not in {"comparisons", "outputs"}}
    slim["bulk_detail_manifest"] = {
        "path": repo_relative(detail_manifest_path, repo_root=repo_root),
        "format": "json",
        "contains": (
            "full steady-state comparison payloads, dense response profiles, "
            "onset-window profile arrays, adapter detail, and checkpoint provenance"
        ),
    }
    slim["comparisons"] = {
        str(comparison_id): _compact_comparison_payload(comparison)
        for comparison_id, comparison in dict(manifest.get("comparisons", {})).items()
    }
    return slim


def _compact_comparison_payload(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Return a scalar summary for one steady-state comparison."""

    compact = {
        key: comparison[key]
        for key in (
            "schema_version",
            "issue",
            "comparison_id",
            "title",
            "source_experiment",
            "run_id",
            "n_rollout_trials",
            "pulse_duration_steps",
            "feedback_offset_scales",
            "response_window",
            "timing_by_condition",
            "washin_contract",
            "feedback_dim_by_condition",
            "figure",
        )
        if key in comparison
    }
    compact["conditions"] = {
        str(condition_id): _compact_condition_payload(condition)
        for condition_id, condition in dict(comparison.get("conditions", {})).items()
    }
    return compact


def _compact_condition_payload(condition: Mapping[str, Any]) -> dict[str, Any]:
    """Return scalar condition metadata and row metrics without profile arrays."""

    compact = {
        key: condition[key]
        for key in (
            "condition_id",
            "label",
            "metadata",
            "run_id",
            "run_spec_path",
            "artifact_dir",
            "n_replicates",
            "n_rollout_trials_per_replicate",
            "dt_s",
            "washin",
            "response_label",
        )
        if key in condition
    }
    rows = condition.get("rows", [])
    compact["n_rows"] = len(rows) if isinstance(rows, Sequence) else 0
    compact["rows"] = [_compact_row_payload(row) for row in rows if isinstance(row, Mapping)]
    compact["family_summary"] = _compact_family_summary(condition.get("family_summary", {}))
    checkpoint_selection = condition.get("checkpoint_selection")
    if isinstance(checkpoint_selection, Sequence) and not isinstance(checkpoint_selection, str):
        compact["checkpoint_selection_summary"] = _checkpoint_selection_summary(
            checkpoint_selection
        )
    return compact


def _compact_row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return row identity plus scalar metrics."""

    return {
        key: row[key]
        for key in (
            "perturbation_id",
            "family",
            "status",
            "reason",
            "direction",
            "projection_basis",
            "sign",
            "amplitude",
            "units",
            "metrics",
        )
        if key in row
    }


def _compact_family_summary(family_summary: Any) -> dict[str, Any]:
    """Keep family scalar metrics while dropping profile arrays."""

    if not isinstance(family_summary, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for family, payload in family_summary.items():
        if not isinstance(payload, Mapping):
            continue
        compact[str(family)] = {
            key: value for key, value in payload.items() if _is_compact_family_field(key, value)
        }
    return compact


def _is_compact_family_field(key: str, value: Any) -> bool:
    """Return whether a family-summary field belongs in tracked JSON."""

    if key == "relative_time_steps" or "profile" in key:
        return False
    return _is_json_scalar(value) or _is_scalar_mapping(value)


def _checkpoint_selection_summary(selections: Sequence[Any]) -> dict[str, Any]:
    """Summarize checkpoint-selection provenance without listing every replica path."""

    rows = [selection for selection in selections if isinstance(selection, Mapping)]
    summary: dict[str, Any] = {"n_replicates": int(len(rows))}
    sources = sorted(
        {str(row["selection_source"]) for row in rows if row.get("selection_source") is not None}
    )
    if sources:
        summary["selection_sources"] = sources
    for source_key, output_key in (
        ("checkpoint_batches", "checkpoint_batches"),
        ("best_logged_validation_batch", "best_logged_validation_batch"),
        ("scoring_validation_log_batch", "scoring_validation_log_batch"),
    ):
        values = [row[source_key] for row in rows if isinstance(row.get(source_key), int | float)]
        if values:
            summary[output_key] = {
                "min": int(min(values)),
                "max": int(max(values)),
            }
    for source_key, output_key in (
        ("best_logged_validation_objective", "best_logged_validation_objective"),
        ("scoring_validation_objective", "scoring_validation_objective"),
        ("final_validation_objective", "final_validation_objective"),
        (
            "final_vs_selected_validation_degradation",
            "final_vs_selected_validation_degradation",
        ),
    ):
        values = [
            float(row[source_key]) for row in rows if isinstance(row.get(source_key), int | float)
        ]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            summary[output_key] = _summary_stats(arr)
    return summary


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _is_scalar_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(_is_json_scalar(nested) for nested in value.values())


def summarize_feedback_row(
    *,
    perturbation: FeedbackPerturbation,
    base: RolloutEvaluation,
    perturbed: RolloutEvaluation,
    pulse_start: int,
    config: SteadyStatePerturbationBankConfig | None = None,
) -> dict[str, Any]:
    """Summarize a signed feedback perturbation row."""

    config = config or SteadyStatePerturbationBankConfig()
    delta_command = perturbed.command - base.command
    delta_hidden = perturbed.hidden - base.hidden
    direction = np.asarray(perturbation.direction, dtype=np.float64)
    signed_direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    orthogonal_direction = right_handed_orthogonal_direction(signed_direction)
    aligned_command = np.tensordot(delta_command, signed_direction, axes=([-1], [0]))
    orthogonal_command = np.tensordot(
        delta_command,
        orthogonal_direction,
        axes=([-1], [0]),
    )
    aligned_position = np.tensordot(
        perturbed.position - base.position,
        signed_direction,
        axes=([-1], [0]),
    )
    orthogonal_position = np.tensordot(
        perturbed.position - base.position,
        orthogonal_direction,
        axes=([-1], [0]),
    )
    aligned_velocity = np.tensordot(
        perturbed.velocity - base.velocity,
        signed_direction,
        axes=([-1], [0]),
    )
    orthogonal_velocity = np.tensordot(
        perturbed.velocity - base.velocity,
        orthogonal_direction,
        axes=([-1], [0]),
    )
    response = aligned_command[:, :, pulse_start:]
    orthogonal_response = orthogonal_command[:, :, pulse_start:]
    position_response = aligned_position[:, :, pulse_start:]
    orthogonal_position_response = orthogonal_position[:, :, pulse_start:]
    velocity_response = aligned_velocity[:, :, pulse_start:]
    orthogonal_velocity_response = orthogonal_velocity[:, :, pulse_start:]
    command_window, relative_steps = _mean_onset_window(
        aligned_command,
        pulse_start=pulse_start,
        config=config,
    )
    orthogonal_command_window, _ = _mean_onset_window(
        orthogonal_command,
        pulse_start=pulse_start,
        config=config,
    )
    position_window, _ = _mean_onset_window(
        aligned_position,
        pulse_start=pulse_start,
        config=config,
    )
    orthogonal_position_window, _ = _mean_onset_window(
        orthogonal_position,
        pulse_start=pulse_start,
        config=config,
    )
    velocity_window, _ = _mean_onset_window(
        aligned_velocity,
        pulse_start=pulse_start,
        config=config,
    )
    orthogonal_velocity_window, _ = _mean_onset_window(
        orthogonal_velocity,
        pulse_start=pulse_start,
        config=config,
    )
    action_norm = np.linalg.norm(delta_command[:, :, pulse_start:, :], axis=-1)
    hidden_norm = np.linalg.norm(delta_hidden[:, :, pulse_start:, :], axis=-1)
    mean_profile = np.mean(response, axis=(0, 1))
    terminal = response[..., -1] if response.shape[-1] else np.zeros(response.shape[:2])
    settling = settling_step(
        np.abs(mean_profile), tolerance=max(0.05 * peak_abs(mean_profile), 1e-8)
    )
    return {
        "perturbation_id": perturbation.perturbation_id,
        "family": perturbation.family,
        "status": "evaluated",
        "direction": [float(value) for value in perturbation.direction],
        "projection_basis": {
            "aligned_direction": [float(value) for value in signed_direction],
            "orthogonal_direction": [float(value) for value in orthogonal_direction],
            "orthogonal_convention": "right_handed_plus_90_degrees_xy",
        },
        "sign": int(perturbation.sign),
        "amplitude": float(perturbation.amplitude),
        "units": perturbation.units,
        "aligned_output_profile": [float(value) for value in mean_profile],
        "orthogonal_output_profile": [
            float(value) for value in np.mean(orthogonal_response, axis=(0, 1))
        ],
        "aligned_position_profile": [
            float(value) for value in np.mean(position_response, axis=(0, 1))
        ],
        "orthogonal_position_profile": [
            float(value) for value in np.mean(orthogonal_position_response, axis=(0, 1))
        ],
        "aligned_velocity_profile": [
            float(value) for value in np.mean(velocity_response, axis=(0, 1))
        ],
        "orthogonal_velocity_profile": [
            float(value) for value in np.mean(orthogonal_velocity_response, axis=(0, 1))
        ],
        "relative_time_steps": [int(value) for value in relative_steps],
        "aligned_output_window_profile": [float(value) for value in command_window],
        "orthogonal_output_window_profile": [float(value) for value in orthogonal_command_window],
        "aligned_position_window_profile": [float(value) for value in position_window],
        "orthogonal_position_window_profile": [
            float(value) for value in orthogonal_position_window
        ],
        "aligned_velocity_window_profile": [float(value) for value in velocity_window],
        "orthogonal_velocity_window_profile": [
            float(value) for value in orthogonal_velocity_window
        ],
        "metrics": {
            "peak_output_response": float(peak_abs(response)),
            "peak_orthogonal_output_response": float(peak_abs(orthogonal_response)),
            "output_auc_impulse": float(np.sum(np.abs(response)) * float(base.dt) / response.size),
            "orthogonal_output_auc_impulse": float(
                np.sum(np.abs(orthogonal_response)) * float(base.dt) / orthogonal_response.size
            ),
            "terminal_residual": float(np.mean(np.abs(terminal))) if terminal.size else 0.0,
            "recovery_settling_step": settling,
            "direction_variability": float(np.std(np.mean(response, axis=-1)))
            if response.size
            else 0.0,
            "hidden_delta_peak": float(peak_abs(hidden_norm)),
            "output_norm_peak": float(peak_abs(action_norm)),
            "peak_position_m": float(peak_abs(position_window)),
            "peak_orthogonal_position_m": float(peak_abs(orthogonal_position_window)),
            "peak_velocity_m_s": float(peak_abs(velocity_window)),
            "peak_orthogonal_velocity_m_s": float(peak_abs(orthogonal_velocity_window)),
        },
    }


def aggregate_family_profiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Direction-align and average row profiles by feedback family."""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "evaluated":
            continue
        groups.setdefault(str(row["family"]), []).append(row)
    summary: dict[str, Any] = {}
    for family, family_rows in groups.items():
        pair_scores = signed_pair_antisymmetry(family_rows)
        metrics = [row["metrics"] for row in family_rows if isinstance(row.get("metrics"), Mapping)]
        summary[family] = {
            "n_rows": int(len(family_rows)),
            "peak_output_response": float(
                np.mean([metric["peak_output_response"] for metric in metrics])
            ),
            "peak_orthogonal_output_response": float(
                np.mean([metric["peak_orthogonal_output_response"] for metric in metrics])
            ),
            "output_auc_impulse": float(
                np.mean([metric["output_auc_impulse"] for metric in metrics])
            ),
            "orthogonal_output_auc_impulse": float(
                np.mean([metric["orthogonal_output_auc_impulse"] for metric in metrics])
            ),
            "terminal_residual": float(
                np.mean([metric["terminal_residual"] for metric in metrics])
            ),
            "direction_variability": float(
                np.std([metric["peak_output_response"] for metric in metrics])
            ),
            "signed_pair_antisymmetry": pair_scores,
        }
        summary[family].update(_aggregate_profiles(family_rows))
        summary[family].update(_aggregate_window_profiles(family_rows))
    return summary


def _aggregate_profiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return mean/SEM traces for post-onset response profiles."""

    profile_keys = (
        "aligned_output_profile",
        "orthogonal_output_profile",
        "aligned_position_profile",
        "orthogonal_position_profile",
        "aligned_velocity_profile",
        "orthogonal_velocity_profile",
    )
    return _mean_sem_profile_fields(rows, profile_keys)


def _aggregate_window_profiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return mean/SEM traces for all onset-centered figure profiles."""

    output: dict[str, Any] = {}
    if not rows:
        return output
    first = rows[0]
    if "relative_time_steps" in first:
        output["relative_time_steps"] = [int(value) for value in first["relative_time_steps"]]
    else:
        output["relative_time_steps"] = list(range(len(first["aligned_output_profile"])))
    profile_keys = (
        "aligned_output_window_profile",
        "orthogonal_output_window_profile",
        "aligned_position_window_profile",
        "orthogonal_position_window_profile",
        "aligned_velocity_window_profile",
        "orthogonal_velocity_window_profile",
    )
    return output | _mean_sem_profile_fields(rows, profile_keys)


def _mean_sem_profile_fields(
    rows: Sequence[Mapping[str, Any]],
    profile_keys: Sequence[str],
) -> dict[str, Any]:
    """Return mean/SEM fields for profile keys present on row dictionaries."""

    output: dict[str, Any] = {}
    if not rows:
        return output
    first = rows[0]
    for key in profile_keys:
        if key not in first:
            fallback = "aligned_output_profile"
            profiles = np.asarray([row[fallback] for row in rows], dtype=np.float64)
        else:
            profiles = np.asarray([row[key] for row in rows], dtype=np.float64)
        mean_profile = np.mean(profiles, axis=0)
        sem_profile = (
            np.std(profiles, axis=0, ddof=1) / np.sqrt(profiles.shape[0])
            if profiles.shape[0] > 1
            else np.zeros_like(mean_profile)
        )
        output[f"{key}_mean"] = [float(value) for value in mean_profile]
        output[f"{key}_sem"] = [float(value) for value in sem_profile]
    return output


def signed_pair_antisymmetry(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | str]:
    """Return a signed-pair antisymmetry score from +/- axis pairs."""

    by_axis: dict[str, dict[int, np.ndarray]] = {}
    for row in rows:
        direction = tuple(float(value) for value in row.get("direction", (0.0, 0.0)))
        axis = "x" if abs(direction[0]) > abs(direction[1]) else "y"
        sign = 1 if (direction[0] or direction[1]) > 0 else -1
        by_axis.setdefault(axis, {})[sign] = np.asarray(row["aligned_output_profile"], dtype=float)
    scores = []
    for pair in by_axis.values():
        if 1 not in pair or -1 not in pair:
            continue
        denom = max(float(np.linalg.norm(pair[1]) + np.linalg.norm(pair[-1])), 1e-12)
        scores.append(float(np.linalg.norm(pair[1] - pair[-1]) / denom))
    if not scores:
        return {"status": "not_available"}
    return {"status": "available", "mean_aligned_pair_difference_ratio": float(np.mean(scores))}


def washin_diagnostics(
    evaluation: RolloutEvaluation,
    *,
    pulse_start: int,
    config: SteadyStatePerturbationBankConfig | None = None,
    final_window_steps: int | None = None,
) -> dict[str, Any]:
    """Summarize baseline drift over the final wash-in window."""

    config = config or SteadyStatePerturbationBankConfig()
    final_window_steps = (
        config.final_window_steps if final_window_steps is None else final_window_steps
    )
    stop = max(min(pulse_start, evaluation.command.shape[2]), 1)
    start = max(stop - final_window_steps, 0)
    command = evaluation.command[:, :, start:stop, :]
    hidden = evaluation.hidden[:, :, start:stop, :]
    plant = np.concatenate(
        [evaluation.position[:, :, start:stop, :], evaluation.velocity[:, :, start:stop, :]],
        axis=-1,
    )
    command_drift = _window_step_drift(command)
    hidden_drift = _window_step_drift(hidden)
    plant_drift = _window_step_drift(plant)
    baseline_command = np.linalg.norm(command, axis=-1)
    return {
        "window_start_step": int(start),
        "window_stop_step": int(stop),
        "network_output_drift": command_drift,
        "hidden_state_drift": hidden_drift,
        "plant_state_drift": plant_drift,
        "baseline_command_magnitude": _summary_stats(baseline_command),
    }


def _window_step_drift(values: np.ndarray) -> dict[str, float]:
    if values.shape[2] < 2:
        return {"mean": 0.0, "max": 0.0}
    diffs = np.linalg.norm(np.diff(values, axis=2), axis=-1)
    return _summary_stats(diffs)


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
    }


def _response_label(wash: Mapping[str, Any]) -> WashinStatus:
    command = float(wash["network_output_drift"]["max"])
    hidden = float(wash["hidden_state_drift"]["max"])
    plant = float(wash["plant_state_drift"]["max"])
    if command <= 1e-3 and hidden <= 1e-3 and plant <= 1e-5:
        return "steady_state_response"
    return "washin_endpoint_response"


def peak_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def right_handed_orthogonal_direction(direction: np.ndarray) -> np.ndarray:
    """Return the +90 degree right-handed x-y rotation of ``direction``."""

    arr = np.asarray(direction, dtype=np.float64)
    unit = arr / max(float(np.linalg.norm(arr)), 1e-12)
    return np.asarray([-unit[1], unit[0]], dtype=np.float64)


def _mean_onset_window(
    aligned_values: np.ndarray,
    *,
    pulse_start: int,
    config: SteadyStatePerturbationBankConfig | None = None,
    pre_steps: int | None = None,
    post_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return trial/replicate mean in a small pre-onset and recovery window."""

    config = config or SteadyStatePerturbationBankConfig()
    pre_steps = config.pre_onset_figure_steps if pre_steps is None else pre_steps
    post_steps = config.post_onset_figure_steps if post_steps is None else post_steps
    start = max(int(pulse_start) - int(pre_steps), 0)
    stop = min(int(pulse_start) + int(post_steps), aligned_values.shape[2])
    window = aligned_values[:, :, start:stop]
    relative_steps = np.arange(start, stop, dtype=int) - int(pulse_start)
    return np.mean(window, axis=(0, 1)), relative_steps


def settling_step(profile: np.ndarray, *, tolerance: float) -> int | None:
    """Return first step after which the profile remains within tolerance."""

    arr = np.asarray(profile, dtype=np.float64)
    for idx in range(arr.shape[0]):
        if np.all(arr[idx:] <= tolerance):
            return int(idx)
    return None


def _washin_contract(*, config: SteadyStatePerturbationBankConfig | None = None) -> dict[str, Any]:
    config = config or SteadyStatePerturbationBankConfig()
    return {
        "schema_version": SCHEMA_VERSION,
        "initial_mechanics": (
            "mechanics.vector is zeroed, then every 8D current/delayed mechanics "
            "block receives target x/y position with zero velocity, force/filter, "
            "and integrator state."
        ),
        "noise": "epsilon inputs are zeroed before evaluation.",
        "delayed_go_cue": (
            f"go cue off for {int(config.pre_go_steps)} steps, then on; target visible throughout."
        ),
        "fanout_policy": (
            "prefix_equivalent_batched_trials because the current Feedbax eval API "
            "does not expose a supported hidden-state resume hook."
        ),
    }


def _response_window_contract(
    *, config: SteadyStatePerturbationBankConfig | None = None
) -> dict[str, Any]:
    config = config or SteadyStatePerturbationBankConfig()
    return {
        "pre_onset_steps": int(config.pre_onset_figure_steps),
        "post_onset_steps": int(config.post_onset_figure_steps),
        "x_axis": "seconds relative to perturbation onset",
        "projection_basis": {
            "aligned": "signed projection onto the normalized perturbation direction",
            "orthogonal": (
                "signed projection onto the normalized perturbation direction rotated "
                "+90 degrees in the right-handed x-y plane: (-dy, dx)"
            ),
        },
        "rows": [
            "network output aligned with perturbation direction plus lower-emphasis orthogonal companion traces",
            "point-mass position along aligned perturbation direction plus lower-emphasis orthogonal companion traces",
            "point-mass velocity along aligned perturbation direction plus lower-emphasis orthogonal companion traces",
        ],
    }


def _feedback_offset_scales(
    *,
    config: SteadyStatePerturbationBankConfig,
) -> dict[str, float]:
    return {
        "position_m": float(config.position_scale_m),
        "velocity_m_s": float(config.velocity_scale_m_s),
        "force_filter": float(config.force_filter_scale),
    }


__all__ = [
    "FeedbackPerturbation",
    "SteadyStatePerturbationBankConfig",
    "aggregate_family_profiles",
    "default_feedback_perturbations",
    "right_handed_orthogonal_direction",
    "slim_steady_state_manifest",
    "signed_pair_antisymmetry",
    "summarize_feedback_row",
    "washin_diagnostics",
]
