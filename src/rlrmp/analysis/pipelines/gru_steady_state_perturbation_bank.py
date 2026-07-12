"""Steady-state feedback perturbation bank for C&S GRU controllers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.plot import save_figure
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.analysis.perturbation_rows import PerturbationSpec
from rlrmp.analysis.gru_standard_certificate import normalize_gru_hps
from rlrmp.analysis.pipelines.diagnostic_provenance import repo_relative, write_regeneration_spec
from rlrmp.eval.checkpoint_selection import (
    CheckpointSelectionMode,
    load_validation_selected_checkpoint_model,
)
from rlrmp.eval.gru_diagnostics import RolloutEvaluation
from rlrmp.eval.perturbation_bank import (
    apply_perturbation_to_trial_specs,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    RunFigureInputs,
    repeat_single_validation_trial,
    resolve_run_inputs,
)
from rlrmp.eval.sisu_spectrum import (
    set_sisu_condition,
    zero_disturbance_payload,
)
from rlrmp.io import update_marked_section
from rlrmp.model.feedback_descriptors import (
    COMPONENT_FORCE_FILTER,
    COMPONENT_POSITION,
    COMPONENT_VELOCITY,
    resolve_controller_feedback_view,
)
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.runtime.run_spec_access import require_run_seed
from rlrmp.train.task_model import setup_task_model_pair


SCHEMA_VERSION = "rlrmp.gru_steady_state_perturbation_bank.v1"
ISSUE = "87424a4"
_ANALYSIS_PRESET = load_analysis_parameter_preset("gru_steady_state_perturbation_bank").parameters
DEFAULT_PRE_GO_STEPS = int(_ANALYSIS_PRESET["pre_go_steps"])
DEFAULT_POST_GO_WASHIN_STEPS = int(_ANALYSIS_PRESET["post_go_washin_steps"])
DEFAULT_PULSE_DURATION_STEPS = int(_ANALYSIS_PRESET["pulse_duration_steps"])
DEFAULT_FINAL_WINDOW_STEPS = int(_ANALYSIS_PRESET["final_window_steps"])
DEFAULT_N_ROLLOUT_TRIALS = int(_ANALYSIS_PRESET["n_rollout_trials"])
DEFAULT_POSITION_SCALE_M = float(_ANALYSIS_PRESET["position_scale_m"])
DEFAULT_VELOCITY_SCALE_M_S = float(_ANALYSIS_PRESET["velocity_scale_m_s"])
DEFAULT_FORCE_FILTER_SCALE = float(_ANALYSIS_PRESET["force_filter_scale"])
DEFAULT_PRE_ONSET_FIGURE_STEPS = int(_ANALYSIS_PRESET["pre_onset_figure_steps"])
DEFAULT_POST_ONSET_FIGURE_STEPS = int(_ANALYSIS_PRESET["post_onset_figure_steps"])
SUMMARY_MARKER = "steady_state_perturbation_bank"
SUMMARY_FILENAME = "steady_state_perturbation_bank_summary.json"
DETAIL_FILENAME = "steady_state_perturbation_bank_detail.json"

PerturbationFamily = Literal["position", "velocity", "force_filter"]
ComparisonKind = Literal["sisu", "pgd"]
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


@dataclass(frozen=True)
class ConditionSpec:
    """One evaluated row/condition inside a comparison."""

    condition_id: str
    label: str
    transform: Callable[[Any], Any] | None = None
    metadata: Mapping[str, Any] | None = None
    source_experiment: str | None = None
    run_id: str | None = None
    delayed: bool | None = None


@dataclass(frozen=True)
class ComparisonSpec:
    """User-facing comparison to materialize into one figure."""

    comparison_id: str
    title: str
    kind: ComparisonKind
    source_experiment: str
    run_id: str
    conditions: tuple[ConditionSpec, ...]
    delayed: bool
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history"
    preferred_checkpoint_manifest_path: Path | None = None


@dataclass(frozen=True)
class SteadyStatePerturbationBankConfig:
    """Entry-level defaults for steady-state feedback-bank materialization."""

    result_experiment: str = ISSUE
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS
    pre_go_steps: int = DEFAULT_PRE_GO_STEPS
    post_go_washin_steps: int = DEFAULT_POST_GO_WASHIN_STEPS
    pulse_duration_steps: int = DEFAULT_PULSE_DURATION_STEPS
    final_window_steps: int = DEFAULT_FINAL_WINDOW_STEPS
    position_scale_m: float = DEFAULT_POSITION_SCALE_M
    velocity_scale_m_s: float = DEFAULT_VELOCITY_SCALE_M_S
    force_filter_scale: float = DEFAULT_FORCE_FILTER_SCALE
    pre_onset_figure_steps: int = DEFAULT_PRE_ONSET_FIGURE_STEPS
    post_onset_figure_steps: int = DEFAULT_POST_ONSET_FIGURE_STEPS


def sisu_condition(value: float, *, label: str | None = None) -> ConditionSpec:
    """Return a condition that sets the SISU input to ``value``."""

    return ConditionSpec(
        condition_id=f"sisu_{value:g}".replace(".", "p"),
        label=label or f"SISU={value:g}",
        transform=lambda trials, sisu=float(value): set_sisu_condition(trials, sisu),
        metadata={"sisu": float(value)},
    )


def identity_condition(
    condition_id: str,
    label: str,
    *,
    source_experiment: str | None = None,
    run_id: str | None = None,
    delayed: bool | None = None,
) -> ConditionSpec:
    """Return an unmodified condition for a single trained run."""

    return ConditionSpec(
        condition_id=condition_id,
        label=label,
        transform=None,
        metadata={},
        source_experiment=source_experiment,
        run_id=run_id,
        delayed=delayed,
    )


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


def materialize_steady_state_comparisons(
    *,
    comparisons: Sequence[ComparisonSpec],
    result_experiment: str = ISSUE,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
    pulse_duration_steps: int = DEFAULT_PULSE_DURATION_STEPS,
    position_scale_m: float = DEFAULT_POSITION_SCALE_M,
    velocity_scale_m_s: float = DEFAULT_VELOCITY_SCALE_M_S,
    force_filter_scale: float = DEFAULT_FORCE_FILTER_SCALE,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize steady-state feedback responses and comparison figures."""

    config = SteadyStatePerturbationBankConfig(
        result_experiment=result_experiment,
        n_rollout_trials=int(n_rollout_trials),
        pulse_duration_steps=int(pulse_duration_steps),
        position_scale_m=float(position_scale_m),
        velocity_scale_m_s=float(velocity_scale_m_s),
        force_filter_scale=float(force_filter_scale),
    )
    repo_root = repo_root.resolve()
    notes_dir = mkdir_p(repo_root / "results" / config.result_experiment / "notes")
    feedback_offset_scales = _feedback_offset_scales(config=config)
    all_results: dict[str, Any] = {}
    for comparison in comparisons:
        all_results[comparison.comparison_id] = evaluate_comparison(
            comparison,
            config=config,
            repo_root=repo_root,
        )

    detail_manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": config.result_experiment,
        "n_rollout_trials": int(config.n_rollout_trials),
        "pulse_duration_steps": int(config.pulse_duration_steps),
        "feedback_offset_scales": feedback_offset_scales,
        "response_window": _response_window_contract(config=config),
        "washin_contract": _washin_contract(config=config),
        "comparisons": all_results,
    }
    summary_path = notes_dir / SUMMARY_FILENAME
    detail_path = repo_root / "_artifacts" / config.result_experiment / "notes" / DETAIL_FILENAME
    markdown_path = notes_dir / "steady_state_perturbation_bank.md"
    regeneration_path = notes_dir / "steady_state_perturbation_bank_regeneration_spec.json"
    summary_manifest = slim_steady_state_manifest(
        detail_manifest,
        detail_manifest_path=detail_path,
        repo_root=repo_root,
    )
    outputs = {
        "summary_json": repo_relative(summary_path, repo_root=repo_root),
        "summary_markdown": repo_relative(markdown_path, repo_root=repo_root),
        "detail_json": repo_relative(detail_path, repo_root=repo_root),
        "regeneration_spec": repo_relative(regeneration_path, repo_root=repo_root),
    }
    detail_manifest["outputs"] = outputs
    summary_manifest["outputs"] = outputs
    mkdir_p(detail_path.parent)
    detail_path.write_text(
        json.dumps(detail_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_summary_markdown(markdown_path, render_summary_markdown(summary_manifest))
    write_regeneration_spec(
        spec_path=regeneration_path,
        diagnostic_name="gru_steady_state_perturbation_bank",
        materializer=(
            "rlrmp.analysis.pipelines.gru_steady_state_perturbation_bank."
            "materialize_steady_state_comparisons"
        ),
        command="PYTHONPATH=$PWD/src uv run --no-sync python results/87424a4/scripts/materialize_steady_state_perturbation_bank.py",
        parameters={
            "result_experiment": config.result_experiment,
            "n_rollout_trials": config.n_rollout_trials,
            "pulse_duration_steps": config.pulse_duration_steps,
            "position_scale_m": config.position_scale_m,
            "velocity_scale_m_s": config.velocity_scale_m_s,
            "force_filter_scale": config.force_filter_scale,
        },
        inputs=[
            {"role": "run_spec", "path": row["run_spec_path"]}
            for payload in all_results.values()
            for row in payload["conditions"].values()
        ],
        outputs=[
            {"role": "summary_json", "path": summary_path},
            {"role": "summary_markdown", "path": markdown_path},
            {"role": "steady_state_perturbation_bank_detail_json", "path": detail_path},
        ],
        source_files=[
            "src/rlrmp/analysis/pipelines/gru_steady_state_perturbation_bank.py",
            "results/87424a4/scripts/materialize_steady_state_perturbation_bank.py",
        ],
        notes=[
            "Local model evaluation only; no retraining, pod, or reach-context bank rerun.",
            "Tracked summary JSON is scalar-only and points to ignored _artifacts detail bytes.",
        ],
        repo_root=repo_root,
    )
    return summary_manifest


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


def evaluate_comparison(
    comparison: ComparisonSpec,
    *,
    config: SteadyStatePerturbationBankConfig,
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate one comparison and write its three-panel figure."""

    condition_payloads: dict[str, Any] = {}
    timing_payloads: dict[str, Any] = {}
    feedback_dims: dict[str, int] = {}
    feedback_offset_scales = _feedback_offset_scales(config=config)
    for condition in comparison.conditions:
        source_experiment = condition.source_experiment or comparison.source_experiment
        run_id = condition.run_id or comparison.run_id
        delayed = comparison.delayed if condition.delayed is None else condition.delayed
        run = resolve_run_inputs(
            experiment=source_experiment,
            run_ids=[run_id],
            labels=[condition.label],
            repo_root=repo_root,
        )[0]
        hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
        seed = require_run_seed(run.run_spec, source=run.run_spec_path)
        pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
        n_replicates = int(hps.model.n_replicates)
        model, checkpoint_selection = load_validation_selected_checkpoint_model(
            experiment=source_experiment,
            run_id=run.run_id,
            run_spec=run.run_spec,
            preferred_manifest_path=comparison.preferred_checkpoint_manifest_path,
            checkpoint_selection_mode=comparison.checkpoint_selection_mode,
            repo_root=repo_root,
        )
        base_trials = repeat_single_validation_trial(
            pair.task.validation_trials,
            config.n_rollout_trials,
        )
        steady_trials, timing = make_steady_state_trial_specs(
            base_trials,
            delayed=delayed,
            target_position=np.asarray(_target_position(run, base_trials), dtype=np.float64),
            config=config,
            min_post_onset_steps=config.post_onset_figure_steps,
        )
        steady_trials = pad_feedback_offset_inputs(
            steady_trials,
            expected_feedback_dim=_expected_feedback_dim_from_hps(hps),
        )
        steady_trials = zero_disturbance_payload(steady_trials)
        feedback_dim = _feedback_dim(steady_trials)
        perturbations = default_feedback_perturbations(
            feedback_dim=feedback_dim,
            config=config,
        )
        trials = (
            steady_trials if condition.transform is None else condition.transform(steady_trials)
        )
        condition_payloads[condition.condition_id] = evaluate_condition(
            condition=condition,
            run=run,
            model=model,
            task=pair.task,
            trial_specs=trials,
            perturbations=perturbations,
            feedback_dim=feedback_dim,
            timing=timing,
            n_replicates=n_replicates,
            checkpoint_selection=checkpoint_selection,
            config=config,
            repo_root=repo_root,
        )
        timing_payloads[condition.condition_id] = timing
        feedback_dims[condition.condition_id] = int(feedback_dim)

    figure = build_response_figure(
        comparison_title=comparison.title,
        conditions=condition_payloads,
        dt=float(next(iter(condition_payloads.values()))["dt_s"]),
        pulse_duration_steps=config.pulse_duration_steps,
    )
    spec = {
        "schema_version": SCHEMA_VERSION,
        "issue": config.result_experiment,
        "comparison_id": comparison.comparison_id,
        "title": comparison.title,
        "source_experiment": comparison.source_experiment,
        "run_id": comparison.run_id,
        "n_rollout_trials": int(config.n_rollout_trials),
        "pulse_duration_steps": int(config.pulse_duration_steps),
        "feedback_offset_scales": feedback_offset_scales,
        "response_window": _response_window_contract(config=config),
        "timing_by_condition": timing_payloads,
        "washin_contract": _washin_contract(config=config),
    }
    _ensure_rlrmp_registered()
    saved = save_figure(
        fig=figure,
        spec=spec,
        package="rlrmp",
        experiment=config.result_experiment,
        topic=comparison.comparison_id,
        extra_packages=["rlrmp"],
    )
    return {
        **spec,
        "feedback_dim_by_condition": feedback_dims,
        "conditions": condition_payloads,
        "figure": {
            key: None if value is None else repo_relative(value, repo_root=repo_root)
            for key, value in saved.items()
        },
    }


def make_steady_state_trial_specs(
    trial_specs: Any,
    *,
    delayed: bool,
    target_position: np.ndarray,
    config: SteadyStatePerturbationBankConfig | None = None,
    pre_go_steps: int | None = None,
    post_go_washin_steps: int | None = None,
    pulse_duration_steps: int | None = None,
    min_post_onset_steps: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return trial specs initialized at the target with a steady-state wash-in prefix."""

    config = config or SteadyStatePerturbationBankConfig()
    pre_go_steps = config.pre_go_steps if pre_go_steps is None else pre_go_steps
    post_go_washin_steps = (
        config.post_go_washin_steps if post_go_washin_steps is None else post_go_washin_steps
    )
    pulse_duration_steps = (
        config.pulse_duration_steps if pulse_duration_steps is None else pulse_duration_steps
    )
    updated = set_mechanics_vector_to_target(trial_specs, target_position)
    updated = set_target_streams_to_constant(updated, target_position)
    updated = zero_disturbance_payload(updated)
    original_horizon = _trial_horizon(updated)
    nominal_pre_go_steps = pre_go_steps if delayed else 0
    nominal_pulse_start = int(nominal_pre_go_steps + post_go_washin_steps)
    horizon_extension = {"extended": False, "original_horizon_steps": int(original_horizon)}
    if min_post_onset_steps is not None:
        min_horizon = int(nominal_pulse_start) + int(min_post_onset_steps)
        updated, horizon_extension = extend_trial_specs_to_horizon(
            updated,
            min_horizon_steps=min_horizon,
        )
    horizon = _trial_horizon(updated)
    if delayed:
        effective_pre_go = min(pre_go_steps, max(horizon - 1, 0))
        updated = set_delayed_go_cue(updated, pre_go_steps=effective_pre_go)
        horizon_clamped_pulse_start = min(
            effective_pre_go + post_go_washin_steps,
            max(horizon - 1, 0),
        )
        washin_policy = "10_pre_go_then_post_go_hold_prefix"
    else:
        effective_pre_go = 0
        horizon_clamped_pulse_start = min(post_go_washin_steps, max(horizon - 1, 0))
        washin_policy = "immediate_hold_prefix"
    pulse_start = _figure_compatible_pulse_start(
        horizon_clamped_pulse_start,
        horizon_steps=horizon,
        config=config,
    )
    response_steps = max(horizon - pulse_start, 0)
    return updated, {
        "horizon_steps": int(horizon),
        "original_horizon_steps": int(original_horizon),
        "pre_go_steps_requested": int(nominal_pre_go_steps),
        "pre_go_steps": int(effective_pre_go),
        "post_go_washin_steps_requested": int(post_go_washin_steps),
        "post_go_washin_steps_actual": int(max(pulse_start - effective_pre_go, 0)),
        "horizon_extension": horizon_extension,
        "pulse_start_step_nominal": int(nominal_pulse_start),
        "pulse_start_step_requested": int(nominal_pulse_start),
        "pulse_start_step_requested_meaning": (
            "legacy compatibility key; nominal onset before horizon/window clamping"
        ),
        "pulse_start_step_horizon_clamped": int(horizon_clamped_pulse_start),
        "pulse_start_step": int(pulse_start),
        "pulse_duration_steps": int(min(max(pulse_duration_steps, 1), max(response_steps, 1))),
        "response_steps": int(response_steps),
        "post_onset_steps_requested": (
            None if min_post_onset_steps is None else int(min_post_onset_steps)
        ),
        "post_onset_steps_available": int(response_steps),
        "washin_policy": washin_policy,
        "fanout_policy": (
            "prefix_equivalent_batched_trials; Feedbax task API does not expose a stable "
            "hidden-state resume hook, so every perturbation row shares the same deterministic "
            "wash-in prefix and zero-noise inputs inside the same materialization pass."
        ),
    }


def extend_trial_specs_to_horizon(
    trial_specs: Any,
    *,
    min_horizon_steps: int,
) -> tuple[Any, dict[str, Any]]:
    """Extend time-indexed trial-spec leaves by holding their final time value."""

    current_horizon = _trial_horizon(trial_specs)
    if current_horizon >= min_horizon_steps:
        return trial_specs, {
            "extended": False,
            "original_horizon_steps": int(current_horizon),
            "horizon_steps": int(current_horizon),
            "min_horizon_steps": int(min_horizon_steps),
        }

    def extend_leaf(leaf: Any) -> Any:
        shape = getattr(leaf, "shape", None)
        if shape is None or len(shape) < 2:
            return leaf
        if int(shape[1]) == current_horizon:
            return _pad_time_axis_with_final_value(leaf, min_horizon_steps)
        if int(shape[1]) == current_horizon - 1:
            return _pad_time_axis_with_final_value(leaf, min_horizon_steps - 1)
        return leaf

    updated = jt.map(extend_leaf, trial_specs)
    updated = _extend_timeline_to_horizon(
        updated,
        original_horizon=current_horizon,
        min_horizon_steps=min_horizon_steps,
    )
    return updated, {
        "extended": True,
        "original_horizon_steps": int(current_horizon),
        "horizon_steps": int(min_horizon_steps),
        "min_horizon_steps": int(min_horizon_steps),
        "policy": "pad_time_axis_with_final_value_for_hold_at_target_endpoint_eval",
    }


def _pad_time_axis_with_final_value(leaf: Any, target_steps: int) -> Any:
    """Pad axis 1 to ``target_steps`` by repeating the final slice."""

    current_steps = int(leaf.shape[1])
    if current_steps >= target_steps:
        return leaf
    pad_count = int(target_steps - current_steps)
    final = leaf[:, -1:, ...]
    padding = jnp.repeat(final, pad_count, axis=1)
    return jnp.concatenate([leaf, padding.astype(leaf.dtype)], axis=1)


def _extend_timeline_to_horizon(
    trial_specs: Any,
    *,
    original_horizon: int,
    min_horizon_steps: int,
) -> Any:
    """Update optional timeline metadata when it uses the old horizon endpoint."""

    timeline = getattr(trial_specs, "timeline", None)
    if timeline is None:
        return trial_specs
    updated = trial_specs
    if getattr(timeline, "n_steps", None) is not None:
        updated = eqx.tree_at(lambda ts: ts.timeline.n_steps, updated, int(min_horizon_steps))
        timeline = updated.timeline
    epoch_bounds = getattr(timeline, "epoch_bounds", None)
    if epoch_bounds is not None:
        bounds = jnp.asarray(epoch_bounds)
        if bool(np.all(np.asarray(bounds[..., -1]) == int(original_horizon))):
            new_bounds = bounds.at[..., -1].set(int(min_horizon_steps))
            updated = eqx.tree_at(lambda ts: ts.timeline.epoch_bounds, updated, new_bounds)
    return updated


def _figure_compatible_pulse_start(
    requested_pulse_start: int,
    *,
    horizon_steps: int,
    config: SteadyStatePerturbationBankConfig,
) -> int:
    """Keep the requested wash-in unless it would starve the recovery figure window."""

    if horizon_steps >= config.pre_onset_figure_steps + config.post_onset_figure_steps:
        latest = max(horizon_steps - config.post_onset_figure_steps, 0)
        return int(min(requested_pulse_start, latest))
    return int(requested_pulse_start)


def set_mechanics_vector_to_target(trial_specs: Any, target_position: np.ndarray) -> Any:
    """Set every 8D mechanics block to target position with zero velocity/force/filter."""

    if "mechanics.vector" not in trial_specs.inits:
        return trial_specs
    current = jnp.asarray(trial_specs.inits["mechanics.vector"])
    target = jnp.asarray(target_position, dtype=current.dtype)
    vector = jnp.zeros_like(current)
    block_count = vector.shape[-1] // 8
    for block in range(block_count):
        start = block * 8
        vector = vector.at[..., start : start + 2].set(target)
    return eqx.tree_at(lambda t: t.inits["mechanics.vector"], trial_specs, vector)


def set_target_streams_to_constant(trial_specs: Any, target_position: np.ndarray) -> Any:
    """Set target-like input streams to a constant hold-at-target value."""

    updated = trial_specs
    target = jnp.asarray(target_position)
    for key in ("target",):
        if key in updated.inputs:
            current = jnp.asarray(updated.inputs[key])
            payload = jnp.broadcast_to(target.astype(current.dtype), current.shape)
            updated = eqx.tree_at(lambda t, k=key: t.inputs[k], updated, payload)
    if "effector_target" in updated.inputs:
        effector_target = updated.inputs["effector_target"]
        pos = jnp.asarray(effector_target.pos)
        vel = (
            None
            if effector_target.vel is None
            else jnp.zeros_like(jnp.asarray(effector_target.vel))
        )
        new_effector_target = eqx.tree_at(
            lambda s: (s.pos, s.vel),
            effector_target,
            (
                jnp.broadcast_to(target.astype(pos.dtype), pos.shape),
                vel,
            ),
        )
        updated = eqx.tree_at(lambda t: t.inputs["effector_target"], updated, new_effector_target)
    if "task" in updated.inputs:
        task = updated.inputs["task"]
        effector_target = task.effector_target
        pos = jnp.asarray(effector_target.pos)
        vel = jnp.zeros_like(jnp.asarray(effector_target.vel))
        new_effector_target = eqx.tree_at(
            lambda s: (s.pos, s.vel),
            effector_target,
            (jnp.broadcast_to(target.astype(pos.dtype), pos.shape), vel),
        )
        task = eqx.tree_at(lambda s: s.effector_target, task, new_effector_target)
        target_on = jnp.ones_like(jnp.asarray(task.target_on))
        task = eqx.tree_at(lambda s: s.target_on, task, target_on)
        updated = eqx.tree_at(lambda t: t.inputs["task"], updated, task)
    return updated


def set_delayed_go_cue(trial_specs: Any, *, pre_go_steps: int) -> Any:
    """Set delayed task go cue off for ``pre_go_steps`` and on afterward."""

    updated = trial_specs
    if "input" in updated.inputs:
        controller_input = jnp.asarray(updated.inputs["input"])
        if controller_input.ndim == 3 and controller_input.shape[-1] >= 1:
            go = jnp.ones_like(controller_input[..., 0])
            go = go.at[:, :pre_go_steps].set(0)
            controller_input = controller_input.at[..., 0].set(go)
            updated = eqx.tree_at(lambda t: t.inputs["input"], updated, controller_input)
    if "task" in updated.inputs:
        task = updated.inputs["task"]
        hold = jnp.zeros_like(jnp.asarray(task.hold))
        hold = hold.at[:, :pre_go_steps, :].set(1)
        task = eqx.tree_at(lambda s: s.hold, task, hold)
        updated = eqx.tree_at(lambda t: t.inputs["task"], updated, task)
    return updated


def pad_feedback_offset_inputs(trial_specs: Any, *, expected_feedback_dim: int | None) -> Any:
    """Pad preexisting feedback-offset input streams to the controller feedback width."""

    if expected_feedback_dim is None:
        return trial_specs
    updated = trial_specs
    for key in (
        "perturbation_training.sensory_feedback",
        "perturbation_training.delayed_observation",
    ):
        if key not in updated.inputs:
            continue
        current = jnp.asarray(updated.inputs[key])
        if current.shape[-1] >= expected_feedback_dim:
            continue
        padding = jnp.zeros((*current.shape[:-1], expected_feedback_dim - current.shape[-1]))
        padded = jnp.concatenate([current, padding.astype(current.dtype)], axis=-1)
        updated = eqx.tree_at(lambda t, k=key: t.inputs[k], updated, padded)
    return updated


def evaluate_condition(
    *,
    condition: ConditionSpec,
    run: RunFigureInputs,
    model: Any,
    task: Any,
    trial_specs: Any,
    perturbations: Sequence[FeedbackPerturbation],
    feedback_dim: int,
    timing: Mapping[str, Any],
    n_replicates: int,
    checkpoint_selection: Sequence[Any],
    config: SteadyStatePerturbationBankConfig,
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate one condition on the steady-state feedback bank."""

    base = _evaluate_model_on_trial_specs(
        model=model,
        task=task,
        trial_specs=trial_specs,
        n_replicates=n_replicates,
        seed=0,
    )
    wash = washin_diagnostics(
        base,
        pulse_start=int(timing["pulse_start_step"]),
        config=config,
    )
    rows = []
    for perturbation in perturbations:
        row = perturbation.to_bank_row(
            feedback_dim=feedback_dim,
            pulse_start=int(timing["pulse_start_step"]),
            pulse_duration=int(timing["pulse_duration_steps"]),
        )
        adapter = apply_perturbation_to_trial_specs(trial_specs, row, model=model)
        if adapter.status != "evaluated":
            rows.append(
                {
                    "perturbation_id": perturbation.perturbation_id,
                    "family": perturbation.family,
                    "status": adapter.status,
                    "reason": adapter.reason,
                    "adapter": adapter.to_json(),
                }
            )
            continue
        perturbed = _evaluate_model_on_trial_specs(
            model=adapter.model if adapter.model is not None else model,
            task=task,
            trial_specs=adapter.trial_specs,
            n_replicates=n_replicates,
            seed=0,
        )
        rows.append(
            summarize_feedback_row(
                perturbation=perturbation,
                base=base,
                perturbed=perturbed,
                pulse_start=int(timing["pulse_start_step"]),
                config=config,
            )
            | {"adapter": adapter.to_json()}
        )
    groups = aggregate_family_profiles(rows)
    return {
        "condition_id": condition.condition_id,
        "label": condition.label,
        "metadata": dict(condition.metadata or {}),
        "run_id": run.run_id,
        "run_spec_path": repo_relative(run.run_spec_path, repo_root=repo_root),
        "artifact_dir": repo_relative(run.artifact_dir, repo_root=repo_root),
        "n_replicates": int(base.command.shape[0]),
        "n_rollout_trials_per_replicate": int(base.command.shape[1]),
        "dt_s": float(base.dt),
        "washin": wash,
        "response_label": _response_label(wash),
        "checkpoint_selection": [
            selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
        ],
        "rows": rows,
        "family_summary": groups,
    }


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


def build_response_figure(
    *,
    comparison_title: str,
    conditions: Mapping[str, Mapping[str, Any]],
    dt: float,
    pulse_duration_steps: int,
) -> go.Figure:
    """Build one condition-comparison figure with output and plant responses."""

    families = ("position", "velocity", "force_filter")
    row_specs = (
        ("aligned_output_window_profile", "orthogonal_output_window_profile", "Output", None),
        (
            "aligned_position_window_profile",
            "orthogonal_position_window_profile",
            "Position (m)",
            "m",
        ),
        (
            "aligned_velocity_window_profile",
            "orthogonal_velocity_window_profile",
            "Velocity (m/s)",
            "m/s",
        ),
    )
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Position feedback",
            "Velocity feedback",
            "Force/filter feedback",
            "",
            "",
            "",
            "",
            "",
            "",
        ),
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.055,
    )
    for col, family in enumerate(families, start=1):
        for row_index, (profile_key, orthogonal_profile_key, _, _) in enumerate(row_specs, start=1):
            for idx, condition in enumerate(conditions.values()):
                family_summary = condition.get("family_summary", {}).get(family)
                if not family_summary:
                    continue
                y = np.asarray(family_summary[f"{profile_key}_mean"], dtype=float)
                sem = np.asarray(family_summary[f"{profile_key}_sem"], dtype=float)
                x = np.asarray(family_summary["relative_time_steps"], dtype=float) * dt
                color = _condition_color(idx)
                label = str(condition["label"])
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line={"color": color, "width": 2.1},
                        name=f"{label} aligned" if row_index == 1 else label,
                        legendgroup=label,
                        showlegend=(col, row_index) == (1, 1),
                    ),
                    row=row_index,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([y - sem, (y + sem)[::-1]]),
                        mode="lines",
                        line={"width": 0, "color": color},
                        fill="toself",
                        fillcolor=_rgba(color, 0.14),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=label,
                    ),
                    row=row_index,
                    col=col,
                )
                if f"{orthogonal_profile_key}_mean" in family_summary:
                    orthogonal_y = np.asarray(
                        family_summary[f"{orthogonal_profile_key}_mean"],
                        dtype=float,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=orthogonal_y,
                            mode="lines",
                            line={
                                "color": _rgba(color, 0.60),
                                "width": 1.6,
                                "dash": "solid",
                            },
                            name=f"{label} orthogonal",
                            legendgroup=f"{label} orthogonal",
                            showlegend=(col, row_index) == (1, 1),
                        ),
                        row=row_index,
                        col=col,
                    )
    if pulse_duration_steps > 1:
        fig.add_vrect(
            x0=0,
            x1=float(pulse_duration_steps) * dt,
            fillcolor="lightgray",
            opacity=0.35,
            line_width=0,
            row="all",
            col="all",
            layer="below",
        )
    else:
        fig.add_vline(
            x=0,
            line={"color": "rgba(90,90,90,0.65)", "width": 1.4},
            row="all",
            col="all",
        )
    fig.update_layout(
        title=comparison_title,
        width=1180,
        height=760,
        margin={"l": 78, "r": 34, "t": 92, "b": 86},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.09,
            "xanchor": "center",
            "x": 0.5,
        },
        hovermode="x unified",
    )
    for row_index, (_, _, title, _) in enumerate(row_specs, start=1):
        fig.update_yaxes(title_text=title, row=row_index, col=1)
        for col in (2, 3):
            fig.update_yaxes(title_text=None, row=row_index, col=col)
    for col in (1, 2, 3):
        fig.update_xaxes(title_text="Time from onset (s)", row=3, col=col)
    return fig


def render_summary_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact Markdown summary for the materialized comparisons."""

    lines = [
        "# Steady-State Perturbation Bank",
        "",
        "This materialization probes GRU feedback sensitivity around a hold-at-target "
        "endpoint state. It does not rerun the reach-context perturbation bank.",
        "",
        "## Wash-In Contract",
        "",
        f"- Fan-out policy: {manifest['washin_contract']['fanout_policy']}",
        (
            "- Delayed rows use a 10-step pre-go prefix followed by 30 post-go hold "
            "steps, then preserve a 50-step post-onset response window."
        ),
        (
            "- Undelayed rows use a 30-step immediate hold prefix and extend short "
            "hold-at-target validation trials when needed, rather than shortening "
            "the 50-step post-onset window."
        ),
        (
            "- Default pulse shape: "
            f"{manifest['pulse_duration_steps']} steps; "
            f"position={manifest['feedback_offset_scales']['position_m']} m, "
            f"velocity={manifest['feedback_offset_scales']['velocity_m_s']} m/s, "
            f"force/filter={manifest['feedback_offset_scales']['force_filter']}."
        ),
        (
            "- Output, position, and velocity rows show primary aligned traces plus "
            "lower-emphasis orthogonal companion traces. The orthogonal trace uses the "
            "same signed direction rotated +90 degrees in the right-handed x-y plane."
        ),
        "",
        "## Comparisons",
        "",
    ]
    for comparison_id, comparison in manifest["comparisons"].items():
        lines.append(f"### `{comparison_id}`")
        lines.append("")
        lines.append(f"- Figure: `{comparison['figure']['spec_path']}`")
        lines.append(f"- Source run: `{comparison['source_experiment']}/{comparison['run_id']}`")
        lines.append("")
        lines.append("| Condition | Response label | Baseline command | Peak output by family |")
        lines.append("|---|---:|---:|---|")
        for condition in comparison["conditions"].values():
            command = condition["washin"]["baseline_command_magnitude"]["mean"]
            peaks = ", ".join(
                f"{family}={payload['peak_output_response']:.4g}"
                for family, payload in condition["family_summary"].items()
            )
            lines.append(
                f"| {condition['label']} | {condition['response_label']} | "
                f"{command:.4g} | {peaks} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _evaluate_model_on_trial_specs(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
) -> RolloutEvaluation:
    from rlrmp.eval.perturbation_bank import (
        _evaluate_model_on_trial_specs as evaluate,
    )

    return evaluate(
        model=model,
        task=task,
        trial_specs=trial_specs,
        n_replicates=n_replicates,
        seed=seed,
    )


def _ensure_rlrmp_registered() -> None:
    """Register rlrmp figure routing when entry-point discovery is unavailable."""

    from feedbax.plugins import EXPERIMENT_REGISTRY
    from rlrmp import register_experiment_package

    try:
        EXPERIMENT_REGISTRY.get_figure_routing("rlrmp")
    except ValueError:
        register_experiment_package(EXPERIMENT_REGISTRY)


def _target_position(run: RunFigureInputs, trial_specs: Any) -> np.ndarray:
    card_target = run.run_spec.get("game_card", {}).get("target_pos_m")
    if card_target is not None:
        return np.asarray(card_target, dtype=np.float64)
    if "target" in trial_specs.inputs:
        target = np.asarray(trial_specs.inputs["target"], dtype=np.float64)
        return target.reshape(-1, target.shape[-1])[-1]
    if "effector_target" in trial_specs.inputs:
        target = np.asarray(trial_specs.inputs["effector_target"].pos, dtype=np.float64)
        return target.reshape(-1, target.shape[-1])[-1]
    raise ValueError(f"could not resolve target position for {run.run_id}")


def _trial_horizon(trial_specs: Any) -> int:
    for key in ("target", "sisu", "effector_target"):
        if key in trial_specs.inputs:
            value = trial_specs.inputs[key]
            if hasattr(value, "pos"):
                return int(np.asarray(value.pos).shape[1])
            arr = np.asarray(value)
            if arr.ndim >= 2:
                return int(arr.shape[1])
    raise ValueError("could not infer trial horizon")


def _feedback_dim(trial_specs: Any) -> int:
    for key in (
        "perturbation_training.sensory_feedback",
        "perturbation.channel.sensory_feedback",
    ):
        if key in trial_specs.inputs:
            return int(np.asarray(trial_specs.inputs[key]).shape[-1])
    for key, value in trial_specs.inputs.items():
        if str(key).endswith("sensory_feedback"):
            return int(np.asarray(value).shape[-1])
    raise ValueError("could not infer sensory feedback dimension")


def _expected_feedback_dim_from_hps(hps: Any) -> int | None:
    contract = getattr(getattr(hps, "target_relative_multitarget", None), "input_contract", None)
    shape = getattr(contract, "shape", None)
    if shape:
        return int(shape[-1])
    return None


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


def _update_summary_markdown(path: Path, content: str) -> None:
    """Update the generated Markdown block, migrating the legacy whole-file summary."""

    marker = f"<!-- AUTO-GENERATED: {SUMMARY_MARKER} -->"
    if path.exists():
        text = path.read_text()
        marker_pos = text.find(marker)
        if marker_pos >= 0 and marker_pos > 0 and text[:marker_pos].strip() == "":
            path.write_text(text[marker_pos:])
        elif marker not in text and text.startswith("# Steady-State Perturbation Bank\n"):
            path.unlink()
    update_marked_section(path, SUMMARY_MARKER, content)


def _condition_color(index: int) -> str:
    colors = ("#2563eb", "#dc2626", "#16a34a", "#7c3aed")
    return colors[index % len(colors)]


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


__all__ = [
    "ComparisonSpec",
    "ConditionSpec",
    "FeedbackPerturbation",
    "SteadyStatePerturbationBankConfig",
    "aggregate_family_profiles",
    "default_feedback_perturbations",
    "identity_condition",
    "make_steady_state_trial_specs",
    "materialize_steady_state_comparisons",
    "pad_feedback_offset_inputs",
    "right_handed_orthogonal_direction",
    "slim_steady_state_manifest",
    "sisu_condition",
    "signed_pair_antisymmetry",
    "washin_diagnostics",
]
