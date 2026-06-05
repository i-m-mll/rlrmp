"""Feedback-ablation diagnostics for validation-selected C&S GRU checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax.tree as jt
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax.graph import Component, Wire
from feedbax.types import TreeNamespace, dict_to_namespace
from jaxtyping import PRNGKeyArray, PyTree

from rlrmp.analysis.gru_checkpoint_selection import (
    available_checkpoint_batches,
    checkpoint_path_for_batches,
    load_validation_selected_checkpoint_model,
)
from rlrmp.analysis.gru_evaluation_diagnostics import RolloutEvaluation
from rlrmp.analysis.gru_perturbation_bank import (
    apply_perturbation_to_trial_specs,
    default_cs_perturbation_bank,
    delta_full_qrf_cost_summary,
    full_qrf_cost_summary,
)
from rlrmp.analysis.gru_pilot_figures import (
    RunFigureInputs,
    initial_effector_velocity,
    repeat_single_validation_trial,
    resolve_run_inputs,
)
from rlrmp.analysis.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p


SCHEMA_VERSION = "rlrmp.gru_feedback_ablation.v1"
FEEDBACK_SELECTION_SCHEMA_VERSION = "rlrmp.gru_feedback_checkpoint_selection_audit.v1"
DEFAULT_SOURCE_EXPERIMENT = "aacb9ed"
DEFAULT_RESULT_EXPERIMENT = "57ab156"
DEFAULT_SCOPE = "fixed_target_random_perturb_validation_selected"
DEFAULT_RUN_IDS = (
    "fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64",
    "fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64",
)
DEFAULT_NOTE_FILENAME = "gru_feedback_ablation_fixed_target_random_perturb_validation_selected.md"
DEFAULT_OUTPUT_FILENAME = (
    "gru_feedback_ablation_fixed_target_random_perturb_validation_selected.json"
)

AblationMode = Literal[
    "normal",
    "frozen_nominal_observation_tape",
    "zeroed_perturbation_observation_deviation",
    "shuffled_observation_history",
    "lagged_observation_history",
    "position_only_observation",
    "velocity_only_observation",
]

EvaluationBin = Literal[
    "nominal",
    "initial_state",
    "process_epsilon",
    "sensory_feedback",
    "delayed_observation",
]

OBSERVATION_ABLATION_INPUT_PREFIX = "feedback_ablation"
OBSERVATION_DIM = 4
POSITION_ONLY_MASK = (1.0, 1.0, 0.0, 0.0)
VELOCITY_ONLY_MASK = (0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class ObservationAblationSpec:
    """Graph-level delayed-feedback ablation configuration."""

    mode: AblationMode
    label: str
    input_key: str | None = None
    mask: tuple[float, ...] | None = None

    @property
    def requires_payload(self) -> bool:
        """Return whether this ablation consumes an external observation tape."""

        return self.input_key is not None

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable ablation provenance."""

        return {
            "mode": self.mode,
            "label": self.label,
            "input_key": self.input_key,
            "mask": None if self.mask is None else list(self.mask),
            "intervention_point": "sensory.output -> net.feedback",
            "controller_weights_mutated": False,
            "checkpoint_selection_role": "validation_selected_per_replicate",
            "analytical_metrics_role": "audit_only_not_used_for_checkpoint_selection",
        }


@dataclass(frozen=True)
class DetailedRolloutEvaluation:
    """Rollout arrays needed for feedback-ablation scoring.

    Array shapes follow ``[replicate, trial, time, feature]`` unless stated
    otherwise.
    """

    rollout: RolloutEvaluation
    feedback: np.ndarray
    mechanics_vector: np.ndarray


class ObservationOverride(Component):
    """Replace a delayed-feedback graph signal with an external tape."""

    input_ports = ("signal", "replacement")
    output_ports = ("feedback",)

    label: str = eqx.field(static=True)

    def __init__(self, *, label: str):
        self.label = str(label)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        del key
        return {"feedback": inputs["replacement"]}, state


class ObservationMask(Component):
    """Apply a fixed mask to the delayed-feedback graph signal."""

    input_ports = ("signal",)
    output_ports = ("feedback",)

    label: str = eqx.field(static=True)
    mask: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, *, label: str, mask: Sequence[float]):
        self.label = str(label)
        self.mask = tuple(float(value) for value in mask)
        if len(self.mask) != OBSERVATION_DIM:
            raise ValueError(f"observation mask must have {OBSERVATION_DIM} entries")

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        del key
        return {
            "feedback": jnp.asarray(inputs["signal"]) * jnp.asarray(self.mask),
        }, state


def default_ablation_modes() -> tuple[AblationMode, ...]:
    """Return the standard feedback-ablation mode order."""

    return (
        "normal",
        "frozen_nominal_observation_tape",
        "zeroed_perturbation_observation_deviation",
        "shuffled_observation_history",
        "lagged_observation_history",
        "position_only_observation",
        "velocity_only_observation",
    )


def selected_feedback_ablation_bins() -> dict[EvaluationBin, str | None]:
    """Return representative perturbation rows for the standard evaluation bins."""

    return {
        "nominal": None,
        "initial_state": "initial_position_offset__x_pos",
        "process_epsilon": "process_epsilon_pulse__position_x__t40_pos",
        "sensory_feedback": "sensory_feedback_offset__x_pos",
        "delayed_observation": "delayed_observation_offset__x_pos",
    }


def build_observation_tape(
    mode: AblationMode,
    *,
    bin_feedback: np.ndarray,
    nominal_feedback: np.ndarray,
) -> np.ndarray | None:
    """Return the external delayed-feedback tape for a payload-based ablation."""

    feedback = np.asarray(bin_feedback, dtype=np.float64)
    nominal = np.asarray(nominal_feedback, dtype=np.float64)
    if feedback.shape != nominal.shape:
        raise ValueError(
            "bin_feedback and nominal_feedback must have the same shape; "
            f"got {feedback.shape} and {nominal.shape}"
        )
    if mode == "normal":
        return None
    if mode in {
        "frozen_nominal_observation_tape",
        "zeroed_perturbation_observation_deviation",
    }:
        return nominal
    if mode == "shuffled_observation_history":
        if feedback.shape[1] < 2:
            return feedback.copy()
        return np.roll(feedback, shift=1, axis=1)
    if mode == "lagged_observation_history":
        first = feedback[:, :, :1, :]
        return np.concatenate([first, feedback[:, :, :-1, :]], axis=2)
    if mode in {"position_only_observation", "velocity_only_observation"}:
        return None
    raise ValueError(f"unsupported feedback ablation mode {mode!r}")


def build_observation_ablation_spec(
    mode: AblationMode,
    *,
    bin_id: str,
) -> ObservationAblationSpec:
    """Return graph-adapter metadata for one ablation/bin pair."""

    safe_bin = bin_id.replace("/", "_").replace(":", "_")
    label = f"feedback_ablation_{mode}_{safe_bin}"
    if mode == "position_only_observation":
        return ObservationAblationSpec(mode=mode, label=label, mask=POSITION_ONLY_MASK)
    if mode == "velocity_only_observation":
        return ObservationAblationSpec(mode=mode, label=label, mask=VELOCITY_ONLY_MASK)
    if mode == "normal":
        return ObservationAblationSpec(mode=mode, label=label)
    return ObservationAblationSpec(
        mode=mode,
        label=label,
        input_key=f"{OBSERVATION_ABLATION_INPUT_PREFIX}:{mode}:{safe_bin}",
    )


def insert_observation_ablation(model: Any, spec: ObservationAblationSpec) -> Any:
    """Insert a feedback-ablation node on ``sensory.output -> net.feedback``."""

    if spec.mode == "normal":
        return model
    if spec.label in getattr(model, "nodes", {}):
        return model
    old_wire = Wire("sensory", "output", "net", "feedback")
    graph = model.remove_wire(old_wire)
    node: Component
    if spec.requires_payload:
        node = ObservationOverride(label=spec.label)
    else:
        if spec.mask is None:
            raise ValueError(f"ablation mode {spec.mode!r} has no payload or mask")
        node = ObservationMask(label=spec.label, mask=spec.mask)
    graph = graph.add_node(spec.label, node)
    graph = graph.add_wire(Wire("sensory", "output", spec.label, "signal"))
    graph = graph.add_wire(Wire(spec.label, "feedback", "net", "feedback"))
    if spec.requires_payload:
        graph = eqx.tree_at(
            lambda g: g.input_ports,
            graph,
            (*graph.input_ports, spec.input_key),
        )
        graph = eqx.tree_at(
            lambda g: g.input_bindings,
            graph,
            {**graph.input_bindings, spec.input_key: (spec.label, "replacement")},
        )
    return graph


def summarize_ablation_delta(
    *,
    baseline: DetailedRolloutEvaluation,
    ablated: DetailedRolloutEvaluation,
    baseline_cost: Mapping[str, Any],
    ablated_cost: Mapping[str, Any],
) -> dict[str, Any]:
    """Return compact per-ablation deltas against a normal-observation baseline."""

    return {
        "baseline_action_norm": _summary_stats(
            np.linalg.norm(baseline.rollout.command, axis=-1)
        ),
        "ablated_action_norm": _summary_stats(
            np.linalg.norm(ablated.rollout.command, axis=-1)
        ),
        "delta_action_norm": _summary_stats(
            np.linalg.norm(ablated.rollout.command - baseline.rollout.command, axis=-1)
        ),
        "delta_observation_norm": _summary_stats(
            np.linalg.norm(ablated.feedback - baseline.feedback, axis=-1)
        ),
        "delta_endpoint_error_m": _summary_stats(
            _endpoint_error(ablated.rollout) - _endpoint_error(baseline.rollout)
        ),
        "delta_terminal_speed_m_s": _summary_stats(
            _terminal_speed(ablated.rollout) - _terminal_speed(baseline.rollout)
        ),
        "rollout_full_qrf": delta_full_qrf_cost_summary(baseline_cost, ablated_cost),
        "validation_objective": {
            "status": "substitute",
            "substitute": "paired_realized_deterministic_rollout_full_qrf.total",
            "reason": (
                "checkpoint validation objective is a selection scalar in the saved "
                "manifest; post-hoc ablation rollouts are scored with the realized "
                "full-Q/R/Q_f substitute on the same rollout bank"
            ),
        },
    }


def summarize_normalized_feedback_use(
    rows: Sequence[Mapping[str, Any]],
    *,
    open_loop_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return normalized feedback-use indices from evaluated ablation rows.

    The returned indices are audit metrics. Missing denominators and unavailable
    open-loop references are represented explicitly rather than silently
    converted to zero.
    """

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    warnings: list[str] = []
    ablation_dependence = _ablation_dependence_index(evaluated, warnings=warnings)
    perturbation_rescue = _perturbation_rescue_index(evaluated, warnings=warnings)
    correction = _correction_index_vs_open_loop(
        evaluated,
        open_loop_reference=open_loop_reference,
        warnings=warnings,
    )
    available_scores = [
        metric.get("value")
        for metric in (ablation_dependence, perturbation_rescue, correction)
        if metric.get("status") == "available"
    ]
    score = (
        float(np.nanmean(np.asarray(available_scores, dtype=np.float64)))
        if available_scores
        else None
    )
    return {
        "status": "available" if available_scores else "not_available",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "score": score,
        "score_components": [
            name
            for name, metric in (
                ("ablation_dependence_index", ablation_dependence),
                ("perturbation_rescue_index", perturbation_rescue),
                ("correction_index_vs_open_loop", correction),
            )
            if metric.get("status") == "available"
        ],
        "ablation_dependence_index": ablation_dependence,
        "perturbation_rescue_index": perturbation_rescue,
        "correction_index_vs_open_loop": correction,
        "warnings": warnings,
    }


def feedback_checkpoint_selection_audit(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an audit-only feedback-selected checkpoint sidecar.

    Prefer a per-replicate checkpoint rescore when it has been materialized.
    Older manifests without that pass fall back to the legacy run-level summary
    so consumers get an explicit status rather than silent primary-selection
    changes.
    """

    runs = manifest.get("runs", {})
    if not isinstance(runs, Mapping) or not runs:
        return {
            "schema_version": FEEDBACK_SELECTION_SCHEMA_VERSION,
            "status": "not_available",
            "reason": "no_feedback_ablation_runs_available",
            "selection_use": "audit_only_not_primary_checkpoint_selection",
            "primary_checkpoint_policy": manifest.get("checkpoint_policy"),
        }
    checkpoint_audits = {
        str(run_id): run.get("feedback_checkpoint_rescore")
        for run_id, run in runs.items()
        if isinstance(run, Mapping)
    }
    materialized = {
        run_id: audit
        for run_id, audit in checkpoint_audits.items()
        if isinstance(audit, Mapping) and audit.get("status") == "materialized"
    }
    if materialized:
        return {
            "schema_version": FEEDBACK_SELECTION_SCHEMA_VERSION,
            "status": "materialized",
            "selection_use": "audit_only_not_primary_checkpoint_selection",
            "primary_checkpoint_policy": manifest.get("checkpoint_policy"),
            "feedback_selection_policy": (
                "per-replicate minimum mean feedback-bank full-Q/R/Q_f delta cost"
            ),
            "candidate_granularity": "checkpoint_batch_per_replicate",
            "runs": materialized,
            "note": (
                "Feedback-selected checkpoints are reported for audit only. The primary "
                "materialization path still loads validation-selected checkpoints."
            ),
        }

    candidates: list[dict[str, Any]] = []
    missing: list[str] = []
    for run_id, run in runs.items():
        if not isinstance(run, Mapping):
            missing.append(str(run_id))
            continue
        normalized = run.get("normalized_feedback_use", {})
        score = normalized.get("score") if isinstance(normalized, Mapping) else None
        score_float = _coerce_reference_float(score)
        if score_float is None:
            missing.append(str(run_id))
            continue
        candidates.append(
            {
                "run_id": str(run_id),
                "label": run.get("label"),
                "feedback_score": score_float,
                "score_components": list(normalized.get("score_components", ())),
                "checkpoint_selection": run.get("checkpoint_selection", []),
            }
        )
    if missing:
        return {
            "schema_version": FEEDBACK_SELECTION_SCHEMA_VERSION,
            "status": "not_available",
            "reason": "feedback_scores_missing_for_some_candidates",
            "missing_candidates": missing,
            "n_candidates": len(runs),
            "n_scored_candidates": len(candidates),
            "selection_use": "audit_only_not_primary_checkpoint_selection",
            "primary_checkpoint_policy": manifest.get("checkpoint_policy"),
        }
    selected = max(candidates, key=lambda candidate: candidate["feedback_score"])
    return {
        "schema_version": FEEDBACK_SELECTION_SCHEMA_VERSION,
        "status": "available",
        "selection_use": "audit_only_not_primary_checkpoint_selection",
        "primary_checkpoint_policy": manifest.get("checkpoint_policy"),
        "feedback_selection_policy": "max_normalized_feedback_use_score",
        "candidate_granularity": "run_legacy_fallback",
        "selected_candidate": selected,
        "candidates": sorted(
            candidates,
            key=lambda candidate: candidate["feedback_score"],
            reverse=True,
        ),
        "note": (
            "Feedback-selected candidates are reported for audit only. The primary "
            "materialization path still loads validation-selected checkpoints."
        ),
    }


def interpret_run_feedback_ablation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify the run-level ablation pattern."""

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    if not evaluated:
        return {
            "label": "inconclusive",
            "reason": "no ablation rows were evaluated",
        }

    action_means = {
        (str(row["bin"]), str(row["mode"])): _metric_mean(row, "delta_action_norm")
        for row in evaluated
    }
    perturb_sensitive = [
        value
        for (bin_id, mode), value in action_means.items()
        if bin_id != "nominal"
        and mode
        in {
            "frozen_nominal_observation_tape",
            "zeroed_perturbation_observation_deviation",
            "lagged_observation_history",
        }
        and value is not None
    ]
    channel_position = [
        value
        for (_bin_id, mode), value in action_means.items()
        if mode == "position_only_observation" and value is not None
    ]
    channel_velocity = [
        value
        for (_bin_id, mode), value in action_means.items()
        if mode == "velocity_only_observation" and value is not None
    ]
    max_feedback_delta = max(perturb_sensitive) if perturb_sensitive else 0.0
    max_channel_delta = (
        max(channel_position + channel_velocity)
        if channel_position or channel_velocity
        else 0.0
    )
    if max_feedback_delta >= 1e-2:
        return {
            "label": "feedback_sensitive",
            "reason": "observation tape ablations changed actions above the diagnostic threshold",
            "max_feedback_delta_action_norm_mean": max_feedback_delta,
            "max_channel_delta_action_norm_mean": max_channel_delta,
        }
    if max_channel_delta >= 1e-2:
        return {
            "label": "channel_sensitive",
            "reason": "position-only or velocity-only masks changed actions above the threshold",
            "max_feedback_delta_action_norm_mean": max_feedback_delta,
            "max_channel_delta_action_norm_mean": max_channel_delta,
        }
    if evaluated:
        return {
            "label": "motor_tape_like",
            "reason": "all evaluated feedback ablations produced small action changes",
            "max_feedback_delta_action_norm_mean": max_feedback_delta,
            "max_channel_delta_action_norm_mean": max_channel_delta,
        }
    return {"label": "inconclusive", "reason": "insufficient evaluated ablation rows"}


def materialize_gru_feedback_ablation(
    *,
    source_experiment: str = DEFAULT_SOURCE_EXPERIMENT,
    result_experiment: str = DEFAULT_RESULT_EXPERIMENT,
    scope: str = DEFAULT_SCOPE,
    run_ids: Sequence[str] = DEFAULT_RUN_IDS,
    labels: Sequence[str] | None = None,
    n_rollout_trials: int = 4,
    include_checkpoint_rescore: bool = True,
    output_path: Path | None = None,
    note_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize feedback-ablation JSON and Markdown sidecars."""

    result_notes_dir = repo_root / "results" / result_experiment / "notes"
    mkdir_p(result_notes_dir)
    output_path = output_path or result_notes_dir / DEFAULT_OUTPUT_FILENAME
    note_path = note_path or result_notes_dir / DEFAULT_NOTE_FILENAME
    run_inputs = resolve_run_inputs(
        experiment=source_experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )
    runs = {
        run.run_id: evaluate_run_feedback_ablation(
            run,
            source_experiment=source_experiment,
            n_rollout_trials=n_rollout_trials,
            include_checkpoint_rescore=include_checkpoint_rescore,
            repo_root=repo_root,
        )
        for run in run_inputs
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": result_experiment,
        "source_experiment": source_experiment,
        "scope": scope,
        "labels": None if labels is None else list(labels),
        "checkpoint_policy": "validation_selected_per_replicate",
        "selection_role": "validation_selected_checkpoints_only",
        "feedback_checkpoint_rescore_policy": (
            "audit_only_per_replicate_argmin_on_feedback_bank"
            if include_checkpoint_rescore
            else "disabled"
        ),
        "analytical_action_io_metrics_role": "audit_only_not_used_for_checkpoint_selection",
        "ablation_modes": list(default_ablation_modes()),
        "evaluation_bins": selected_feedback_ablation_bins(),
        "runs": runs,
    }
    manifest["feedback_checkpoint_selection_audit"] = feedback_checkpoint_selection_audit(manifest)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    note_path.write_text(render_feedback_ablation_markdown(manifest))
    return manifest


def evaluate_run_feedback_ablation(
    run: RunFigureInputs,
    *,
    source_experiment: str,
    n_rollout_trials: int,
    include_checkpoint_rescore: bool = True,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Evaluate feedback ablations for one validation-selected GRU run."""

    if n_rollout_trials < 1:
        raise ValueError("n_rollout_trials must be at least 1")
    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run.run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=source_experiment,
        run_id=run.run_id,
        run_spec=run.run_spec,
        repo_root=repo_root,
    )
    base_trial_specs = repeat_single_validation_trial(pair.task.validation_trials, n_rollout_trials)
    perturbations = {
        str(row["perturbation_id"]): row for row in default_cs_perturbation_bank()["perturbations"]
    }
    nominal = _evaluate_model_on_trial_specs(
        model=model,
        task=pair.task,
        trial_specs=base_trial_specs,
        n_replicates=n_replicates,
        seed=0,
    )
    rows: list[dict[str, Any]] = []
    for bin_id, perturbation_id in selected_feedback_ablation_bins().items():
        trial_specs = base_trial_specs
        bin_model = model
        adapter_json: dict[str, Any] | None = None
        if perturbation_id is not None:
            perturbation = perturbations[perturbation_id]
            adapter = apply_perturbation_to_trial_specs(
                base_trial_specs,
                perturbation,
                model=model,
            )
            adapter_json = adapter.to_json()
            if adapter.status != "evaluated":
                rows.extend(
                    _not_available_rows(
                        bin_id=bin_id,
                        perturbation_id=perturbation_id,
                        reason=adapter.reason or f"{bin_id} perturbation adapter was not evaluated",
                        adapter=adapter_json,
                    )
                )
                continue
            trial_specs = adapter.trial_specs
            bin_model = adapter.model if adapter.model is not None else model
        baseline = _evaluate_model_on_trial_specs(
            model=bin_model,
            task=pair.task,
            trial_specs=trial_specs,
            n_replicates=n_replicates,
            seed=0,
        )
        baseline_cost = full_qrf_cost_summary(baseline.rollout, trial_specs)
        for mode in default_ablation_modes():
            spec = build_observation_ablation_spec(mode, bin_id=bin_id)
            if mode == "normal":
                rows.append(
                    {
                        "bin": bin_id,
                        "mode": mode,
                        "perturbation_id": perturbation_id,
                        "status": "evaluated",
                        "ablation": spec.to_json(),
                        "adapter": adapter_json,
                        "metrics": summarize_ablation_delta(
                            baseline=baseline,
                            ablated=baseline,
                            baseline_cost=baseline_cost,
                            ablated_cost=baseline_cost,
                        ),
                    }
                )
                continue
            tape = build_observation_tape(
                mode,
                bin_feedback=baseline.feedback,
                nominal_feedback=nominal.feedback,
            )
            ablated_trial_specs = trial_specs
            if tape is not None:
                ablated_trial_specs = _add_trial_input(
                    trial_specs,
                    str(spec.input_key),
                    jnp.asarray(tape, dtype=jnp.float64),
                )
            try:
                ablated_model = insert_observation_ablation(bin_model, spec)
            except ValueError as exc:
                rows.append(
                    {
                        "bin": bin_id,
                        "mode": mode,
                        "perturbation_id": perturbation_id,
                        "status": "not_available",
                        "reason": str(exc),
                        "ablation": spec.to_json(),
                        "adapter": adapter_json,
                    }
                )
                continue
            ablated = _evaluate_model_on_trial_specs(
                model=ablated_model,
                task=pair.task,
                trial_specs=ablated_trial_specs,
                n_replicates=n_replicates,
                seed=0,
            )
            ablated_cost = full_qrf_cost_summary(ablated.rollout, ablated_trial_specs)
            rows.append(
                {
                    "bin": bin_id,
                    "mode": mode,
                    "perturbation_id": perturbation_id,
                    "status": "evaluated",
                    "ablation": spec.to_json(),
                    "adapter": adapter_json,
                    "metrics": summarize_ablation_delta(
                        baseline=baseline,
                        ablated=ablated,
                        baseline_cost=baseline_cost,
                        ablated_cost=ablated_cost,
                    ),
                }
            )
    run_result = {
        "label": run.label,
        "run_spec_path": _repo_relative(run.run_spec_path, repo_root=repo_root),
        "artifact_dir": _repo_relative(run.artifact_dir, repo_root=repo_root),
        "checkpoint_selection": [
            selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
        ],
        "n_replicates": n_replicates,
        "n_rollout_trials_per_replicate": n_rollout_trials,
        "n_time_steps": int(nominal.rollout.command.shape[2]),
        "dt_s": float(nominal.rollout.dt),
        "status_counts": _status_counts(rows),
        "interpretation": interpret_run_feedback_ablation(rows),
        "normalized_feedback_use": summarize_normalized_feedback_use(rows),
        "ablations": rows,
    }
    if include_checkpoint_rescore:
        run_result["feedback_checkpoint_rescore"] = feedback_checkpoint_rescore_audit_for_run(
            run=run,
            source_experiment=source_experiment,
            pair=pair,
            base_trial_specs=base_trial_specs,
            perturbations=perturbations,
            validation_checkpoint_selection=checkpoint_selection,
            n_replicates=n_replicates,
            repo_root=repo_root,
        )
    else:
        run_result["feedback_checkpoint_rescore"] = {
            "status": "skipped",
            "reason": "disabled_by_materializer",
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }
    return run_result


def feedback_checkpoint_rescore_audit_for_run(
    *,
    run: RunFigureInputs,
    source_experiment: str,
    pair: Any,
    base_trial_specs: Any,
    perturbations: Mapping[str, Mapping[str, Any]],
    validation_checkpoint_selection: Sequence[Any],
    n_replicates: int,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Score every durable checkpoint on a declared feedback perturbation bank."""

    del source_experiment
    checkpoint_batches = available_checkpoint_batches(run.artifact_dir)
    if not checkpoint_batches:
        return {
            "status": "not_available",
            "reason": f"no numbered checkpoints under {run.artifact_dir / 'checkpoints'}",
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }

    scored_rows: list[dict[str, Any]] = []
    for checkpoint_batch in checkpoint_batches:
        checkpoint_path = checkpoint_path_for_batches(run.artifact_dir, checkpoint_batch)
        model = eqx.tree_deserialise_leaves(checkpoint_path / "model.eqx", pair.model)
        scored_rows.append(
            _score_feedback_checkpoint_batch(
                model=model,
                task=pair.task,
                trial_specs=base_trial_specs,
                perturbations=perturbations,
                checkpoint_batch=checkpoint_batch,
                checkpoint_path=checkpoint_path,
                n_replicates=n_replicates,
                repo_root=repo_root,
            )
        )

    selected = []
    validation_by_replicate = {
        int(selection.replicate): selection for selection in validation_checkpoint_selection
    }
    for replicate in range(n_replicates):
        candidates = [
            row
            for row in scored_rows
            if row["per_replicate_scores"][replicate]["status"] == "available"
        ]
        if not candidates:
            selected.append(
                {
                    "replicate": replicate,
                    "status": "not_available",
                    "reason": "no checkpoint had an available feedback-bank score",
                }
            )
            continue
        best = min(
            candidates,
            key=lambda row: row["per_replicate_scores"][replicate]["score"],
        )
        validation_selection = validation_by_replicate.get(replicate)
        validation_batch = (
            None if validation_selection is None else validation_selection.checkpoint_batches
        )
        selected.append(
            {
                "replicate": replicate,
                "status": "available",
                "feedback_selected_checkpoint_batches": int(best["checkpoint_batches"]),
                "feedback_score": float(best["per_replicate_scores"][replicate]["score"]),
                "validation_selected_checkpoint_batches": validation_batch,
                "feedback_minus_validation_batches": (
                    None
                    if validation_batch is None
                    else int(best["checkpoint_batches"]) - int(validation_batch)
                ),
                "n_available_feedback_bins": int(
                    best["per_replicate_scores"][replicate]["n_available_bins"]
                ),
            }
        )

    return {
        "status": "materialized",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "selection_policy": (
            "per-replicate argmin of mean full-Q/R/Q_f delta cost over evaluated "
            "feedback-bank perturbation bins"
        ),
        "feedback_bank_bins": [
            bin_id
            for bin_id, perturbation_id in selected_feedback_ablation_bins().items()
            if perturbation_id is not None
        ],
        "n_checkpoint_candidates": len(scored_rows),
        "checkpoint_scores": scored_rows,
        "feedback_selected_checkpoints": selected,
        "primary_checkpoint_policy": "validation_selected_per_replicate",
    }


def _score_feedback_checkpoint_batch(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    perturbations: Mapping[str, Mapping[str, Any]],
    checkpoint_batch: int,
    checkpoint_path: Path,
    n_replicates: int,
    repo_root: Path,
) -> dict[str, Any]:
    """Score one checkpoint on normal-controller feedback perturbation bins."""

    baseline = _evaluate_model_on_trial_specs(
        model=model,
        task=task,
        trial_specs=trial_specs,
        n_replicates=n_replicates,
        seed=0,
    )
    baseline_cost = full_qrf_cost_summary(baseline.rollout, trial_specs)
    bin_scores = []
    per_replicate_values: list[list[float]] = [[] for _ in range(n_replicates)]
    for bin_id, perturbation_id in selected_feedback_ablation_bins().items():
        if perturbation_id is None:
            continue
        adapter = apply_perturbation_to_trial_specs(
            trial_specs,
            perturbations[perturbation_id],
            model=model,
        )
        if adapter.status != "evaluated":
            bin_scores.append(
                {
                    "bin": bin_id,
                    "perturbation_id": perturbation_id,
                    "status": adapter.status,
                    "reason": adapter.reason,
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
        perturbed_cost = full_qrf_cost_summary(perturbed.rollout, adapter.trial_specs)
        delta_cost = delta_full_qrf_cost_summary(baseline_cost, perturbed_cost)
        values = _per_replicate_cost_delta_values(
            baseline_cost,
            perturbed_cost,
            n_replicates=n_replicates,
        )
        for replicate, value in enumerate(values):
            if value is not None:
                per_replicate_values[replicate].append(float(value))
        bin_scores.append(
            {
                "bin": bin_id,
                "perturbation_id": perturbation_id,
                "status": delta_cost.get("status", "not_available"),
                "mean_delta_full_qrf_cost": _summary_mean(
                    delta_cost.get("delta_cost", {}).get("total", {})
                ),
                "per_replicate_mean_delta_full_qrf_cost": values,
            }
        )
    per_replicate_scores = []
    for replicate, values in enumerate(per_replicate_values):
        if not values:
            per_replicate_scores.append(
                {
                    "replicate": replicate,
                    "status": "not_available",
                    "reason": "no evaluated feedback bins",
                    "n_available_bins": 0,
                }
            )
        else:
            per_replicate_scores.append(
                {
                    "replicate": replicate,
                    "status": "available",
                    "score": float(np.mean(np.asarray(values, dtype=np.float64))),
                    "n_available_bins": len(values),
                }
            )
    return {
        "checkpoint_batches": int(checkpoint_batch),
        "checkpoint_path": _repo_relative(checkpoint_path, repo_root=repo_root),
        "per_replicate_scores": per_replicate_scores,
        "bin_scores": bin_scores,
    }


def _per_replicate_cost_delta_values(
    base_cost: Mapping[str, Any],
    perturbed_cost: Mapping[str, Any],
    *,
    n_replicates: int,
) -> list[float | None]:
    """Return mean total delta cost per replicate from a paired cost summary."""

    if base_cost.get("status") != "available" or perturbed_cost.get("status") != "available":
        return [None for _ in range(n_replicates)]
    base_values = np.asarray(base_cost.get("total", {}).get("values"), dtype=np.float64)
    perturbed_values = np.asarray(
        perturbed_cost.get("total", {}).get("values"),
        dtype=np.float64,
    )
    if (
        base_values.shape[:1] != (n_replicates,)
        or perturbed_values.shape[:1] != (n_replicates,)
        or base_values.shape != perturbed_values.shape
    ):
        return [None for _ in range(n_replicates)]
    values = perturbed_values - base_values
    if values.ndim == 1:
        return [float(value) for value in values]
    reduced = np.mean(values.reshape((n_replicates, -1)), axis=1)
    return [float(value) for value in reduced]


def render_feedback_ablation_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact Markdown sidecar for a feedback-ablation manifest."""

    lines = [
        "# GRU Feedback Ablation Diagnostic",
        "",
        f"- Issue: `{manifest['issue']}`",
        f"- Source experiment: `{manifest['source_experiment']}`",
        f"- Scope: `{manifest['scope']}`",
        f"- Checkpoint policy: `{manifest['checkpoint_policy']}`",
        "",
        "## Interpretation",
        "",
        "| Run | Label | Reason |",
        "|---|---|---|",
    ]
    for run_id, run in manifest["runs"].items():
        interpretation = run.get("interpretation", {})
        lines.append(
            f"| `{run_id}` | `{interpretation.get('label', 'inconclusive')}` | "
            f"{interpretation.get('reason', '')} |"
        )
    lines.extend(
        [
            "",
            "## Ablation Deltas",
            "",
            "| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | "
            "dEndpoint mean | dTerminal speed mean |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for run_id, run in manifest["runs"].items():
        for row in run.get("ablations", []):
            metrics = row.get("metrics", {})
            full_qrf = metrics.get("rollout_full_qrf", {})
            delta_cost = full_qrf.get("delta_cost", {}) if isinstance(full_qrf, Mapping) else {}
            lines.append(
                f"| `{run_id}` | `{row.get('bin')}` | `{row.get('mode')}` | "
                f"{row.get('status')} | "
                f"{_format_float(_summary_mean(metrics.get('delta_action_norm')))} | "
                f"{_format_float(_summary_mean(delta_cost.get('total')))} | "
                f"{_format_float(_summary_mean(metrics.get('delta_endpoint_error_m')))} | "
                f"{_format_float(_summary_mean(metrics.get('delta_terminal_speed_m_s')))} |"
            )
    lines.extend(
        [
            "",
            "## Normalized Feedback-Use Indices",
            "",
            "| Run | Status | Score | Ablation dependence | Perturbation rescue | "
            "Correction vs open-loop | Warnings |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for run_id, run in manifest["runs"].items():
        normalized = run.get("normalized_feedback_use", {})
        lines.append(
            f"| `{run_id}` | {normalized.get('status', 'not_available')} | "
            f"{_format_float(normalized.get('score'))} | "
            f"{_format_float(_index_value(normalized, 'ablation_dependence_index'))} | "
            f"{_format_float(_index_value(normalized, 'perturbation_rescue_index'))} | "
            f"{_format_float(_index_value(normalized, 'correction_index_vs_open_loop'))} | "
            f"{'; '.join(normalized.get('warnings', ())) or 'none'} |"
        )
    audit = manifest.get("feedback_checkpoint_selection_audit", {})
    lines.extend(
        [
            "",
            "## Feedback-Selected Checkpoint Audit",
            "",
            f"- Status: `{audit.get('status', 'not_available')}`",
            "- Selection use: "
            f"`{audit.get('selection_use', 'audit_only_not_primary_checkpoint_selection')}`",
            "- Primary checkpoint policy: "
            f"`{audit.get('primary_checkpoint_policy', manifest.get('checkpoint_policy'))}`",
        ]
    )
    if isinstance(audit, Mapping) and audit.get("status") == "available":
        selected = audit.get("selected_candidate", {})
        if isinstance(selected, Mapping):
            lines.append(f"- Feedback-selected candidate: `{selected.get('run_id')}`")
    elif isinstance(audit, Mapping) and audit.get("status") == "materialized":
        lines.extend(
            [
                "",
                "| Run | Replicate | Validation checkpoint | Feedback checkpoint | "
                "Feedback - validation | Feedback score | Bins |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run_id, audit_run in audit.get("runs", {}).items():
            if not isinstance(audit_run, Mapping):
                continue
            for selection in audit_run.get("feedback_selected_checkpoints", ()):
                if not isinstance(selection, Mapping):
                    continue
                lines.append(
                    f"| `{run_id}` | {selection.get('replicate')} | "
                    f"{selection.get('validation_selected_checkpoint_batches')} | "
                    f"{selection.get('feedback_selected_checkpoint_batches')} | "
                    f"{selection.get('feedback_minus_validation_batches')} | "
                    f"{_format_float(selection.get('feedback_score'))} | "
                    f"{selection.get('n_available_feedback_bins', 0)} |"
                )
    elif isinstance(audit, Mapping) and audit.get("reason"):
        lines.append(f"- Reason: {audit.get('reason')}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `normal` is the per-bin baseline; all other rows are paired deltas against it.",
            "- `frozen_nominal_observation_tape` and "
            "`zeroed_perturbation_observation_deviation` are separate diagnostic lenses "
            "that share the nominal feedback tape for perturbed bins.",
            "- Validation-selected checkpoints are used for model loading. Analytical "
            "action/I/O metrics, where present in adjacent diagnostics, remain audit-only.",
            "",
        ]
    )
    return "\n".join(lines)


def _ablation_dependence_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    candidates: list[float] = []
    for row in rows:
        if row.get("mode") == "normal":
            continue
        metrics = row.get("metrics", {})
        delta_action = _summary_mean(metrics.get("delta_action_norm"))
        baseline_action = _summary_mean(metrics.get("baseline_action_norm"))
        if delta_action is None:
            continue
        if baseline_action is None or abs(baseline_action) <= 1e-12:
            warnings.append(
                "ablation_dependence_index denominator unavailable or near zero for "
                f"{row.get('bin')}:{row.get('mode')}"
            )
            continue
        candidates.append(float(delta_action) / abs(float(baseline_action)))
    if not candidates:
        return {
            "status": "not_available",
            "reason": "no evaluated ablation rows with nonzero baseline action denominator",
        }
    return {
        "status": "available",
        "value": float(max(candidates)),
        "normalization": "max_delta_action_norm_mean_over_baseline_action_norm_mean",
    }


def _perturbation_rescue_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    candidates: list[float] = []
    for row in rows:
        if row.get("bin") == "nominal" or row.get("mode") not in {
            "frozen_nominal_observation_tape",
            "zeroed_perturbation_observation_deviation",
        }:
            continue
        rollout_full_qrf = row.get("metrics", {}).get("rollout_full_qrf", {})
        if (
            not isinstance(rollout_full_qrf, Mapping)
            or rollout_full_qrf.get("status") != "available"
        ):
            warnings.append(
                "perturbation_rescue_index full-Q/R/Q_f denominator unavailable for "
                f"{row.get('bin')}:{row.get('mode')}"
            )
            continue
        delta_total = _summary_mean(rollout_full_qrf.get("delta_cost", {}).get("total", {}))
        perturbed_total = _summary_mean(
            rollout_full_qrf.get("perturbed_cost", {}).get("total", {})
        )
        if delta_total is None:
            continue
        if perturbed_total is None or abs(perturbed_total) <= 1e-12:
            warnings.append(
                "perturbation_rescue_index denominator unavailable or near zero for "
                f"{row.get('bin')}:{row.get('mode')}"
            )
            continue
        candidates.append(float(delta_total) / abs(float(perturbed_total)))
    if not candidates:
        return {
            "status": "not_available",
            "reason": "no perturbation rows with available paired full-Q/R/Q_f costs",
        }
    return {
        "status": "available",
        "value": float(max(candidates)),
        "normalization": "max_ablation_extra_full_qrf_cost_over_ablated_full_qrf_cost",
    }


def _correction_index_vs_open_loop(
    rows: Sequence[Mapping[str, Any]],
    *,
    open_loop_reference: Mapping[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    if open_loop_reference is None:
        warnings.append("correction_index_vs_open_loop not available: open-loop data not supplied")
        return {
            "status": "not_available",
            "reason": "open_loop_reference_not_supplied",
        }
    values = []
    for row in rows:
        if row.get("mode") != "normal" or row.get("bin") == "nominal":
            continue
        bin_id = str(row.get("bin"))
        open_loop_delta = _coerce_reference_float(open_loop_reference.get(bin_id))
        if open_loop_delta is None or abs(open_loop_delta) <= 1e-12:
            warnings.append(
                f"correction_index_vs_open_loop denominator unavailable or near zero for {bin_id}"
            )
            continue
        closed_loop_delta = _summary_mean(
            row.get("metrics", {})
            .get("rollout_full_qrf", {})
            .get("delta_cost", {})
            .get("total", {})
        )
        if closed_loop_delta is None:
            continue
        values.append(
            (float(open_loop_delta) - float(closed_loop_delta))
            / abs(float(open_loop_delta))
        )
    if not values:
        return {
            "status": "not_available",
            "reason": "no perturbation rows had both closed-loop and open-loop cost deltas",
        }
    return {
        "status": "available",
        "value": float(np.nanmean(np.asarray(values, dtype=np.float64))),
        "normalization": "mean_open_loop_minus_closed_loop_delta_cost_over_open_loop_delta_cost",
    }


def _coerce_reference_float(value: Any) -> float | None:
    if isinstance(value, Mapping):
        if "delta_cost" in value:
            return _coerce_reference_float(value["delta_cost"])
        if "total" in value:
            return _coerce_reference_float(value["total"])
        if "mean" in value:
            return _coerce_reference_float(value["mean"])
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _index_value(normalized: Mapping[str, Any], key: str) -> float | None:
    metric = normalized.get(key, {})
    if not isinstance(metric, Mapping):
        return None
    value = metric.get("value")
    if value is None:
        return None
    return float(value)


def _evaluate_model_on_trial_specs(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
) -> DetailedRolloutEvaluation:
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_array(leaf, n_replicates),
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, _infer_batch_size(trial_specs)),
        )

    if _has_replicate_specific_trial_inputs(trial_specs, n_replicates):
        keys = jr.split(jr.PRNGKey(seed), n_replicates)
        states_by_replicate = []
        for replicate in range(n_replicates):
            replicate_model = eqx.combine(
                _select_replicate_tree(model_arrays, replicate, n_replicates),
                model_other,
            )
            replicate_trial_specs = _select_replicate_trial_inputs(
                trial_specs,
                replicate,
                n_replicates,
            )
            states_by_replicate.append(
                task.eval_trials(
                    replicate_model,
                    replicate_trial_specs,
                    jr.split(keys[replicate], _infer_batch_size(replicate_trial_specs)),
                )
            )
        states = jt.map(lambda *xs: jnp.stack(xs, axis=0), *states_by_replicate)
    else:
        states = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
            model_arrays,
            jr.split(jr.PRNGKey(seed), n_replicates),
        )
    target_position = np.asarray(trial_specs.inputs["effector_target"].pos, dtype=np.float64)
    rollout = RolloutEvaluation(
        position=np.asarray(states.mechanics.effector.pos, dtype=np.float64),
        velocity=np.asarray(states.mechanics.effector.vel, dtype=np.float64),
        command=np.asarray(states.net.output, dtype=np.float64),
        hidden=np.asarray(states.net.hidden, dtype=np.float64),
        gru_input=np.asarray(states.net.input, dtype=np.float64),
        initial_position=np.asarray(_initial_effector_position(trial_specs), dtype=np.float64),
        initial_velocity=np.asarray(initial_effector_velocity(trial_specs), dtype=np.float64),
        target_position=target_position,
        dt=0.01,
    )
    mechanics_vector = np.asarray(states.mechanics.vector, dtype=np.float64)
    object.__setattr__(rollout, "mechanics_vector", mechanics_vector)
    return DetailedRolloutEvaluation(
        rollout=rollout,
        feedback=np.asarray(states.sensory.output, dtype=np.float64),
        mechanics_vector=mechanics_vector,
    )


def _not_available_rows(
    *,
    bin_id: str,
    perturbation_id: str | None,
    reason: str,
    adapter: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    return [
        {
            "bin": bin_id,
            "mode": mode,
            "perturbation_id": perturbation_id,
            "status": "not_available",
            "reason": reason,
            "ablation": build_observation_ablation_spec(mode, bin_id=bin_id).to_json(),
            "adapter": None if adapter is None else dict(adapter),
        }
        for mode in default_ablation_modes()
    ]


def _add_trial_input(trial_specs: Any, key: str, value: Any) -> Any:
    return eqx.tree_at(
        lambda ts: ts.inputs,
        trial_specs,
        {**trial_specs.inputs, key: value},
    )


def _has_replicate_specific_trial_inputs(trial_specs: Any, n_replicates: int) -> bool:
    return any(
        key.startswith(f"{OBSERVATION_ABLATION_INPUT_PREFIX}:")
        and getattr(value, "shape", ())[:1] == (n_replicates,)
        for key, value in trial_specs.inputs.items()
    )


def _select_replicate_trial_inputs(trial_specs: Any, replicate: int, n_replicates: int) -> Any:
    inputs = {}
    for key, value in trial_specs.inputs.items():
        if (
            key.startswith(f"{OBSERVATION_ABLATION_INPUT_PREFIX}:")
            and getattr(value, "shape", ())[:1] == (n_replicates,)
        ):
            inputs[key] = value[replicate]
        else:
            inputs[key] = value
    return eqx.tree_at(lambda ts: ts.inputs, trial_specs, inputs)


def _select_replicate_tree(tree: Any, replicate: int, n_replicates: int) -> Any:
    return jt.map(
        lambda leaf: leaf[replicate] if _is_replicate_array(leaf, n_replicates) else leaf,
        tree,
    )


def _infer_batch_size(trial_specs: Any) -> int:
    for value in trial_specs.inputs.values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
        pos = getattr(value, "pos", None)
        if pos is not None:
            return int(pos.shape[0])
    for value in trial_specs.inits.values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
        pos = getattr(value, "pos", None)
        if pos is not None:
            return int(pos.shape[0])
    raise ValueError("could not infer trial batch size")


def _initial_effector_position(trial_specs: Any) -> jnp.ndarray:
    for init_state in trial_specs.inits.values():
        position = getattr(init_state, "pos", None)
        if position is not None:
            return position
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 2:
            return jnp.asarray(init_state)[..., 0:2]
    raise ValueError("trial spec does not include an effector position initial state")


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates


def _endpoint_error(evaluation: RolloutEvaluation) -> np.ndarray:
    endpoint = evaluation.position[:, :, -1, :]
    target = evaluation.target_position[:, -1, :]
    return np.linalg.norm(endpoint - target[None, :, :], axis=-1)


def _terminal_speed(evaluation: RolloutEvaluation) -> np.ndarray:
    return np.linalg.norm(evaluation.velocity[:, :, -1, :], axis=-1)


def _summary_stats(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    flat = array.reshape(-1)
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p50": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
    }


def _metric_mean(row: Mapping[str, Any], metric: str) -> float | None:
    return _summary_mean(row.get("metrics", {}).get(metric))


def _summary_mean(summary: Any) -> float | None:
    if not isinstance(summary, Mapping):
        return None
    value = summary.get("mean")
    if value is None:
        return None
    return float(value)


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _format_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.6g}"


def _repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


__all__ = [
    "AblationMode",
    "DEFAULT_OUTPUT_FILENAME",
    "DEFAULT_RESULT_EXPERIMENT",
    "DEFAULT_RUN_IDS",
    "DEFAULT_SCOPE",
    "DEFAULT_SOURCE_EXPERIMENT",
    "ObservationAblationSpec",
    "SCHEMA_VERSION",
    "build_observation_ablation_spec",
    "build_observation_tape",
    "default_ablation_modes",
    "evaluate_run_feedback_ablation",
    "feedback_checkpoint_selection_audit",
    "insert_observation_ablation",
    "interpret_run_feedback_ablation",
    "materialize_gru_feedback_ablation",
    "render_feedback_ablation_markdown",
    "selected_feedback_ablation_bins",
    "summarize_ablation_delta",
    "summarize_normalized_feedback_use",
]
