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
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.plot import save_figure
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.diagnostic_provenance import repo_relative, write_regeneration_spec
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    CheckpointSelectionMode,
    load_validation_selected_checkpoint_model,
)
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import RolloutEvaluation
from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    apply_perturbation_to_trial_specs,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    RunFigureInputs,
    repeat_single_validation_trial,
    resolve_run_inputs,
)
from rlrmp.analysis.pipelines.sisu_spectrum_diagnostics import (
    set_sisu_condition,
    zero_disturbance_payload,
)
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.train.task_model import setup_task_model_pair


SCHEMA_VERSION = "rlrmp.gru_steady_state_perturbation_bank.v1"
ISSUE = "87424a4"
DEFAULT_PRE_GO_STEPS = 10
DEFAULT_POST_GO_WASHIN_STEPS = 50
DEFAULT_PULSE_DURATION_STEPS = 5
DEFAULT_FINAL_WINDOW_STEPS = 10
DEFAULT_N_ROLLOUT_TRIALS = 4
DEFAULT_POSITION_SCALE_M = 0.1
DEFAULT_VELOCITY_SCALE_M_S = 0.5
DEFAULT_FORCE_FILTER_SCALE = 1.0
DEFAULT_PRE_ONSET_FIGURE_STEPS = 10
DEFAULT_POST_ONSET_FIGURE_STEPS = 50
SUMMARY_MARKER = "steady_state_perturbation_bank"

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
        return {
            "perturbation_id": self.perturbation_id,
            "channel": "sensory_feedback",
            "family": f"steady_state_{self.family}_feedback_offset",
            "semantic_family": "steady_state_feedback_offset",
            "feedback_quantity": self.family,
            "feedback_payload_index": payload_index,
            "force_filter_feedback_only": self.family == "force_filter",
            "amplitude": float(self.amplitude),
            "units": self.units,
            "axis": axis,
            "basis": f"feedback_{self.family}_xy",
            "sign": int(self.sign),
            "timing": {
                "epoch": "steady_state_endpoint",
                "start_time_index": int(pulse_start),
                "duration_steps": int(pulse_duration),
            },
            "timing_bin": "steady_state_endpoint",
            "adapter": "named_graph_channel_offset",
            "description": (
                f"Add a {self.units} {self.family} feedback offset after the shared "
                "steady-state wash-in prefix."
            ),
            "channel_provenance": {
                "feedback_dim": int(feedback_dim),
                "feedback_quantity": self.family,
                "feedback_payload_index": int(payload_index),
                "direction": [float(self.direction[0]), float(self.direction[1])],
            },
        }


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
    position_scale_m: float = DEFAULT_POSITION_SCALE_M,
    velocity_scale_m_s: float = DEFAULT_VELOCITY_SCALE_M_S,
    force_filter_scale: float = DEFAULT_FORCE_FILTER_SCALE,
) -> tuple[FeedbackPerturbation, ...]:
    """Return symmetric position, velocity, and force/filter feedback offsets."""

    rows: list[FeedbackPerturbation] = []
    families: tuple[tuple[PerturbationFamily, tuple[int, int], float, str], ...] = (
        ("position", (0, 1), position_scale_m, "m"),
        ("velocity", (2, 3), velocity_scale_m_s, "m/s"),
        ("force_filter", (4, 5), force_filter_scale, "model_feedback_units"),
    )
    for family, indices, amplitude, units in families:
        if max(indices) >= feedback_dim:
            continue
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

    repo_root = repo_root.resolve()
    notes_dir = mkdir_p(repo_root / "results" / result_experiment / "notes")
    feedback_offset_scales = _feedback_offset_scales(
        position_scale_m=position_scale_m,
        velocity_scale_m_s=velocity_scale_m_s,
        force_filter_scale=force_filter_scale,
    )
    all_results: dict[str, Any] = {}
    for comparison in comparisons:
        all_results[comparison.comparison_id] = evaluate_comparison(
            comparison,
            result_experiment=result_experiment,
            n_rollout_trials=n_rollout_trials,
            pulse_duration_steps=pulse_duration_steps,
            position_scale_m=position_scale_m,
            velocity_scale_m_s=velocity_scale_m_s,
            force_filter_scale=force_filter_scale,
            repo_root=repo_root,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": result_experiment,
        "n_rollout_trials": int(n_rollout_trials),
        "pulse_duration_steps": int(pulse_duration_steps),
        "feedback_offset_scales": feedback_offset_scales,
        "response_window": _response_window_contract(),
        "washin_contract": _washin_contract(),
        "comparisons": all_results,
    }
    summary_path = notes_dir / "steady_state_perturbation_bank_summary.json"
    markdown_path = notes_dir / "steady_state_perturbation_bank.md"
    regeneration_path = notes_dir / "steady_state_perturbation_bank_regeneration_spec.json"
    summary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _update_summary_markdown(markdown_path, render_summary_markdown(manifest))
    write_regeneration_spec(
        spec_path=regeneration_path,
        diagnostic_name="gru_steady_state_perturbation_bank",
        materializer=(
            "rlrmp.analysis.pipelines.gru_steady_state_perturbation_bank."
            "materialize_steady_state_comparisons"
        ),
        command="PYTHONPATH=$PWD/src uv run --no-sync python results/87424a4/scripts/materialize_steady_state_perturbation_bank.py",
        parameters={
            "result_experiment": result_experiment,
            "n_rollout_trials": n_rollout_trials,
            "pulse_duration_steps": pulse_duration_steps,
            "position_scale_m": position_scale_m,
            "velocity_scale_m_s": velocity_scale_m_s,
            "force_filter_scale": force_filter_scale,
        },
        inputs=[
            {"role": "run_spec", "path": row["run_spec_path"]}
            for payload in all_results.values()
            for row in payload["conditions"].values()
        ],
        outputs=[
            {"role": "summary_json", "path": summary_path},
            {"role": "summary_markdown", "path": markdown_path},
        ],
        source_files=[
            "src/rlrmp/analysis/pipelines/gru_steady_state_perturbation_bank.py",
            "results/87424a4/scripts/materialize_steady_state_perturbation_bank.py",
        ],
        notes=[
            "Local model evaluation only; no retraining, pod, or reach-context bank rerun.",
            "Raw rollout arrays are not written; tracked summaries and figure specs are sufficient.",
        ],
        repo_root=repo_root,
    )
    manifest["outputs"] = {
        "summary_json": repo_relative(summary_path, repo_root=repo_root),
        "summary_markdown": repo_relative(markdown_path, repo_root=repo_root),
        "regeneration_spec": repo_relative(regeneration_path, repo_root=repo_root),
    }
    summary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def evaluate_comparison(
    comparison: ComparisonSpec,
    *,
    result_experiment: str,
    n_rollout_trials: int,
    pulse_duration_steps: int,
    position_scale_m: float,
    velocity_scale_m_s: float,
    force_filter_scale: float,
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate one comparison and write its three-panel figure."""

    condition_payloads: dict[str, Any] = {}
    timing_payloads: dict[str, Any] = {}
    feedback_dims: dict[str, int] = {}
    feedback_offset_scales = _feedback_offset_scales(
        position_scale_m=position_scale_m,
        velocity_scale_m_s=velocity_scale_m_s,
        force_filter_scale=force_filter_scale,
    )
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
        seed = int(run.run_spec.get("seed", 42))
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
        base_trials = repeat_single_validation_trial(pair.task.validation_trials, n_rollout_trials)
        steady_trials, timing = make_steady_state_trial_specs(
            base_trials,
            delayed=delayed,
            target_position=np.asarray(_target_position(run, base_trials), dtype=np.float64),
            pulse_duration_steps=pulse_duration_steps,
        )
        steady_trials = pad_feedback_offset_inputs(
            steady_trials,
            expected_feedback_dim=_expected_feedback_dim_from_hps(hps),
        )
        steady_trials = zero_disturbance_payload(steady_trials)
        feedback_dim = _feedback_dim(steady_trials)
        perturbations = default_feedback_perturbations(
            feedback_dim=feedback_dim,
            position_scale_m=position_scale_m,
            velocity_scale_m_s=velocity_scale_m_s,
            force_filter_scale=force_filter_scale,
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
            repo_root=repo_root,
        )
        timing_payloads[condition.condition_id] = timing
        feedback_dims[condition.condition_id] = int(feedback_dim)

    figure = build_response_figure(
        comparison_title=comparison.title,
        conditions=condition_payloads,
        dt=float(next(iter(condition_payloads.values()))["dt_s"]),
        pulse_duration_steps=pulse_duration_steps,
    )
    spec = {
        "schema_version": SCHEMA_VERSION,
        "issue": result_experiment,
        "comparison_id": comparison.comparison_id,
        "title": comparison.title,
        "source_experiment": comparison.source_experiment,
        "run_id": comparison.run_id,
        "n_rollout_trials": int(n_rollout_trials),
        "pulse_duration_steps": int(pulse_duration_steps),
        "feedback_offset_scales": feedback_offset_scales,
        "response_window": _response_window_contract(),
        "timing_by_condition": timing_payloads,
        "washin_contract": _washin_contract(),
    }
    _ensure_rlrmp_registered()
    saved = save_figure(
        fig=figure,
        spec=spec,
        package="rlrmp",
        experiment=result_experiment,
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
    pre_go_steps: int = DEFAULT_PRE_GO_STEPS,
    post_go_washin_steps: int = DEFAULT_POST_GO_WASHIN_STEPS,
    pulse_duration_steps: int = DEFAULT_PULSE_DURATION_STEPS,
) -> tuple[Any, dict[str, Any]]:
    """Return trial specs initialized at the target with a steady-state wash-in prefix."""

    updated = set_mechanics_vector_to_target(trial_specs, target_position)
    updated = set_target_streams_to_constant(updated, target_position)
    updated = zero_disturbance_payload(updated)
    horizon = _trial_horizon(updated)
    if delayed:
        effective_pre_go = min(pre_go_steps, max(horizon - 1, 0))
        updated = set_delayed_go_cue(updated, pre_go_steps=effective_pre_go)
        requested_pulse_start = min(
            effective_pre_go + post_go_washin_steps,
            max(horizon - 1, 0),
        )
        washin_policy = "10_pre_go_then_post_go_hold_prefix"
    else:
        effective_pre_go = 0
        requested_pulse_start = min(post_go_washin_steps, max(horizon - 1, 0))
        washin_policy = "immediate_hold_prefix"
    pulse_start = _figure_compatible_pulse_start(
        requested_pulse_start,
        horizon_steps=horizon,
    )
    response_steps = max(horizon - pulse_start, 0)
    return updated, {
        "horizon_steps": int(horizon),
        "pre_go_steps": int(effective_pre_go),
        "post_go_washin_steps_requested": int(post_go_washin_steps),
        "post_go_washin_steps_actual": int(max(pulse_start - effective_pre_go, 0)),
        "pulse_start_step_requested": int(requested_pulse_start),
        "pulse_start_step": int(pulse_start),
        "pulse_duration_steps": int(min(max(pulse_duration_steps, 1), max(response_steps, 1))),
        "response_steps": int(response_steps),
        "post_onset_steps_available": int(response_steps),
        "washin_policy": washin_policy,
        "fanout_policy": (
            "prefix_equivalent_batched_trials; Feedbax task API does not expose a stable "
            "hidden-state resume hook, so every perturbation row shares the same deterministic "
            "wash-in prefix and zero-noise inputs inside the same materialization pass."
        ),
    }


def _figure_compatible_pulse_start(requested_pulse_start: int, *, horizon_steps: int) -> int:
    """Keep the requested wash-in unless it would starve the recovery figure window."""

    if horizon_steps >= DEFAULT_PRE_ONSET_FIGURE_STEPS + DEFAULT_POST_ONSET_FIGURE_STEPS:
        latest = max(horizon_steps - DEFAULT_POST_ONSET_FIGURE_STEPS, 0)
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
    wash = washin_diagnostics(base, pulse_start=int(timing["pulse_start_step"]))
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
) -> dict[str, Any]:
    """Summarize a signed feedback perturbation row."""

    delta_command = perturbed.command - base.command
    delta_hidden = perturbed.hidden - base.hidden
    direction = np.asarray(perturbation.direction, dtype=np.float64)
    signed_direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    aligned_command = np.tensordot(delta_command, signed_direction, axes=([-1], [0]))
    aligned_position = np.tensordot(
        perturbed.position - base.position,
        signed_direction,
        axes=([-1], [0]),
    )
    aligned_velocity = np.tensordot(
        perturbed.velocity - base.velocity,
        signed_direction,
        axes=([-1], [0]),
    )
    response = aligned_command[:, :, pulse_start:]
    command_window, relative_steps = _mean_onset_window(
        aligned_command,
        pulse_start=pulse_start,
    )
    position_window, _ = _mean_onset_window(
        aligned_position,
        pulse_start=pulse_start,
    )
    velocity_window, _ = _mean_onset_window(
        aligned_velocity,
        pulse_start=pulse_start,
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
        "sign": int(perturbation.sign),
        "amplitude": float(perturbation.amplitude),
        "units": perturbation.units,
        "aligned_output_profile": [float(value) for value in mean_profile],
        "relative_time_steps": [int(value) for value in relative_steps],
        "aligned_output_window_profile": [float(value) for value in command_window],
        "aligned_position_window_profile": [float(value) for value in position_window],
        "aligned_velocity_window_profile": [float(value) for value in velocity_window],
        "metrics": {
            "peak_output_response": float(peak_abs(response)),
            "output_auc_impulse": float(np.sum(np.abs(response)) * float(base.dt) / response.size),
            "terminal_residual": float(np.mean(np.abs(terminal))) if terminal.size else 0.0,
            "recovery_settling_step": settling,
            "direction_variability": float(np.std(np.mean(response, axis=-1)))
            if response.size
            else 0.0,
            "hidden_delta_peak": float(peak_abs(hidden_norm)),
            "output_norm_peak": float(peak_abs(action_norm)),
            "peak_position_m": float(peak_abs(position_window)),
            "peak_velocity_m_s": float(peak_abs(velocity_window)),
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
        profiles = np.asarray(
            [row["aligned_output_profile"] for row in family_rows], dtype=np.float64
        )
        mean_profile = np.mean(profiles, axis=0)
        pair_scores = signed_pair_antisymmetry(family_rows)
        metrics = [row["metrics"] for row in family_rows if isinstance(row.get("metrics"), Mapping)]
        summary[family] = {
            "n_rows": int(len(family_rows)),
            "aligned_output_profile_mean": [float(value) for value in mean_profile],
            "aligned_output_profile_sem": [
                float(value)
                for value in (
                    np.std(profiles, axis=0, ddof=1) / np.sqrt(profiles.shape[0])
                    if profiles.shape[0] > 1
                    else np.zeros_like(mean_profile)
                )
            ],
            "peak_output_response": float(
                np.mean([metric["peak_output_response"] for metric in metrics])
            ),
            "output_auc_impulse": float(
                np.mean([metric["output_auc_impulse"] for metric in metrics])
            ),
            "terminal_residual": float(
                np.mean([metric["terminal_residual"] for metric in metrics])
            ),
            "direction_variability": float(
                np.std([metric["peak_output_response"] for metric in metrics])
            ),
            "signed_pair_antisymmetry": pair_scores,
        }
        summary[family].update(_aggregate_window_profiles(family_rows))
    return summary


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
        "aligned_position_window_profile",
        "aligned_velocity_window_profile",
    )
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
    final_window_steps: int = DEFAULT_FINAL_WINDOW_STEPS,
) -> dict[str, Any]:
    """Summarize baseline drift over the final wash-in window."""

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
        ("aligned_output_window_profile", "Output", None),
        ("aligned_position_window_profile", "Position (m)", "m"),
        ("aligned_velocity_window_profile", "Velocity (m/s)", "m/s"),
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
        for row_index, (profile_key, _, _) in enumerate(row_specs, start=1):
            for idx, condition in enumerate(conditions.values()):
                family_summary = condition.get("family_summary", {}).get(family)
                if not family_summary:
                    continue
                y = np.asarray(family_summary[f"{profile_key}_mean"], dtype=float)
                sem = np.asarray(family_summary[f"{profile_key}_sem"], dtype=float)
                x = np.asarray(family_summary["relative_time_steps"], dtype=float) * dt
                color = _condition_color(idx)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line={"color": color, "width": 2.1},
                        name=str(condition["label"]),
                        legendgroup=str(condition["label"]),
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
                        legendgroup=str(condition["label"]),
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
    for row_index, (_, title, _) in enumerate(row_specs, start=1):
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
        "- Delayed rows use a 10-step pre-go prefix followed by post-go hold wash-in.",
        "- Undelayed rows use an immediate hold prefix on the validated trial horizon.",
        (
            "- Default pulse shape: "
            f"{manifest['pulse_duration_steps']} steps; "
            f"position={manifest['feedback_offset_scales']['position_m']} m, "
            f"velocity={manifest['feedback_offset_scales']['velocity_m_s']} m/s, "
            f"force/filter={manifest['feedback_offset_scales']['force_filter']}."
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
    from rlrmp.analysis.pipelines.gru_perturbation_bank import (
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


def _mean_onset_window(
    aligned_values: np.ndarray,
    *,
    pulse_start: int,
    pre_steps: int = DEFAULT_PRE_ONSET_FIGURE_STEPS,
    post_steps: int = DEFAULT_POST_ONSET_FIGURE_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return trial/replicate mean in a small pre-onset and recovery window."""

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


def _washin_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "initial_mechanics": (
            "mechanics.vector is zeroed, then every 8D current/delayed mechanics "
            "block receives target x/y position with zero velocity, force/filter, "
            "and integrator state."
        ),
        "noise": "epsilon inputs are zeroed before evaluation.",
        "delayed_go_cue": "go cue off for 10 steps, then on; target visible throughout.",
        "fanout_policy": (
            "prefix_equivalent_batched_trials because the current Feedbax eval API "
            "does not expose a supported hidden-state resume hook."
        ),
    }


def _response_window_contract() -> dict[str, Any]:
    return {
        "pre_onset_steps": DEFAULT_PRE_ONSET_FIGURE_STEPS,
        "post_onset_steps": DEFAULT_POST_ONSET_FIGURE_STEPS,
        "x_axis": "seconds relative to perturbation onset",
        "rows": [
            "network output",
            "point-mass position along aligned perturbation direction",
            "point-mass velocity along aligned perturbation direction",
        ],
    }


def _feedback_offset_scales(
    *,
    position_scale_m: float,
    velocity_scale_m_s: float,
    force_filter_scale: float,
) -> dict[str, float]:
    return {
        "position_m": float(position_scale_m),
        "velocity_m_s": float(velocity_scale_m_s),
        "force_filter": float(force_filter_scale),
    }


def _feedback_offset_scale_defaults() -> dict[str, float]:
    return _feedback_offset_scales(
        position_scale_m=DEFAULT_POSITION_SCALE_M,
        velocity_scale_m_s=DEFAULT_VELOCITY_SCALE_M_S,
        force_filter_scale=DEFAULT_FORCE_FILTER_SCALE,
    )


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
    "aggregate_family_profiles",
    "default_feedback_perturbations",
    "identity_condition",
    "make_steady_state_trial_specs",
    "materialize_steady_state_comparisons",
    "pad_feedback_offset_inputs",
    "sisu_condition",
    "signed_pair_antisymmetry",
    "washin_diagnostics",
]
