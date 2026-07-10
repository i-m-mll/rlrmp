"""Materialize delayed_movement_bank fixed-bank velocity profiles.

This issue-local script evaluates the final delayed_movement_bank checkpoint on
the corrected delayed-reach no-catch and catch banks: go cues 10..30, 20 uniform
center-out directions, 0.15 m reach length, and go-cue-aligned velocity windows.
"""

from __future__ import annotations
from rlrmp.eval.kinematics import initial_effector_position, initial_effector_velocity
from rlrmp.viz.colors import hex_to_rgba
from rlrmp.viz.traces import add_band_trace as canonical_add_band_trace, add_reference_trace as canonical_add_reference_trace

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax import DelayedReachTaskInputs, TaskTrialSpec, TrialTimeline, WhereDict
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.objectives.loss import TargetSpec
from feedbax.runtime.state import CartesianState
from jax_cookbook import load_with_hyperparameters

from rlrmp.analysis.math.trial_alignment import align_trials, trim_to_full_support
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    DEFAULT_DELAYED_REACH_DIRECTION_COUNT,
    DEFAULT_DELAYED_REACH_GO_CUE_STEPS,
    DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M,
    DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import cs_output_feedback_reference_profiles
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.viz import profile_comparison_grid


EXPERIMENT = "6c36536"
RUN_ID = "delayed_movement_bank"
LABEL = "delayed_movement_bank"
TOPIC = "delayed_movement_bank_velocity_profiles"
PRE_GO_CONTEXT_STEPS = 10
REFERENCE_COLOR = "#111827"
BankKind = Literal["no_catch", "catch"]


@dataclass(frozen=True)
class RunInputs:
    """Resolved local files for one run."""

    run_id: str
    label: str
    run_spec_path: Path
    artifact_dir: Path
    run_spec: dict[str, Any]


@dataclass(frozen=True)
class DelayedEvalBank:
    """A concrete delayed evaluation bank plus JSON metadata."""

    trial_specs: TaskTrialSpec
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VelocityProfile:
    """Go-cue-aligned target-radial velocity profile."""

    run_id: str
    label: str
    bank_kind: BankKind
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    replicate_mean: np.ndarray
    replicate_std: np.ndarray
    n_replicates: int
    n_trials_per_replicate: int
    alignment: dict[str, Any]
    evaluation_bank: dict[str, Any]

    @property
    def n_pooled_samples(self) -> int:
        """Return replicate x trial sample count."""

        return int(self.n_replicates * self.n_trials_per_replicate)


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    run = resolve_run_inputs(
        experiment=args.experiment,
        run_id=args.run_id,
        label=args.label,
        repo_root=args.repo_root,
    )
    output_root = args.output_dir or (
        args.repo_root / "_artifacts" / args.experiment / "figures" / args.topic
    )
    summaries: dict[str, Any] = {}
    for bank_kind in ("no_catch", "catch"):
        output_dir = output_root / bank_kind
        mkdir_p(output_dir)
        profile = evaluate_velocity_profile(
            run,
            bank_kind=bank_kind,
            go_cue_steps=tuple(args.go_cue_step),
            direction_count=args.direction_count,
            reach_length_m=args.reach_length_m,
            pre_go_context_steps=args.pre_go_context_steps,
        )
        include_reference = bool(args.include_reference and bank_kind == "no_catch")
        references = (
            cs_output_feedback_reference_profiles(n_samples=profile.n_pooled_samples)
            if include_reference
            else ()
        )
        pooled_file = write_velocity_figure(
            profile,
            output_dir=output_dir,
            references=references,
        )
        replicate_file = write_velocity_by_replicate_figure(
            profile,
            output_dir=output_dir,
            references=references,
        )
        summary = build_summary(
            run=run,
            profile=profile,
            pooled_file=pooled_file,
            replicate_file=replicate_file,
            references=references,
            output_dir=output_dir,
            direction_split_status=(
                "not_materialized: prior good/bad direction split was a "
                "diagnostic grouping for earlier no-PGD rows, not a natural "
                "movement-bank row grouping; by-replicate profiles were "
                "materialized as the cheap directly analogous variant"
            ),
        )
        summary_path = output_dir / "velocity_profile_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries[bank_kind] = summary
    print(json.dumps(summaries, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--label", default=LABEL)
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--go-cue-step",
        type=int,
        action="append",
        default=list(DEFAULT_DELAYED_REACH_GO_CUE_STEPS),
    )
    parser.add_argument(
        "--direction-count",
        type=int,
        default=DEFAULT_DELAYED_REACH_DIRECTION_COUNT,
    )
    parser.add_argument(
        "--reach-length-m",
        type=float,
        default=DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M,
    )
    parser.add_argument("--pre-go-context-steps", type=int, default=PRE_GO_CONTEXT_STEPS)
    parser.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay extLQG references on the no-catch profile.",
    )
    return parser


def resolve_run_inputs(
    *,
    experiment: str,
    run_id: str,
    label: str,
    repo_root: Path,
) -> RunInputs:
    """Resolve the flat or legacy directory-form run spec for one run."""

    flat_path = repo_root / "results" / experiment / "runs" / f"{run_id}.json"
    legacy_path = repo_root / "results" / experiment / "runs" / run_id / "run.json"
    if flat_path.exists():
        run_spec_path = flat_path
    elif legacy_path.exists():
        run_spec_path = legacy_path
    else:
        raise FileNotFoundError(f"Missing run spec for {experiment}/{run_id}")
    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Missing artifact directory: {artifact_dir}")
    return RunInputs(
        run_id=run_id,
        label=label,
        run_spec_path=run_spec_path,
        artifact_dir=artifact_dir,
        run_spec=json.loads(run_spec_path.read_text(encoding="utf-8")),
    )


def evaluate_velocity_profile(
    run: RunInputs,
    *,
    bank_kind: BankKind,
    go_cue_steps: Sequence[int],
    direction_count: int,
    reach_length_m: float,
    pre_go_context_steps: int,
) -> VelocityProfile:
    """Evaluate the final checkpoint on one fixed delayed-reach bank."""

    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run.run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    movement_horizon_steps = canonical_movement_horizon(run.run_spec)
    bank = make_delayed_eval_bank(
        pair.task.validation_trials,
        bank_kind=bank_kind,
        go_cue_steps=go_cue_steps,
        direction_count=direction_count,
        reach_length_m=reach_length_m,
        movement_horizon_steps=movement_horizon_steps,
    )
    model, _hyperparameters = load_with_hyperparameters(
        run.artifact_dir / "trained_model.eqx",
        setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
    )
    eval_trial_count = int(bank.metadata["trial_count"])
    initial_velocity = initial_effector_velocity(bank.trial_specs)
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates,
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        states = pair.task.eval_trials(
            replicate_model,
            bank.trial_specs,
            jr.split(key, eval_trial_count),
        )
        return jnp.concatenate(
            [initial_velocity[:, None, :], states.mechanics.effector.vel],
            axis=1,
        )

    velocity = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(0), n_replicates),
    )
    velocity_np = np.asarray(velocity, dtype=np.float64)
    target_position = target_position_sequence(bank.trial_specs)
    initial_position = initial_effector_position(bank.trial_specs)
    direction, _distance = reach_direction(initial_position, target_position[:, -1, :])
    forward = np.sum(velocity_np * direction[None, :, None, :], axis=-1)
    go_index = np.asarray(bank.trial_specs.timeline.epoch_bounds[:, 1], dtype=np.int64)
    aligned_forward, center = align_trials(forward, go_index)
    _support, support_slice = trim_to_full_support(aligned_forward)
    support_start = int(support_slice.start or 0)
    support_stop = int(support_slice.stop or aligned_forward.shape[-1])
    requested_start = max(0, int(center) - int(pre_go_context_steps))
    requested_stop = min(
        int(aligned_forward.shape[-1]),
        int(center) + int(movement_horizon_steps),
    )
    start = max(support_start, requested_start)
    stop = min(support_stop, requested_stop)
    if stop <= start:
        raise ValueError(f"{bank_kind} aligned profile has no full-support samples")
    window = aligned_forward[..., start:stop]
    flat = window.reshape(-1, window.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0, ddof=1)
    replicate_mean = np.nanmean(window, axis=1)
    replicate_std = np.nanstd(window, axis=1, ddof=1)
    dt = float(run.run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
    time_s = (np.arange(start, stop, dtype=np.float64) - float(center)) * dt
    alignment = {
        "time_basis": "go_cue_aligned_canonical_movement_window",
        "center_index": int(center),
        "go_index_min": int(go_index.min()),
        "go_index_max": int(go_index.max()),
        "movement_horizon_steps": int(movement_horizon_steps),
        "requested_pre_go_context_steps": int(pre_go_context_steps),
        "requested_post_go_movement_steps": int(movement_horizon_steps),
        "full_support_slice": [support_start, support_stop],
        "requested_window_slice": [requested_start, requested_stop],
        "plotted_window_slice": [start, stop],
        "plotted_time_start_s": float(time_s[0]),
        "plotted_time_stop_s": float(time_s[-1]),
    }
    return VelocityProfile(
        run_id=run.run_id,
        label=run.label,
        bank_kind=bank_kind,
        time_s=time_s,
        mean=mean,
        std=std,
        replicate_mean=replicate_mean,
        replicate_std=replicate_std,
        n_replicates=n_replicates,
        n_trials_per_replicate=eval_trial_count,
        alignment=alignment,
        evaluation_bank=bank.metadata,
    )


def make_delayed_eval_bank(
    trial_specs: TaskTrialSpec,
    *,
    bank_kind: BankKind,
    go_cue_steps: Iterable[int],
    direction_count: int,
    reach_length_m: float,
    movement_horizon_steps: int,
) -> DelayedEvalBank:
    """Return a uniform-grid fixed delayed-reach evaluation bank."""

    go_steps = tuple(int(step) for step in go_cue_steps)
    if not go_steps:
        raise ValueError("go_cue_steps must not be empty")
    if min(go_steps) < 0:
        raise ValueError(f"go cue steps must be nonnegative; got {go_steps}")
    source_trials = infer_trial_count(trial_specs)
    n_state_steps = int(
        getattr(getattr(trial_specs, "timeline", None), "n_steps", None)
        or infer_trial_n_time(trial_specs)
    )
    if max(go_steps) + int(movement_horizon_steps) > n_state_steps:
        raise ValueError(
            "delayed eval bank does not fit trial axis: "
            f"go_max={max(go_steps)}, horizon={movement_horizon_steps}, "
            f"n_state_steps={n_state_steps}"
        )
    init_pos = np.asarray(initial_effector_position(trial_specs)[0], dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, int(direction_count), endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    targets = init_pos[None, :] + float(reach_length_m) * directions
    go_index = np.repeat(np.asarray(go_steps, dtype=np.int32), int(direction_count))
    direction_index = np.tile(np.arange(int(direction_count), dtype=np.int32), len(go_steps))
    source_index = np.asarray(direction_index % int(source_trials), dtype=np.int32)
    target_position = targets[direction_index]
    initial_position_batch = np.broadcast_to(init_pos[None, :], target_position.shape)

    state_time = np.arange(n_state_steps, dtype=np.int32)[None, :]
    control_steps = max(1, n_state_steps - 1)
    control_time = np.arange(control_steps, dtype=np.int32)[None, :]
    go_column = go_index[:, None]
    visible_target = np.broadcast_to(
        target_position[:, None, :],
        (go_index.shape[0], n_state_steps, 2),
    )
    init_sequence = np.broadcast_to(
        initial_position_batch[:, None, :],
        (go_index.shape[0], n_state_steps, 2),
    )
    no_catch_target = np.where(state_time[..., None] >= go_column[..., None], visible_target, init_sequence)
    scored_target = init_sequence if bank_kind == "catch" else no_catch_target
    control_hold = (
        np.ones((go_index.shape[0], control_steps), dtype=np.float32)
        if bank_kind == "catch"
        else (control_time < go_column).astype(np.float32)
    )
    target_on = np.ones_like(control_hold, dtype=np.float32)
    go_input = 1.0 - control_hold
    timeline = TrialTimeline.from_epochs_events(
        n_state_steps,
        epoch_bounds=np.stack(
            [
                np.zeros_like(go_index),
                go_index,
                np.full_like(go_index, n_state_steps),
            ],
            axis=-1,
        ),
        epoch_names=("prep", "movement"),
        event_steps=go_index[:, None],
        event_names=("go_cue",),
    )

    remapped = remap_source_trials(
        trial_specs,
        source_index=source_index,
        source_trials=source_trials,
    )
    direction_angles = np.mod(angles, 2.0 * np.pi).astype(np.float64)
    metadata = {
        "schema_version": DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION,
        "bank_family": "delayed_reach_fixed_eval_bank",
        "kind": bank_kind,
        "catch": bank_kind == "catch",
        "go_cue_min": int(min(go_steps)),
        "go_cue_max": int(max(go_steps)),
        "go_cue_steps": [int(step) for step in go_steps],
        "direction_source": "uniform_grid",
        "direction_count": int(direction_count),
        "requested_direction_count": int(direction_count),
        "trial_count": int(go_index.shape[0]),
        "n_time_steps": int(n_state_steps),
        "control_time_steps": int(control_steps),
        "movement_horizon_steps": int(movement_horizon_steps),
        "reach_length_m": float(reach_length_m),
        "reach_length_source": "uniform_grid_default",
        "reach_length_m_explicit": False,
        "direction_source_inferred_from_validation_targets": False,
        "duplicate_direction_count": 0,
        "target_radii_m": [float(reach_length_m)] * int(direction_count),
        "target_angles_rad": [float(angle) for angle in direction_angles],
        "source_trial_indices": [int(index) for index in source_index[: int(direction_count)]],
    }
    updated = TaskTrialSpec(
        inits=WhereDict(update_initial_positions(remapped.inits, initial_position_batch)),
        inputs=update_inputs(
            remapped.inputs,
            visible_target=visible_target,
            scored_target=scored_target,
            hold=control_hold,
            target_on=target_on,
            go_input=go_input,
        ),
        targets=WhereDict(update_targets(remapped.targets, scored_target)),
        intervene=remapped.intervene,
        timeline=timeline,
        extra={
            **dict(remapped.extra or {}),
            "delayed_reach_eval_bank": metadata,
            "is_catch_trial": np.full(go_index.shape[0], bank_kind == "catch"),
        },
    )
    return DelayedEvalBank(trial_specs=updated, metadata=metadata)


def remap_source_trials(
    trial_specs: TaskTrialSpec,
    *,
    source_index: np.ndarray,
    source_trials: int,
) -> TaskTrialSpec:
    """Copy source trial leaves onto the fixed-bank trial axis."""

    indices = jnp.asarray(source_index, dtype=jnp.int32)

    def remap_leaf(leaf: Any) -> Any:
        shape = getattr(leaf, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) == int(source_trials):
            return jnp.asarray(leaf)[indices]
        return leaf

    return jt.map(remap_leaf, trial_specs)


def update_initial_positions(inits: Mapping[Any, Any], position: np.ndarray) -> dict[Any, Any]:
    """Return init mapping with effector position replaced."""

    updated = dict(inits)
    for key, init_state in list(updated.items()):
        current = getattr(init_state, "pos", None)
        if current is not None:
            updated[key] = eqx.tree_at(
                lambda state: state.pos,
                init_state,
                jnp.asarray(position, dtype=jnp.asarray(current).dtype),
            )
            continue
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 2:
            value = jnp.asarray(init_state)
            updated[key] = value.at[..., 0:2].set(jnp.asarray(position, dtype=value.dtype))
    return updated


def update_targets(targets: Mapping[Any, Any], scored_target: np.ndarray) -> dict[Any, Any]:
    """Return target mapping with scored effector position replaced."""

    updated = dict(targets)
    for key, target_spec in list(updated.items()):
        value = getattr(target_spec, "value", None)
        if getattr(value, "shape", None) is not None and value.shape[-1] == 2:
            updated[key] = eqx.tree_at(
                lambda spec: spec.value,
                target_spec,
                jnp.asarray(scored_target, dtype=jnp.asarray(value).dtype),
            )
        elif isinstance(target_spec, TargetSpec):
            updated[key] = eqx.tree_at(
                lambda spec: spec.value,
                target_spec,
                jnp.asarray(scored_target),
            )
    return updated


def update_inputs(
    inputs: Any,
    *,
    visible_target: np.ndarray,
    scored_target: np.ndarray,
    hold: np.ndarray,
    target_on: np.ndarray,
    go_input: np.ndarray,
) -> Any:
    """Return input tree with target/go-cue surfaces replaced."""

    if isinstance(inputs, DelayedReachTaskInputs):
        return update_delayed_task_inputs(
            inputs,
            visible_target=visible_target,
            hold=hold,
            target_on=target_on,
        )
    if not isinstance(inputs, Mapping):
        return inputs
    updated = dict(inputs)
    if "input" in updated:
        updated["input"] = like_existing_time_input(updated["input"], go_input)
    if "target" in updated:
        updated["target"] = like_existing_array(updated["target"], visible_target)
    if "effector_target" in updated:
        updated["effector_target"] = update_cartesian_position(
            updated["effector_target"],
            scored_target,
        )
    if "task" in updated and isinstance(updated["task"], DelayedReachTaskInputs):
        updated["task"] = update_delayed_task_inputs(
            updated["task"],
            visible_target=visible_target,
            hold=hold,
            target_on=target_on,
        )
    return updated


def update_delayed_task_inputs(
    inputs: DelayedReachTaskInputs,
    *,
    visible_target: np.ndarray,
    hold: np.ndarray,
    target_on: np.ndarray,
) -> DelayedReachTaskInputs:
    """Return delayed task inputs with visible target and go cue replaced."""

    effector_target = update_cartesian_position(inputs.effector_target, visible_target)
    if getattr(effector_target, "pos", None) is None:
        effector_target = CartesianState(pos=jnp.asarray(visible_target))
    return DelayedReachTaskInputs(
        effector_target=effector_target,
        hold=like_existing_time_input(inputs.hold, hold),
        target_on=like_existing_time_input(inputs.target_on, target_on),
    )


def update_cartesian_position(value: Any, position: np.ndarray) -> Any:
    """Return a Cartesian-like state with replaced position when possible."""

    current = getattr(value, "pos", None)
    if current is None:
        return value
    return eqx.tree_at(
        lambda state: state.pos,
        value,
        jnp.asarray(position, dtype=jnp.asarray(current).dtype),
    )


def like_existing_time_input(existing: Any, value: np.ndarray) -> jnp.ndarray:
    """Cast and shape a scalar time input like an existing control input."""

    array = jnp.asarray(existing)
    replacement = jnp.asarray(value, dtype=array.dtype)
    if array.ndim >= 3 and array.shape[-1] == 1 and replacement.ndim == array.ndim - 1:
        replacement = replacement[..., None]
    return replacement


def like_existing_array(existing: Any, value: np.ndarray) -> jnp.ndarray:
    """Cast a replacement array like an existing array."""

    return jnp.asarray(value, dtype=jnp.asarray(existing).dtype)






def target_position_sequence(trial_specs: Any) -> np.ndarray:
    """Return the scored target position sequence, shape ``(trials, time, 2)``."""

    for target_spec in trial_specs.targets.values():
        value = getattr(target_spec, "value", None)
        if getattr(value, "shape", None) is not None and value.shape[-1] == 2:
            return np.asarray(value, dtype=np.float64)
    raise ValueError("Trial spec does not include 2D target sequence")


def reach_direction(
    initial_position: np.ndarray,
    target_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-radial unit vectors and distances."""

    delta = target_position - initial_position
    distance = np.linalg.norm(delta, axis=-1)
    safe_distance = np.where(distance > 0.0, distance, 1.0)
    return delta / safe_distance[:, None], distance


def infer_trial_count(trial_specs: Any) -> int:
    """Infer trial count from common TaskTrialSpec leaves."""

    for target_spec in getattr(trial_specs, "targets", {}).values():
        value = getattr(target_spec, "value", None)
        if getattr(value, "shape", None) is not None and len(value.shape) >= 1:
            return int(value.shape[0])
    initial = initial_effector_position(trial_specs)
    return int(initial.shape[0])


def infer_trial_n_time(trial_specs: Any) -> int:
    """Infer state-time length from targets or task inputs."""

    try:
        return int(target_position_sequence(trial_specs).shape[1])
    except ValueError:
        pass
    inputs = getattr(trial_specs, "inputs", None)
    if isinstance(inputs, Mapping) and "target" in inputs:
        target = jnp.asarray(inputs["target"])
        if target.ndim >= 2:
            return int(target.shape[1])
    raise ValueError("Could not infer trial time axis")


def canonical_movement_horizon(run_spec: Mapping[str, Any]) -> int:
    """Return the canonical delayed movement horizon from run metadata."""

    movement = run_spec.get("delayed_reach", {}).get("movement_epoch", {})
    if "cs_schedule_horizon_steps" in movement:
        return int(movement["cs_schedule_horizon_steps"])
    projection = run_spec.get("game_card", {}).get("delayed_reach_projection", {})
    if "canonical_cs_movement_horizon_steps" in projection:
        return int(projection["canonical_cs_movement_horizon_steps"])
    if "horizon_steps" in run_spec.get("game_card", {}):
        return int(run_spec["game_card"]["horizon_steps"])
    return 60


def write_velocity_figure(
    profile: VelocityProfile,
    *,
    output_dir: Path,
    references: Sequence[Any],
) -> Path:
    """Write the pooled fixed-bank velocity profile."""

    fig = profile_comparison_grid(
        n_panels=1,
        subplot_titles=[f"{profile.label} ({profile.bank_kind})"],
        vertical_spacing=0.04,
    )
    add_band_trace(
        fig,
        x=profile.time_s,
        mean=profile.mean,
        std=profile.std,
        row=1,
        color="#2563eb",
        name=profile.label,
        legendgroup="gru",
        showlegend=True,
    )
    for reference in references:
        add_reference_trace(fig, reference=reference, row=1)
    fig.add_vline(x=0.0, line={"color": "black", "dash": "dash", "width": 1}, row=1, col=1)
    fig.update_layout(
        title=f"Delayed movement-bank target-radial velocity ({profile.bank_kind})",
        width=900,
        height=520,
        margin={"l": 72, "r": 24, "t": 76, "b": 72},
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=1, col=1)
    fig.update_yaxes(title_text="Target-radial velocity (m/s)", zeroline=True)
    path = output_dir / "forward_velocity_profiles_stochastic.html"
    fig.write_html(path)
    return path


def write_velocity_by_replicate_figure(
    profile: VelocityProfile,
    *,
    output_dir: Path,
    references: Sequence[Any],
) -> Path:
    """Write replicate-resolved fixed-bank velocity profiles."""

    fig = profile_comparison_grid(
        n_panels=1,
        subplot_titles=[f"{profile.label} by replicate ({profile.bank_kind})"],
        vertical_spacing=0.04,
    )
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#be123c")
    for rep_idx in range(profile.n_replicates):
        add_band_trace(
            fig,
            x=profile.time_s,
            mean=profile.replicate_mean[rep_idx],
            std=profile.replicate_std[rep_idx],
            row=1,
            color=colors[rep_idx % len(colors)],
            name=f"replicate {rep_idx}",
            legendgroup=f"replicate-{rep_idx}",
            showlegend=True,
            fill_alpha=0.10,
            line_width=1.8,
        )
    for reference in references:
        add_reference_trace(fig, reference=reference, row=1)
    fig.add_vline(x=0.0, line={"color": "black", "dash": "dash", "width": 1}, row=1, col=1)
    fig.update_layout(
        title=f"Delayed movement-bank target-radial velocity by replicate ({profile.bank_kind})",
        width=940,
        height=560,
        margin={"l": 72, "r": 24, "t": 76, "b": 76},
        hovermode="x unified",
        legend={"groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=1, col=1)
    fig.update_yaxes(title_text="Target-radial velocity (m/s)", zeroline=True)
    path = output_dir / "forward_velocity_profiles_by_replicate_stochastic.html"
    fig.write_html(path)
    return path


add_band_trace = canonical_add_band_trace


add_reference_trace = canonical_add_reference_trace


def build_summary(
    *,
    run: RunInputs,
    profile: VelocityProfile,
    pooled_file: Path,
    replicate_file: Path,
    references: Sequence[Any],
    output_dir: Path,
    direction_split_status: str,
) -> dict[str, Any]:
    """Return JSON-compatible sidecar metadata."""

    peak_idx = int(np.nanargmax(profile.mean))
    return {
        "schema_version": "rlrmp.delayed_movement_bank_velocity_profiles.v1",
        "issue": EXPERIMENT,
        "run_id": run.run_id,
        "run_label": run.label,
        "run_spec": repo_relative(run.run_spec_path),
        "artifact_dir": repo_relative(run.artifact_dir),
        "output_dir": repo_relative(output_dir),
        "bank_kind": profile.bank_kind,
        "checkpoint_policy": "final_checkpoint",
        "checkpoint_source": repo_relative(run.artifact_dir / "trained_model.eqx"),
        "figure": pooled_file.name,
        "replicate_figure": replicate_file.name,
        "projection": "target-radial velocity: dot(effector velocity, unit(target - initial_position))",
        "error_band": (
            "mean +/- 1 SD over pooled replicate x fixed-bank go-cue/direction trials"
        ),
        "direction_split": {"status": direction_split_status},
        "alignment": profile.alignment,
        "evaluation_bank": profile.evaluation_bank,
        "profile": {
            "n_replicates": int(profile.n_replicates),
            "n_trials_per_replicate": int(profile.n_trials_per_replicate),
            "n_pooled_samples": int(profile.n_pooled_samples),
            "n_time_steps": int(profile.mean.shape[0]),
            "peak_mean_forward_velocity_m_s": float(profile.mean[peak_idx]),
            "time_of_peak_mean_forward_velocity_s": float(profile.time_s[peak_idx]),
            "mean_forward_velocity_min_m_s": float(np.nanmin(profile.mean)),
            "mean_forward_velocity_max_m_s": float(np.nanmax(profile.mean)),
            "time_start_s": float(profile.time_s[0]),
            "time_stop_s": float(profile.time_s[-1]),
            "finite": bool(
                np.isfinite(profile.mean).all()
                and np.isfinite(profile.std).all()
                and np.isfinite(profile.replicate_mean).all()
                and np.isfinite(profile.replicate_std).all()
            ),
        },
        "replicates": [
            replicate_summary(profile, rep_idx) for rep_idx in range(profile.n_replicates)
        ],
        "references": {
            reference.label: {
                "controller": "analytical_lqr_kalman_output_feedback",
                "display_label": reference.label,
                "observation_channel": reference.observation_channel,
                "observation_dim": int(reference.observation_dim),
                "observed_physical_indices": list(reference.observed_physical_indices),
                "gamma_factor_recorded_for_certificate": float(reference.gamma_factor),
                "n_stochastic_samples": int(reference.n_samples),
                "parity_status": reference.parity_status,
                "n_time_steps": int(reference.forward_velocity.shape[0]),
                "peak_forward_velocity_m_s": float(reference.peak_forward_velocity_m_s),
                "time_of_peak_forward_velocity_s": float(
                    reference.time_of_peak_forward_velocity_s
                ),
                "terminal_position_error_m": float(reference.terminal_position_error_m),
            }
            for reference in references
        },
    }


def replicate_summary(profile: VelocityProfile, rep_idx: int) -> dict[str, float | int]:
    """Return one replicate profile summary."""

    peak_idx = int(np.nanargmax(profile.replicate_mean[rep_idx]))
    return {
        "replicate": int(rep_idx),
        "peak_mean_forward_velocity_m_s": float(profile.replicate_mean[rep_idx, peak_idx]),
        "time_of_peak_mean_forward_velocity_s": float(profile.time_s[peak_idx]),
        "trial_sd_at_peak_m_s": float(profile.replicate_std[rep_idx, peak_idx]),
    }


rgba = hex_to_rgba


def repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    """Return repo-relative path text when possible."""

    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
