"""Ensemble (replicate-vmapped) trial evaluation.

Bug: 8404108 — extracted from ``results/2ef67ca/scripts/eval_part2_5_figures.py`` and made
training-method-agnostic.
"""

from __future__ import annotations

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
    DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.eval.kinematics import initial_effector_position, initial_effector_velocity
from rlrmp.runtime.parameter_presets import EvaluationEnsemblePreset, load_runtime_preset
from rlrmp.train.task_model import setup_task_model_pair

__all__ = [
    "DelayedEvalBank",
    "DelayedVelocityProfile",
    "N_REPLICATES",
    "eval_ensemble_on_trials",
    "evaluate_velocity_profile",
    "fixed_bank_projection_direction",
    "make_delayed_eval_bank",
]


#: Default number of replicates across the rlrmp project.
#:
#: Kept here as a module-level constant for backwards compatibility with the
#: legacy ``eval_part2_5_figures.py`` API. New code should prefer passing
#: ``n_replicates=`` explicitly to :func:`eval_ensemble_on_trials`.
N_REPLICATES: int = load_runtime_preset(
    "rlrmp.evaluation_ensemble.default",
    EvaluationEnsemblePreset,
).n_replicates


def eval_ensemble_on_trials(task, model, trial_specs, *, key, n_replicates: int = N_REPLICATES):
    """Evaluate ``n_replicates`` models on the given ``trial_specs``.

    Mirrors feedbax's ``_eval_ensemble`` partitioning strategy: model leaves
    that carry the replicate dimension (i.e. arrays whose leading axis has
    length ``n_replicates``) are vmapped over; everything else is held fixed
    via :func:`equinox.partition` / :func:`equinox.combine`.

    Args:
        task: The task object whose ``eval_trials`` is called per replicate.
        model: An ensembled model (replicate-batched along leading axis on the
            array leaves; ``StateIndex.init.field`` and similar non-batched
            leaves are handled automatically).
        trial_specs: ``TaskTrialSpec`` for the trials to evaluate.
        key: PRNGKey, split into ``n_replicates`` sub-keys.
        n_replicates: Replicate-axis size. Default :data:`N_REPLICATES`.

    Returns:
        States PyTree with leading replicate dimension:
        ``(n_replicates, n_trials, n_steps, ...)``.
    """
    n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]

    def _is_batched_array(x):
        return eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_replicates

    models_arrays, models_other = eqx.partition(model, _is_batched_array)

    def eval_one_replicate(model_arrays, model_other, rep_key):
        rep_model = eqx.combine(model_arrays, model_other)
        keys = jr.split(rep_key, n_trials)
        return task.eval_trials(rep_model, trial_specs, keys)

    rep_keys = jr.split(key, n_replicates)
    states = eqx.filter_vmap(
        eval_one_replicate,
        in_axes=(0, None, 0),
    )(models_arrays, models_other, rep_keys)
    return states


BankKind = Literal["no_catch", "catch"]


@dataclass(frozen=True)
class DelayedEvalBank:
    """A fixed delayed-reach evaluation bank and its versioned metadata."""

    trial_specs: TaskTrialSpec
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DelayedVelocityProfile:
    """A go-cue-aligned target-radial velocity profile."""

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
    experiment: str | None = None
    run_spec_path: Path | None = None
    artifact_dir: Path | None = None
    sisu_level: float | None = None

    @property
    def n_pooled_samples(self) -> int:
        """Return the replicate-by-trial sample count."""

        return int(self.n_replicates * self.n_trials_per_replicate)


def evaluate_velocity_profile(
    run: Any,
    *,
    bank_kind: BankKind,
    go_cue_steps: Sequence[int],
    direction_count: int,
    reach_length_m: float,
    pre_go_context_steps: int,
    sisu_level: float | None = None,
    include_sisu_metadata: bool = False,
) -> DelayedVelocityProfile:
    """Evaluate a final checkpoint on one canonical delayed-reach bank."""

    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run.run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    movement_horizon_steps = _canonical_movement_horizon(run.run_spec)
    bank = make_delayed_eval_bank(
        pair.task.validation_trials,
        bank_kind=bank_kind,
        go_cue_steps=go_cue_steps,
        direction_count=direction_count,
        reach_length_m=reach_length_m,
        movement_horizon_steps=movement_horizon_steps,
        sisu_level=sisu_level,
        include_sisu_metadata=include_sisu_metadata,
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
    direction = _target_radial_projection_direction(bank.trial_specs, bank.metadata)
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
    return DelayedVelocityProfile(
        experiment=getattr(run, "experiment", None),
        run_id=run.run_id,
        label=run.label,
        run_spec_path=getattr(run, "run_spec_path", None),
        artifact_dir=getattr(run, "artifact_dir", None),
        bank_kind=bank_kind,
        sisu_level=sisu_level,
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
    sisu_level: float | None = None,
    include_sisu_metadata: bool = False,
) -> DelayedEvalBank:
    """Return a uniform-grid, schema-versioned delayed-reach evaluation bank."""

    go_steps = tuple(int(step) for step in go_cue_steps)
    if not go_steps:
        raise ValueError("go_cue_steps must not be empty")
    if min(go_steps) < 0:
        raise ValueError(f"go cue steps must be nonnegative; got {go_steps}")
    source_trials = _infer_trial_count(trial_specs)
    n_state_steps = int(
        getattr(getattr(trial_specs, "timeline", None), "n_steps", None)
        or _infer_trial_n_time(trial_specs)
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
    no_catch_target = np.where(
        state_time[..., None] >= go_column[..., None],
        visible_target,
        init_sequence,
    )
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
            [np.zeros_like(go_index), go_index, np.full_like(go_index, n_state_steps)],
            axis=-1,
        ),
        epoch_names=("prep", "movement"),
        event_steps=go_index[:, None],
        event_names=("go_cue",),
    )
    remapped = _remap_source_trials(
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
    if include_sisu_metadata:
        metadata.update(
            {
                "sisu_level": None if sisu_level is None else float(sisu_level),
                "sisu_conditioning": (
                    "preserved_from_source_trials"
                    if sisu_level is None
                    else "explicit_constant_sisu_level"
                ),
            }
        )
    updated = TaskTrialSpec(
        inits=WhereDict(_update_initial_positions(remapped.inits, initial_position_batch)),
        inputs=_update_inputs(
            remapped.inputs,
            visible_target=visible_target,
            scored_target=scored_target,
            hold=control_hold,
            target_on=target_on,
            go_input=go_input,
            sisu_level=sisu_level,
        ),
        targets=WhereDict(_update_targets(remapped.targets, scored_target)),
        intervene=remapped.intervene,
        timeline=timeline,
        extra={
            **dict(remapped.extra or {}),
            "delayed_reach_eval_bank": metadata,
            "is_catch_trial": np.full(go_index.shape[0], bank_kind == "catch"),
        },
    )
    return DelayedEvalBank(trial_specs=updated, metadata=metadata)


def _remap_source_trials(
    trial_specs: TaskTrialSpec,
    *,
    source_index: np.ndarray,
    source_trials: int,
) -> TaskTrialSpec:
    indices = jnp.asarray(source_index, dtype=jnp.int32)

    def remap_leaf(leaf: Any) -> Any:
        shape = getattr(leaf, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) == int(source_trials):
            return jnp.asarray(leaf)[indices]
        return leaf

    return jt.map(remap_leaf, trial_specs)


def _update_initial_positions(
    inits: Mapping[Any, Any],
    position: np.ndarray,
) -> dict[Any, Any]:
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


def _update_targets(
    targets: Mapping[Any, Any],
    scored_target: np.ndarray,
) -> dict[Any, Any]:
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


def _update_inputs(
    inputs: Any,
    *,
    visible_target: np.ndarray,
    scored_target: np.ndarray,
    hold: np.ndarray,
    target_on: np.ndarray,
    go_input: np.ndarray,
    sisu_level: float | None = None,
) -> Any:
    if isinstance(inputs, DelayedReachTaskInputs):
        return _update_delayed_task_inputs(
            inputs,
            visible_target=visible_target,
            hold=hold,
            target_on=target_on,
        )
    if not isinstance(inputs, Mapping):
        return inputs
    updated = dict(inputs)
    if "input" in updated:
        updated["input"] = _update_controller_input(
            updated["input"],
            go_input=go_input,
            sisu_level=sisu_level,
        )
    if "sisu" in updated and sisu_level is not None:
        updated["sisu"] = jnp.full_like(jnp.asarray(updated["sisu"]), float(sisu_level))
    if "target" in updated:
        updated["target"] = _like_existing_array(updated["target"], visible_target)
    if "effector_target" in updated:
        updated["effector_target"] = _update_cartesian_position(
            updated["effector_target"],
            scored_target,
        )
    if "task" in updated and isinstance(updated["task"], DelayedReachTaskInputs):
        updated["task"] = _update_delayed_task_inputs(
            updated["task"],
            visible_target=visible_target,
            hold=hold,
            target_on=target_on,
        )
    return updated


def _update_controller_input(
    existing: Any,
    *,
    go_input: np.ndarray,
    sisu_level: float | None,
) -> jnp.ndarray:
    array = jnp.asarray(existing)
    if array.ndim >= 3 and array.shape[-1] >= 2:
        replacement = array.at[..., 0].set(jnp.asarray(go_input, dtype=array.dtype))
        if sisu_level is not None:
            replacement = replacement.at[..., 1].set(
                jnp.full_like(replacement[..., 1], float(sisu_level))
            )
        return replacement
    return _like_existing_time_input(existing, go_input)


def _update_delayed_task_inputs(
    inputs: DelayedReachTaskInputs,
    *,
    visible_target: np.ndarray,
    hold: np.ndarray,
    target_on: np.ndarray,
) -> DelayedReachTaskInputs:
    effector_target = _update_cartesian_position(inputs.effector_target, visible_target)
    if getattr(effector_target, "pos", None) is None:
        effector_target = CartesianState(pos=jnp.asarray(visible_target))
    return DelayedReachTaskInputs(
        effector_target=effector_target,
        hold=_like_existing_time_input(inputs.hold, hold),
        target_on=_like_existing_time_input(inputs.target_on, target_on),
    )


def _update_cartesian_position(value: Any, position: np.ndarray) -> Any:
    current = getattr(value, "pos", None)
    if current is None:
        return value
    return eqx.tree_at(
        lambda state: state.pos,
        value,
        jnp.asarray(position, dtype=jnp.asarray(current).dtype),
    )


def _like_existing_time_input(existing: Any, value: np.ndarray) -> jnp.ndarray:
    array = jnp.asarray(existing)
    replacement = jnp.asarray(value, dtype=array.dtype)
    if array.ndim >= 3 and array.shape[-1] == 1 and replacement.ndim == array.ndim - 1:
        replacement = replacement[..., None]
    return replacement


def _like_existing_array(existing: Any, value: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.asarray(existing).dtype)


def _target_position_sequence(trial_specs: Any) -> np.ndarray:
    for target_spec in trial_specs.targets.values():
        value = getattr(target_spec, "value", None)
        if getattr(value, "shape", None) is not None and value.shape[-1] == 2:
            return np.asarray(value, dtype=np.float64)
    raise ValueError("Trial spec does not include 2D target sequence")


def _target_radial_projection_direction(
    trial_specs: Any,
    bank_metadata: Mapping[str, Any],
) -> np.ndarray:
    initial_position = initial_effector_position(trial_specs)
    fixed_direction = fixed_bank_projection_direction(
        bank_metadata,
        trial_count=int(initial_position.shape[0]),
    )
    if fixed_direction is not None:
        return fixed_direction
    target_position = _target_position_sequence(trial_specs)
    direction, _distance = _reach_direction(initial_position, target_position[:, -1, :])
    return direction


def fixed_bank_projection_direction(
    bank_metadata: Mapping[str, Any],
    *,
    trial_count: int,
) -> np.ndarray | None:
    """Return fixed-bank target directions from the bank metadata."""

    if bank_metadata.get("bank_family") != "delayed_reach_fixed_eval_bank":
        return None
    angles = bank_metadata.get("target_angles_rad")
    direction_count = int(bank_metadata.get("direction_count", 0))
    if angles is None or direction_count <= 0:
        return None
    angles_array = np.asarray(angles, dtype=np.float64)
    if angles_array.shape != (direction_count,):
        raise ValueError(
            "fixed-bank target_angles_rad must have shape "
            f"({direction_count},); got {angles_array.shape}"
        )
    if trial_count % direction_count != 0:
        raise ValueError(
            "fixed-bank trial count must be a multiple of direction_count; "
            f"got trial_count={trial_count}, direction_count={direction_count}"
        )
    directions = np.stack([np.cos(angles_array), np.sin(angles_array)], axis=-1)
    return np.tile(directions, (trial_count // direction_count, 1))


def _reach_direction(
    initial_position: np.ndarray,
    target_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = target_position - initial_position
    distance = np.linalg.norm(delta, axis=-1)
    safe_distance = np.where(distance > 0.0, distance, 1.0)
    return delta / safe_distance[:, None], distance


def _infer_trial_count(trial_specs: Any) -> int:
    for target_spec in getattr(trial_specs, "targets", {}).values():
        value = getattr(target_spec, "value", None)
        if getattr(value, "shape", None) is not None and len(value.shape) >= 1:
            return int(value.shape[0])
    return int(initial_effector_position(trial_specs).shape[0])


def _infer_trial_n_time(trial_specs: Any) -> int:
    try:
        return int(_target_position_sequence(trial_specs).shape[1])
    except ValueError:
        pass
    inputs = getattr(trial_specs, "inputs", None)
    if isinstance(inputs, Mapping) and "target" in inputs:
        target = jnp.asarray(inputs["target"])
        if target.ndim >= 2:
            return int(target.shape[1])
    raise ValueError("Could not infer trial time axis")


def _canonical_movement_horizon(run_spec: Mapping[str, Any]) -> int:
    movement = run_spec.get("delayed_reach", {}).get("movement_epoch", {})
    if "cs_schedule_horizon_steps" in movement:
        return int(movement["cs_schedule_horizon_steps"])
    projection = run_spec.get("game_card", {}).get("delayed_reach_projection", {})
    if "canonical_cs_movement_horizon_steps" in projection:
        return int(projection["canonical_cs_movement_horizon_steps"])
    if "horizon_steps" in run_spec.get("game_card", {}):
        return int(run_spec["game_card"]["horizon_steps"])
    return 60
