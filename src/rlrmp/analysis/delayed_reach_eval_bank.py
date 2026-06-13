"""Fixed delayed-reach evaluation banks for post-run analyses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax.tree as jt
import jax.numpy as jnp
import numpy as np
from feedbax._mapping import WhereDict
from feedbax.loss import TargetSpec
from feedbax.state import CartesianState
from feedbax.task import DelayedReachTaskInputs, TaskTrialSpec, TrialTimeline

from rlrmp.analysis.trial_alignment import infer_trial_count, infer_trial_n_time


DEFAULT_GO_CUE_STEPS = tuple(range(10, 31))
DEFAULT_DIRECTION_COUNT = 20
DEFAULT_UNIFORM_REACH_LENGTH_M = 0.15
DirectionSource = Literal["uniform_grid", "validation_targets"]


@dataclass(frozen=True)
class DelayedReachEvalBank:
    """A deterministic delayed-reach evaluation bank plus JSON metadata."""

    trial_specs: TaskTrialSpec
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _DirectionBank:
    direction_source: str
    targets: np.ndarray
    source_indices: np.ndarray
    target_radii_m: np.ndarray
    target_angles_rad: np.ndarray
    reach_length_m: float
    reach_length_source: str
    duplicate_direction_count: int
    source_target_radii_m: np.ndarray | None = None
    source_target_angles_rad: np.ndarray | None = None

    @property
    def count(self) -> int:
        return int(self.targets.shape[0])


def make_delayed_reach_eval_bank(
    trial_specs: TaskTrialSpec,
    *,
    catch: bool,
    go_cue_steps: Iterable[int] = DEFAULT_GO_CUE_STEPS,
    direction_count: int = DEFAULT_DIRECTION_COUNT,
    direction_source: DirectionSource = "uniform_grid",
    movement_horizon_steps: int | None = None,
    reach_length_m: float | None = None,
) -> DelayedReachEvalBank:
    """Return a fixed delayed-reach eval bank spanning go cues and directions.

    The supplied ``trial_specs`` is used as an adapted template, so C&S LSS
    initial-state vectors, epsilon channels, target-relative inputs, and graph
    adapter payloads keep the shape and dtype conventions of the trained run.
    The helper deterministically remaps the delayed-reach axes only: center-out
    direction, go-cue step, and no-catch/catch scoring.
    """

    go_steps = tuple(int(step) for step in go_cue_steps)
    if not go_steps:
        raise ValueError("go_cue_steps must contain at least one step")
    if min(go_steps) < 0:
        raise ValueError(f"go cue steps must be nonnegative; got {go_steps}")
    if direction_source not in {"uniform_grid", "validation_targets"}:
        raise ValueError(
            "direction_source must be one of 'uniform_grid' or 'validation_targets'; "
            f"got {direction_source!r}"
        )
    if direction_count < 1:
        raise ValueError("direction_count must be at least 1")
    if direction_source == "uniform_grid" and direction_count < 2:
        raise ValueError("direction_count must be at least 2 for uniform center-out evaluation")

    source_trials = infer_trial_count(trial_specs)
    n_time_steps = int(
        getattr(getattr(trial_specs, "timeline", None), "n_steps", None)
        or infer_trial_n_time(trial_specs)
    )
    max_go = int(max(go_steps))
    horizon = int(movement_horizon_steps or (n_time_steps - max_go))
    if horizon < 1:
        raise ValueError(f"movement_horizon_steps must be positive; got {horizon}")
    if max_go + horizon > n_time_steps:
        raise ValueError(
            "delayed evaluation bank does not fit the trial time axis: "
            f"go_max={max_go}, horizon={horizon}, n_time_steps={n_time_steps}"
        )

    initial_positions = _initial_positions(trial_specs)
    initial_position = np.asarray(initial_positions[0], dtype=np.float64)
    direction_bank = _direction_bank(
        trial_specs,
        initial_position=initial_position,
        direction_count=direction_count,
        direction_source=direction_source,
        reach_length_m=reach_length_m,
    )

    go_index = np.repeat(np.asarray(go_steps, dtype=np.int32), direction_bank.count)
    direction_index = np.tile(
        np.arange(direction_bank.count, dtype=np.int32),
        len(go_steps),
    )
    source_index = np.asarray(direction_bank.source_indices[direction_index], dtype=np.int32)
    target_position = direction_bank.targets[direction_index]
    initial_position_batch = np.broadcast_to(
        initial_position[None, :],
        (go_index.shape[0], 2),
    )
    time_index = np.arange(n_time_steps, dtype=np.int32)[None, :]
    go_column = go_index[:, None]
    target_sequence = np.broadcast_to(
        target_position[:, None, :],
        (go_index.shape[0], n_time_steps, 2),
    )
    init_sequence = np.broadcast_to(
        initial_position_batch[:, None, :],
        (go_index.shape[0], n_time_steps, 2),
    )
    movement_mask = time_index >= go_column
    scored_target = (
        init_sequence
        if catch
        else np.where(movement_mask[..., None], target_sequence, init_sequence)
    )
    hold = (
        np.ones((go_index.shape[0], n_time_steps), dtype=np.float32)
        if catch
        else (time_index < go_column).astype(np.float32)
    )
    go_input = 1.0 - hold
    target_on = np.ones_like(hold, dtype=np.float32)
    timeline = TrialTimeline.from_epochs_events(
        n_time_steps,
        epoch_bounds=np.stack(
            [
                np.zeros_like(go_index),
                go_index,
                np.full_like(go_index, n_time_steps),
            ],
            axis=-1,
        ),
        epoch_names=("prep", "movement"),
    )

    remapped = _remap_source_trials(
        trial_specs,
        source_index=source_index,
        source_trials=source_trials,
    )
    bank_metadata = {
        "schema_version": "rlrmp.delayed_reach_eval_bank.v2",
        "kind": "catch" if catch else "no_catch",
        "catch": bool(catch),
        "go_cue_min": int(min(go_steps)),
        "go_cue_max": int(max(go_steps)),
        "go_cue_steps": [int(step) for step in go_steps],
        "direction_source": direction_bank.direction_source,
        "direction_count": int(direction_bank.count),
        "requested_direction_count": int(direction_count),
        "trial_count": int(go_index.shape[0]),
        "n_time_steps": int(n_time_steps),
        "movement_horizon_steps": int(horizon),
        "reach_length_m": float(direction_bank.reach_length_m),
        "reach_length_source": direction_bank.reach_length_source,
        "reach_length_m_explicit": bool(reach_length_m is not None),
        "direction_source_inferred_from_validation_targets": (
            direction_bank.direction_source == "validation_targets"
        ),
        "duplicate_direction_count": int(direction_bank.duplicate_direction_count),
        "target_radii_m": [float(radius) for radius in direction_bank.target_radii_m],
        "target_angles_rad": [float(angle) for angle in direction_bank.target_angles_rad],
        "source_trial_indices": [int(index) for index in direction_bank.source_indices],
    }
    if direction_bank.source_target_radii_m is not None:
        bank_metadata["source_target_radii_m"] = [
            float(radius) for radius in direction_bank.source_target_radii_m
        ]
    if direction_bank.source_target_angles_rad is not None:
        bank_metadata["source_target_angles_rad"] = [
            float(angle) for angle in direction_bank.source_target_angles_rad
        ]

    updated = TaskTrialSpec(
        inits=WhereDict(_update_initial_positions(remapped.inits, initial_position_batch)),
        inputs=_update_inputs(
            remapped.inputs,
            visible_target=target_sequence,
            scored_target=scored_target,
            hold=hold,
            target_on=target_on,
            go_input=go_input,
        ),
        targets=WhereDict(_update_targets(remapped.targets, scored_target)),
        intervene=remapped.intervene,
        timeline=timeline,
        extra={
            **dict(remapped.extra or {}),
            "delayed_reach_eval_bank": bank_metadata,
        },
    )
    return DelayedReachEvalBank(trial_specs=updated, metadata=bank_metadata)


def make_delayed_reach_eval_banks(
    trial_specs: TaskTrialSpec,
    **kwargs: Any,
) -> dict[str, DelayedReachEvalBank]:
    """Return paired no-catch and catch delayed-reach evaluation banks."""

    return {
        "no_catch": make_delayed_reach_eval_bank(
            trial_specs,
            catch=False,
            **kwargs,
        ),
        "catch": make_delayed_reach_eval_bank(
            trial_specs,
            catch=True,
            **kwargs,
        ),
    }


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


def _initial_positions(trial_specs: TaskTrialSpec) -> np.ndarray:
    for init_state in trial_specs.inits.values():
        position = getattr(init_state, "pos", None)
        if position is not None:
            return np.asarray(position, dtype=np.float64)
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 2:
            return np.asarray(init_state, dtype=np.float64)[..., 0:2]
    raise ValueError("trial_specs does not include an effector position initial state")


def _target_positions(trial_specs: TaskTrialSpec) -> np.ndarray:
    inputs = getattr(trial_specs, "inputs", {})
    if isinstance(inputs, Mapping):
        target_input = inputs.get("target")
        if getattr(target_input, "shape", None) is not None and target_input.shape[-1] == 2:
            return np.asarray(target_input, dtype=np.float64)[..., -1, :]
    for target_spec in trial_specs.targets.values():
        value = getattr(target_spec, "value", None)
        if getattr(value, "shape", None) is not None and value.shape[-1] == 2:
            return np.asarray(value, dtype=np.float64)[..., -1, :]
    raise ValueError("trial_specs does not include a 2D target sequence")


def _direction_bank(
    trial_specs: TaskTrialSpec,
    *,
    initial_position: np.ndarray,
    direction_count: int,
    direction_source: DirectionSource,
    reach_length_m: float | None,
) -> _DirectionBank:
    target_positions = _target_positions(trial_specs)
    deltas = target_positions - initial_position[None, :]
    distances = np.linalg.norm(deltas, axis=-1)
    valid = distances > 0.0
    source_directions = np.zeros_like(deltas, dtype=np.float64)
    source_directions[valid] = deltas[valid] / distances[valid, None]
    source_angles = _direction_angles_rad(source_directions[valid])
    source_radii = distances[valid].astype(np.float64)

    if direction_source == "uniform_grid":
        reach_length = (
            float(DEFAULT_UNIFORM_REACH_LENGTH_M)
            if reach_length_m is None
            else float(reach_length_m)
        )
        reach_length_source = "uniform_default" if reach_length_m is None else "explicit"
        angles = np.linspace(0.0, 2.0 * np.pi, int(direction_count), endpoint=False)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        targets = initial_position[None, :] + reach_length * directions
        source_indices = np.arange(int(direction_count), dtype=np.int32) % int(
            infer_trial_count(trial_specs)
        )
        return _DirectionBank(
            direction_source=direction_source,
            targets=targets.astype(np.float64),
            source_indices=source_indices,
            target_radii_m=np.full(int(direction_count), reach_length, dtype=np.float64),
            target_angles_rad=_direction_angles_rad(directions),
            reach_length_m=reach_length,
            reach_length_source=reach_length_source,
            duplicate_direction_count=_duplicate_direction_count(directions),
            source_target_radii_m=source_radii,
            source_target_angles_rad=source_angles,
        )

    valid_indices = np.flatnonzero(valid)
    if valid_indices.shape[0] < int(direction_count):
        raise ValueError(
            "direction_source='validation_targets' requires at least direction_count "
            "nonzero validation target directions; "
            f"got {valid_indices.shape[0]} valid targets for direction_count={direction_count}"
        )
    selected_indices = valid_indices[: int(direction_count)]
    selected_directions = source_directions[selected_indices]
    if reach_length_m is None:
        targets = target_positions[selected_indices].astype(np.float64)
        radii = distances[selected_indices].astype(np.float64)
        reach_length = float(np.median(radii))
        reach_length_source = "validation_targets_median"
    else:
        reach_length = float(reach_length_m)
        targets = initial_position[None, :] + reach_length * selected_directions
        radii = np.full(int(direction_count), reach_length, dtype=np.float64)
        reach_length_source = "explicit"
    return _DirectionBank(
        direction_source=direction_source,
        targets=targets.astype(np.float64),
        source_indices=selected_indices.astype(np.int32),
        target_radii_m=radii,
        target_angles_rad=_direction_angles_rad(selected_directions),
        reach_length_m=reach_length,
        reach_length_source=reach_length_source,
        duplicate_direction_count=_duplicate_direction_count(selected_directions),
        source_target_radii_m=source_radii,
        source_target_angles_rad=source_angles,
    )


def _direction_angles_rad(directions: np.ndarray) -> np.ndarray:
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    return np.mod(angles, 2.0 * np.pi).astype(np.float64)


def _duplicate_direction_count(directions: np.ndarray) -> int:
    if directions.shape[0] == 0:
        return 0
    rounded = np.round(_direction_angles_rad(directions), decimals=12)
    return int(rounded.shape[0] - np.unique(rounded).shape[0])


def _update_initial_positions(
    inits: Mapping[Any, Any],
    initial_position: np.ndarray,
) -> dict[Any, Any]:
    updated = dict(inits)
    for key, init_state in list(updated.items()):
        position = getattr(init_state, "pos", None)
        if position is not None:
            updated[key] = eqx.tree_at(
                lambda state: state.pos,
                init_state,
                jnp.asarray(initial_position, dtype=jnp.asarray(position).dtype),
            )
            continue
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 2:
            value = jnp.asarray(init_state)
            updated[key] = value.at[..., 0:2].set(
                jnp.asarray(initial_position, dtype=value.dtype)
            )
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
            continue
        if isinstance(target_spec, TargetSpec):
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
) -> Any:
    if isinstance(inputs, DelayedReachTaskInputs):
        return _update_delayed_task_inputs(
            inputs,
            visible_target=visible_target,
            scored_target=scored_target,
            hold=hold,
            target_on=target_on,
        )
    if not isinstance(inputs, Mapping):
        return inputs
    updated = dict(inputs)
    if "input" in updated:
        updated["input"] = _like_existing_time_input(updated["input"], go_input)
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
            scored_target=scored_target,
            hold=hold,
            target_on=target_on,
        )
    return updated


def _update_delayed_task_inputs(
    inputs: DelayedReachTaskInputs,
    *,
    visible_target: np.ndarray,
    scored_target: np.ndarray,
    hold: np.ndarray,
    target_on: np.ndarray,
) -> DelayedReachTaskInputs:
    effector_target = _update_cartesian_position(inputs.effector_target, visible_target)
    if getattr(effector_target, "pos", None) is None:
        effector_target = CartesianState(pos=jnp.asarray(visible_target))
    del scored_target
    return DelayedReachTaskInputs(
        effector_target=effector_target,
        hold=_like_existing_time_input(inputs.hold, hold),
        target_on=_like_existing_time_input(inputs.target_on, target_on),
    )


def _update_cartesian_position(value: Any, position: np.ndarray) -> Any:
    current = getattr(value, "pos", None)
    if current is not None:
        return eqx.tree_at(
            lambda state: state.pos,
            value,
            jnp.asarray(position, dtype=jnp.asarray(current).dtype),
        )
    return value


def _like_existing_time_input(existing: Any, value: np.ndarray) -> jnp.ndarray:
    array = jnp.asarray(existing)
    replacement = jnp.asarray(value, dtype=array.dtype)
    if array.ndim >= 3 and array.shape[-1] == 1:
        replacement = replacement[..., None]
    return replacement


def _like_existing_array(existing: Any, value: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.asarray(existing).dtype)


__all__ = [
    "DEFAULT_DIRECTION_COUNT",
    "DEFAULT_GO_CUE_STEPS",
    "DEFAULT_UNIFORM_REACH_LENGTH_M",
    "DelayedReachEvalBank",
    "DirectionSource",
    "make_delayed_reach_eval_bank",
    "make_delayed_reach_eval_banks",
]
