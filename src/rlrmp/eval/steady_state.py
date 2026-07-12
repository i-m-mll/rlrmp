"""Evaluation-time steady-state trial transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
import numpy as np

from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.eval.sisu_spectrum import zero_disturbance_payload
from rlrmp.eval.gru_diagnostics import RolloutEvaluation

_PRESET = load_analysis_parameter_preset("gru_steady_state_perturbation_bank").parameters
DEFAULT_PRE_GO_STEPS = int(_PRESET["pre_go_steps"])
DEFAULT_POST_GO_WASHIN_STEPS = int(_PRESET["post_go_washin_steps"])
DEFAULT_PULSE_DURATION_STEPS = int(_PRESET["pulse_duration_steps"])
DEFAULT_FINAL_WINDOW_STEPS = int(_PRESET["final_window_steps"])
DEFAULT_N_ROLLOUT_TRIALS = int(_PRESET["n_rollout_trials"])
DEFAULT_POSITION_SCALE_M = float(_PRESET["position_scale_m"])
DEFAULT_VELOCITY_SCALE_M_S = float(_PRESET["velocity_scale_m_s"])
DEFAULT_FORCE_FILTER_SCALE = float(_PRESET["force_filter_scale"])
DEFAULT_PRE_ONSET_FIGURE_STEPS = int(_PRESET["pre_onset_figure_steps"])
DEFAULT_POST_ONSET_FIGURE_STEPS = int(_PRESET["post_onset_figure_steps"])


@dataclass(frozen=True)
class SteadyStatePerturbationBankConfig:
    """Entry-level defaults for steady-state feedback-bank materialization."""

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


def evaluate_model_on_trial_specs(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
) -> RolloutEvaluation:
    """Evaluate one transformed trial specification through the canonical evaluator."""

    from rlrmp.eval.perturbation_bank import _evaluate_model_on_trial_specs

    return _evaluate_model_on_trial_specs(
        model=model,
        task=task,
        trial_specs=trial_specs,
        n_replicates=n_replicates,
        seed=seed,
    )


def target_position(run: Any, trial_specs: Any) -> np.ndarray:
    """Resolve the governed target position for a steady-state evaluation."""

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


def feedback_dim(trial_specs: Any) -> int:
    """Return the active sensory-feedback width from trial inputs."""

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


def expected_feedback_dim_from_hps(hps: Any) -> int | None:
    """Return the declared controller-feedback width, when available."""

    contract = getattr(getattr(hps, "target_relative_multitarget", None), "input_contract", None)
    shape = getattr(contract, "shape", None)
    return int(shape[-1]) if shape else None


__all__ = [
    "SteadyStatePerturbationBankConfig",
    "evaluate_model_on_trial_specs",
    "expected_feedback_dim_from_hps",
    "extend_trial_specs_to_horizon",
    "feedback_dim",
    "make_steady_state_trial_specs",
    "pad_feedback_offset_inputs",
    "set_delayed_go_cue",
    "set_mechanics_vector_to_target",
    "set_target_streams_to_constant",
    "target_position",
]
