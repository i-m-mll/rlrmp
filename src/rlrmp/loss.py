from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from feedbax import TaskTrialSpec
from feedbax.runtime.model import AbstractModel
from feedbax.objectives.loss import (
    AbstractLoss,
    CompositeLoss,
    EpochMaskedLoss,
    FuncTermsLoss,
    OutputJerkLoss,
    StateDerivativeLoss,
    TargetSpec,
    TargetStateLoss,
    TermTree,
    reduce_over_time_with_weights,
    target_final_state,
    target_zero,
)
from feedbax.config.utils import deep_merge
from feedbax.training.loss import get_readout_norm_loss
from feedbax.config.namespace import TreeNamespace
from jax_cookbook.misc import window_take
from jaxtyping import Array, PyTree

from rlrmp.analysis.math.cs_game_card import (
    TARGET_POS,
    build_canonical_game,
    build_no_integrator_game,
)
from rlrmp.loss_presets import load_loss_preset

CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE = "full_analytical_qrf"
CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE = "partial_net_output_force_filter"
CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE = "partial_feedbax_terms"
CS_LOSS_OBJECTIVES = (
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
)


def softmin_and_alpha(values, tau, axis=-1, eps=1e-12, keepdims=False):
    # values: (..., T) with T = |W|
    m = jnp.min(values, axis=axis, keepdims=True)  # shift for stability
    z = jnp.exp(-(values - m) / tau)
    S = jnp.sum(z, axis=axis, keepdims=True)
    W = values.shape[axis]
    softmin = tau * (jnp.log(W) - jnp.log(S)) + m
    alpha = z / (S + eps)
    if not keepdims:
        softmin = jnp.squeeze(softmin, axis=axis)
    return softmin, alpha


class GoalHitLossCtx(eqx.Module):
    pos_err: Array
    pos_err_in_window: Array
    vel_in_window: Array
    alpha: Array
    go_idx: Array
    hit_time: Array
    t_all: Array
    window_bounds: tuple[int, int]
    pos_hit_term: Array


class GoalHitCtxBuilder(eqx.Module):
    window_bounds: tuple[int, int]
    tau: float
    eps: float

    def __call__(self, states, trial_specs, model) -> GoalHitLossCtx:
        pos = states.mechanics.effector.pos[:, 1:]
        vel = states.mechanics.effector.vel[:, 1:]
        goal = trial_specs.targets["mechanics.effector.pos"].value
        go_idx = trial_specs.timeline.epoch_bounds[:, -2]

        #! Might be better to use axis=0 for trial and axis=1 for time. Is that right?
        # axis_p: time axis; axis_q: trial axis
        pos_err = jnp.sum((pos - goal) ** 2, axis=-1)
        pos_err_win = window_take(  # your existing util
            pos_err, go_idx, self.window_bounds, axis_p=-1, axis_q=-2
        )

        pos_hit_term, alpha = softmin_and_alpha(pos_err_win, self.tau, axis=-1, eps=self.eps)

        # Velocity still has components here.
        vel_win = window_take(vel, go_idx, self.window_bounds, axis_p=-2, axis_q=-3)

        start, end = self.window_bounds
        hit_time = go_idx + jnp.dot(alpha, jnp.arange(start, end))

        t_all = jnp.arange(pos.shape[1])

        return GoalHitLossCtx(
            pos_err=pos_err,
            pos_err_in_window=pos_err_win,
            vel_in_window=vel_win,
            alpha=alpha,
            go_idx=go_idx,
            hit_time=hit_time,
            t_all=t_all,
            window_bounds=self.window_bounds,
            pos_hit_term=pos_hit_term,
        )


def goal_hit_pos_loss_term_fn(ctx: GoalHitLossCtx) -> Array:
    return ctx.pos_hit_term


def goal_hit_vel_loss_term_fn(ctx: GoalHitLossCtx) -> Array:
    v_at_hit = jnp.einsum("btd,bt->bd", ctx.vel_in_window, ctx.alpha)
    # norm over velocity components
    return jnp.linalg.norm(v_at_hit, axis=-1)


def post_hit_pos_loss_term_fn(ctx: GoalHitLossCtx, sigma_t: float, eps: float = 1e-12) -> Array:
    s_t = jax.nn.sigmoid((ctx.t_all[None, :] - ctx.hit_time[:, None]) / sigma_t)

    # only aggregate after go (optional): mask = (t >= go_idx)
    after_go_mask = ctx.t_all[None, :] >= ctx.go_idx[:, None]  # [B, T]
    w = s_t * after_go_mask
    # weighted post-hit position error per batch
    #! assume the time axis is always the last one, by now
    post = jnp.sum(w * ctx.pos_err, axis=-1) / (jnp.sum(w, axis=-1) + eps)  # [B]
    return post


#! TODO: Implement this
def goal_hit_late_pos_loss_term_fn(ctx: GoalHitLossCtx) -> Array:
    # Reverse-discounted (heavier toward trial end) sum of `ctx.pos_err` for timesteps starting from
    # `ctx.go_idx + ctx.window_bounds[1]`
    ...


def _nsget(ns: Any, path: str, default: Any = None) -> Any:
    """Dot-path getattr for TreeNamespace/dict hybrids."""
    cur = ns
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _as_number(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _pick(d: Mapping[str, Any], keys) -> dict[str, Any]:
    return {k: d[k] for k in keys if k in d}


def _filter_nonzero(d: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in d.items() if float(v) != 0.0}


class SimpleReachPositionLoss(TargetStateLoss):
    """Position target loss for Feedbax ``SimpleReaches`` transition timelines.

    ``SimpleReaches`` emits one target per transition (``n_steps - 1``) and the
    rollout states are already transition-aligned. The generic Feedbax
    ``TargetStateLoss`` drops ``states[:, 0]`` for tasks that include an initial
    state in their history, which is wrong for this task/loss contract.
    """

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert states is not None, "SimpleReachPositionLoss requires states, but states is None"
        assert trial_specs is not None, (
            "SimpleReachPositionLoss requires trial_specs, but trial_specs is None"
        )

        state = self.where(states)

        if (task_target_spec := trial_specs.targets.get(self.key, None)) is None:
            if self.spec is None:
                raise ValueError(
                    "`TargetSpec` must be provided on construction of "
                    "`SimpleReachPositionLoss`, or as part of the trial specifications"
                )
            target_spec = self.spec
        elif isinstance(task_target_spec, TargetSpec):
            target_spec: TargetSpec = eqx.combine(self.spec, task_target_spec)
        elif isinstance(task_target_spec, Mapping):
            target_spec = eqx.combine(self.spec, task_target_spec[self.label])
        else:
            raise ValueError("Invalid target spec encountered")

        loss_over_time = self.norm(state - target_spec.value)

        time_mask = target_spec.time_mask
        if time_mask is None:
            time_mask = target_spec.get_time_mask(loss_over_time.shape[-1])

        discount = target_spec.discount
        if discount is not None and getattr(discount, "ndim", 0) > 1:
            discount = discount[0]

        masks = [x for x in [time_mask, discount] if x is not None]
        return reduce_over_time_with_weights(
            label=self.label,
            arr=loss_over_time,
            trial_specs=trial_specs,
            time_axis=-1,
            trial_axis=0,
            trial_axis_specs=0,
            masks=masks,
        )


class CsAnalyticalQrfLoss(AbstractLoss):
    """Full C&S analytical finite-horizon quadratic loss over LSS state and command."""

    label: str
    Q: Array
    R: Array
    Q_f: Array
    target_pos: Array
    n_phys: int = eqx.field(static=True)
    delayed_movement_cost_tail_mode: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        Q: Array,
        R: Array,
        Q_f: Array,
        target_pos: Array,
        n_phys: int = 8,
        delayed_movement_cost_tail_mode: str = "canonical_window",
        label: str = CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    ):
        self.label = label
        self.Q = jnp.asarray(Q)
        self.R = jnp.asarray(R)
        self.Q_f = jnp.asarray(Q_f)
        self.target_pos = jnp.asarray(target_pos)
        self.n_phys = int(n_phys)
        self.delayed_movement_cost_tail_mode = str(delayed_movement_cost_tail_mode)

        if self.Q.ndim != 3:
            raise ValueError(f"Q must have shape [T, n, n], got {self.Q.shape}.")
        if self.R.ndim != 3:
            raise ValueError(f"R must have shape [T, m, m], got {self.R.shape}.")
        if self.Q_f.shape != self.Q.shape[1:]:
            raise ValueError(f"Q_f must have shape {self.Q.shape[1:]}, got {self.Q_f.shape}.")
        if self.Q.shape[0] != self.R.shape[0]:
            raise ValueError(
                f"Q and R must have the same horizon length, got {self.Q.shape[0]} "
                f"and {self.R.shape[0]}."
            )
        if self.Q.shape[1] % self.n_phys != 0:
            raise ValueError(
                f"state dimension {self.Q.shape[1]} is not divisible by n_phys={self.n_phys}."
            )
        if self.target_pos.shape != (2,):
            raise ValueError(f"target_pos must have shape [2], got {self.target_pos.shape}.")
        if self.delayed_movement_cost_tail_mode not in {
            "canonical_window",
            "flat_after_canonical_horizon",
        }:
            raise ValueError(
                "delayed_movement_cost_tail_mode must be 'canonical_window' or "
                f"'flat_after_canonical_horizon', got {delayed_movement_cost_tail_mode!r}."
            )

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        del model
        assert states is not None, "CsAnalyticalQrfLoss requires states, but states is None"
        assert trial_specs is not None, (
            "CsAnalyticalQrfLoss requires trial_specs, but trial_specs is None"
        )
        if not hasattr(states, "mechanics") or not hasattr(states.mechanics, "vector"):
            raise ValueError(
                "Full analytical Q/R/Q_f loss requires states.mechanics.vector from the "
                "cs_lss LinearStateSpace backend."
            )
        if not hasattr(states, "net") or not hasattr(states.net, "output"):
            raise ValueError(
                "Full analytical Q/R/Q_f loss requires states.net.output controller command "
                "history from the cs_lss GRU graph."
            )

        vector = jnp.asarray(states.mechanics.vector)
        command = jnp.asarray(states.net.output)
        if vector.shape[-1] != self.Q.shape[1]:
            raise ValueError(
                f"Full analytical Q/R/Q_f loss expected state dim {self.Q.shape[1]}, "
                f"got {vector.shape[-1]}."
            )
        if command.shape[-1] != self.R.shape[-1]:
            raise ValueError(
                f"Full analytical Q/R/Q_f loss expected command dim {self.R.shape[-1]}, "
                f"got {command.shape[-1]}."
            )
        horizon = self.Q.shape[0]
        movement_start = self._movement_start(trial_specs)
        if vector.shape[-2] != horizon and movement_start is None:
            raise ValueError(
                f"Full analytical Q/R/Q_f loss expected {horizon} rollout states for "
                "non-delayed rows, or delayed-reach epoch bounds for movement slicing; "
                f"got {vector.shape[-2]} states."
            )
        if command.shape[-2] != vector.shape[-2]:
            raise ValueError(
                "Full analytical Q/R/Q_f loss expected command and state histories to "
                f"have the same time length; got {command.shape[-2]} and {vector.shape[-2]}."
            )

        initial_vector = self._initial_vector(trial_specs, vector)
        target_pos = self._target_pos_for_trial(trial_specs, vector)
        x_pre_all = jnp.concatenate([initial_vector[..., None, :], vector[..., :-1, :]], axis=-2)
        if movement_start is None:
            x_pre = x_pre_all
            command_window = command
            x_terminal_raw = vector[..., -1, :]
        elif self.delayed_movement_cost_tail_mode == "flat_after_canonical_horizon":
            return self._delayed_flat_tail_term(
                x_pre_all=x_pre_all,
                command=command,
                vector=vector,
                movement_start=movement_start,
                target_pos=target_pos,
            )
        else:
            x_pre = self._time_window(x_pre_all, movement_start, horizon)
            command_window = self._time_window(command, movement_start, horizon)
            x_terminal_raw = self._time_index(vector, movement_start + horizon - 1)
        x_pre = self._goal_centered(x_pre, target_pos)
        x_terminal = self._goal_centered(x_terminal_raw, target_pos)
        state_terms = jnp.einsum("...ti,tij,...tj->...t", x_pre, self.Q, x_pre)
        command_terms = jnp.einsum("...ti,tij,...tj->...t", command_window, self.R, command_window)
        terminal = jnp.einsum("...i,ij,...j->...", x_terminal, self.Q_f, x_terminal)
        return jnp.sum(state_terms + command_terms, axis=-1) + terminal

    def _delayed_flat_tail_term(
        self,
        *,
        x_pre_all: Array,
        command: Array,
        vector: Array,
        movement_start: Array,
        target_pos: Array,
    ) -> Array:
        horizon = self.Q.shape[0]
        time = x_pre_all.shape[-2]
        t = jnp.arange(time, dtype=jnp.int32)
        starts = jnp.asarray(movement_start, dtype=jnp.int32)
        leading_shape = x_pre_all.shape[:-2]
        flat_x = x_pre_all.reshape((-1, time, x_pre_all.shape[-1]))
        flat_command = command.reshape((-1, time, command.shape[-1]))
        flat_target = jnp.broadcast_to(target_pos, (*leading_shape, 2)).reshape((-1, 2))
        flat_starts = jnp.broadcast_to(starts, leading_shape).reshape((-1,))
        flat_vector = vector.reshape((-1, time, vector.shape[-1]))

        def score_one_with_terminal(x_one, command_one, vector_one, target_one, start_one):
            age = t - start_one
            active = age >= 0
            stage = jnp.clip(age, 0, horizon - 1)
            x_centered = self._goal_centered(x_one, target_one)
            terminal = self._goal_centered(vector_one[-1], target_one)
            q = self.Q[stage]
            r = self.R[stage]
            state_terms = jnp.einsum("ti,tij,tj->t", x_centered, q, x_centered)
            command_terms = jnp.einsum("ti,tij,tj->t", command_one, r, command_one)
            terminal_term = jnp.einsum("i,ij,j->", terminal, self.Q_f, terminal)
            return jnp.sum(jnp.where(active, state_terms + command_terms, 0.0)) + terminal_term

        flat_result = jax.vmap(score_one_with_terminal)(
            flat_x,
            flat_command,
            flat_vector,
            flat_target,
            flat_starts,
        )
        return flat_result.reshape(leading_shape)

    def _movement_start(self, trial_specs: "TaskTrialSpec") -> Array | None:
        bounds = trial_specs.timeline.epoch_bounds
        if bounds is None:
            return None
        bounds = jnp.asarray(bounds)
        if bounds.shape[-1] < 3:
            return None
        return bounds[..., -2]

    def _time_window(self, values: Array, start: Array, length: int) -> Array:
        start = jnp.asarray(start, dtype=jnp.int32)
        if start.ndim == 0:
            return jax.lax.dynamic_slice_in_dim(values, start, int(length), axis=-2)
        flat_values = values.reshape((-1, values.shape[-2], values.shape[-1]))
        flat_start = start.reshape((-1,))

        def take_one(value, start_one):
            return jax.lax.dynamic_slice_in_dim(value, start_one, int(length), axis=0)

        sliced = jax.vmap(take_one)(flat_values, flat_start)
        return sliced.reshape((*start.shape, int(length), values.shape[-1]))

    def _time_index(self, values: Array, index: Array) -> Array:
        start = jnp.asarray(index, dtype=jnp.int32)
        if start.ndim == 0:
            return jax.lax.dynamic_index_in_dim(values, start, axis=-2, keepdims=False)
        flat_values = values.reshape((-1, values.shape[-2], values.shape[-1]))
        flat_start = start.reshape((-1,))

        def take_one(value, start_one):
            return jax.lax.dynamic_index_in_dim(value, start_one, axis=0, keepdims=False)

        sliced = jax.vmap(take_one)(flat_values, flat_start)
        return sliced.reshape((*start.shape, values.shape[-1]))

    def _initial_vector(self, trial_specs: "TaskTrialSpec", vector: Array) -> Array:
        if "mechanics.vector" not in trial_specs.inits:
            raise ValueError(
                "Full analytical Q/R/Q_f loss requires trial_specs.inits['mechanics.vector'] "
                "from the cs_lss task adapter."
            )
        initial = jnp.asarray(trial_specs.inits["mechanics.vector"], dtype=vector.dtype)
        if initial.shape[-1] != vector.shape[-1]:
            raise ValueError(
                f"Initial mechanics.vector shape {initial.shape} is incompatible with "
                f"rollout state shape {vector.shape}."
            )
        return jnp.broadcast_to(initial, (*vector.shape[:-2], vector.shape[-1]))

    def _target_pos_for_trial(self, trial_specs: "TaskTrialSpec", vector: Array) -> Array:
        target_spec = trial_specs.targets.get("mechanics.effector.pos", None)
        if target_spec is None or not hasattr(target_spec, "value"):
            return jnp.broadcast_to(self.target_pos, (*vector.shape[:-2], 2))
        target_value = jnp.asarray(target_spec.value, dtype=vector.dtype)
        target_pos = target_value[..., -1, :]
        return jnp.broadcast_to(target_pos, (*vector.shape[:-2], 2))

    def _goal_centered(self, vector: Array, target_pos: Array) -> Array:
        result = vector
        target = jnp.asarray(target_pos, dtype=vector.dtype)
        while target.ndim < result.ndim:
            target = jnp.expand_dims(target, axis=-2)
        for start in range(0, self.Q.shape[1], self.n_phys):
            result = result.at[..., start : start + 2].add(-target)
        return result


class DelayedReachTrialTypeNormalizedLoss(AbstractLoss):
    """Trial-type-normalized wrapper for delayed no-catch/catch objectives.

    This is an RLRMP-side bridge while Feedbax grouped reductions are still
    tracked on Mandible issue 69d8d76.
    """

    label: str
    base_loss: AbstractLoss
    trial_type: str = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        base_loss: AbstractLoss,
        trial_type: str,
        label: str,
        eps: float = 1e-8,
    ):
        if trial_type not in {"no_catch", "catch"}:
            raise ValueError("trial_type must be 'no_catch' or 'catch'")
        self.label = label
        self.base_loss = base_loss
        self.trial_type = trial_type
        self.eps = float(eps)

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert trial_specs is not None, (
            "DelayedReachTrialTypeNormalizedLoss requires trial_specs, but trial_specs is None"
        )
        values = self.base_loss.term(states, trial_specs, model)
        catch_mask = self._catch_mask(trial_specs, values)
        mask = catch_mask if self.trial_type == "catch" else 1.0 - catch_mask
        denom = jnp.maximum(jnp.sum(mask), self.eps)
        scale = mask.size / denom
        return values * mask * scale

    def _catch_mask(self, trial_specs: "TaskTrialSpec", values: Array) -> Array:
        if trial_specs.extra is not None and "is_catch_trial" in trial_specs.extra:
            catch = jnp.asarray(trial_specs.extra["is_catch_trial"], dtype=values.dtype)
            while catch.ndim < values.ndim:
                catch = jnp.expand_dims(catch, axis=0)
            return jnp.broadcast_to(catch, values.shape)

        inputs = trial_specs.inputs
        if isinstance(inputs, Mapping) and "task" in inputs:
            inputs = inputs["task"]
        if not hasattr(inputs, "hold"):
            raise ValueError(
                "DelayedReachTrialTypeNormalizedLoss requires "
                "trial_specs.extra['is_catch_trial'] or delayed reach inputs with a hold sequence."
            )
        hold = jnp.asarray(inputs.hold)
        if hold.ndim > 0 and hold.shape[-1] == 1:
            hold = jnp.squeeze(hold, axis=-1)
        catch = (
            jnp.all(hold > 0.5).astype(values.dtype)
            if hold.ndim == 1
            else jnp.all(hold > 0.5, axis=-1).astype(values.dtype)
        )
        while catch.ndim < values.ndim:
            catch = jnp.expand_dims(catch, axis=0)
        return jnp.broadcast_to(catch, values.shape)


class CsForceFilterStateLoss(AbstractLoss):
    """Squared C&S force/filter state over every delay block."""

    label: str
    n_phys: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        label: str = "mechanics_force_filter",
        n_phys: int = 8,
    ):
        self.label = label
        self.n_phys = int(n_phys)

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        del trial_specs, model
        assert states is not None, "CsForceFilterStateLoss requires states, but states is None"
        if not hasattr(states, "mechanics") or not hasattr(states.mechanics, "vector"):
            raise ValueError(
                "Force/filter ablation loss requires states.mechanics.vector from the "
                "cs_lss LinearStateSpace backend."
            )
        vector = jnp.asarray(states.mechanics.vector)
        if vector.shape[-1] % self.n_phys != 0:
            raise ValueError(
                f"state dimension {vector.shape[-1]} is not divisible by n_phys={self.n_phys}."
            )
        blocks = vector.reshape(vector.shape[:-1] + (-1, self.n_phys))
        force_filter = blocks[..., 4:6]
        return jnp.sum(force_filter**2, axis=(-3, -2, -1))


def _build_epoch_mask(
    trial_specs: "TaskTrialSpec", T: int, epoch_indices: tuple[int, ...]
) -> Array:
    """Return a per-trial boolean mask over ``T`` timestep samples."""

    timeline = trial_specs.timeline
    if timeline.epoch_bounds is None:
        raise ValueError("Prep-only loss requires trial_specs.timeline.epoch_bounds.")
    bounds = jnp.asarray(timeline.epoch_bounds)
    if bounds.ndim == 1:
        bounds = bounds[None, :]
    t = jnp.arange(T, dtype=bounds.dtype)
    mask = jnp.zeros((bounds.shape[0], T), dtype=jnp.bool_)
    for epoch in epoch_indices:
        start = bounds[:, epoch : epoch + 1]
        end = bounds[:, epoch + 1 : epoch + 2]
        mask = mask | ((t[None, :] >= start) & (t[None, :] < end))
    return mask


def _sum_masked_time_density(
    density: Array,
    trial_specs: "TaskTrialSpec",
    epoch_indices: tuple[int, ...],
) -> Array:
    """Sum per-trial/per-time loss density over selected epoch samples."""

    mask = _build_epoch_mask(trial_specs, int(density.shape[-1]) + 1, epoch_indices)
    mask = mask[:, 1:].astype(density.dtype)
    mask_shape = [1] * density.ndim
    mask_shape[0] = mask.shape[0]
    mask_shape[-1] = mask.shape[1]
    return jnp.sum(density * mask.reshape(mask_shape), axis=-1)


class InitialEffectorPositionHoldLoss(AbstractLoss):
    """Prep-epoch distance from the trial's initial effector position."""

    label: str
    norm: str = eqx.field(static=True)
    epoch_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        label: str = "delayed_pre_go_start_pos_hold",
        norm: str = "l2",
        epoch_indices: tuple[int, ...] = (0,),
    ):
        if norm not in {"l2", "l1"}:
            raise ValueError(f"Unknown start-position hold norm {norm!r}; expected 'l2' or 'l1'.")
        self.label = label
        self.norm = norm
        self.epoch_indices = tuple(epoch_indices)

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        del model
        assert states is not None, "InitialEffectorPositionHoldLoss requires states."
        assert trial_specs is not None, "InitialEffectorPositionHoldLoss requires trial_specs."
        if "mechanics.vector" not in trial_specs.inits:
            raise ValueError(
                "Initial position hold requires trial_specs.inits['mechanics.vector']."
            )
        pos = jnp.asarray(states.mechanics.effector.pos)
        initial = jnp.asarray(trial_specs.inits["mechanics.vector"], dtype=pos.dtype)[..., :2]
        initial = jnp.broadcast_to(initial, (*pos.shape[:-2], pos.shape[-1]))
        delta = pos[..., 1:, :] - initial[..., None, :]
        if self.norm == "l1":
            density = jnp.sum(jnp.abs(delta), axis=-1)
        else:
            density = jnp.sum(delta**2, axis=-1)
        return _sum_masked_time_density(density, trial_specs, self.epoch_indices)


class PrepZeroVelocityHoldLoss(AbstractLoss):
    """Prep-epoch squared effector velocity."""

    label: str
    epoch_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        label: str = "delayed_pre_go_zero_vel_hold",
        epoch_indices: tuple[int, ...] = (0,),
    ):
        self.label = label
        self.epoch_indices = tuple(epoch_indices)

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        del model
        assert states is not None, "PrepZeroVelocityHoldLoss requires states."
        assert trial_specs is not None, "PrepZeroVelocityHoldLoss requires trial_specs."
        vel = jnp.asarray(states.mechanics.effector.vel)
        density = jnp.sum(vel[..., 1:, :] ** 2, axis=-1)
        return _sum_masked_time_density(density, trial_specs, self.epoch_indices)


class PrepForceFilterHoldLoss(AbstractLoss):
    """Prep-epoch squared C&S force/filter state over every delay block."""

    label: str
    n_phys: int = eqx.field(static=True)
    epoch_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        label: str = "delayed_pre_go_force_filter_hold",
        n_phys: int = 8,
        epoch_indices: tuple[int, ...] = (0,),
    ):
        self.label = label
        self.n_phys = int(n_phys)
        self.epoch_indices = tuple(epoch_indices)

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        del model
        assert states is not None, "PrepForceFilterHoldLoss requires states."
        assert trial_specs is not None, "PrepForceFilterHoldLoss requires trial_specs."
        vector = jnp.asarray(states.mechanics.vector)
        if vector.shape[-1] % self.n_phys != 0:
            raise ValueError(
                f"state dimension {vector.shape[-1]} is not divisible by n_phys={self.n_phys}."
            )
        blocks = vector[..., 1:, :].reshape(
            vector.shape[:-2] + (vector.shape[-2] - 1, -1, self.n_phys)
        )
        force_filter = blocks[..., 4:6]
        density = jnp.sum(force_filter**2, axis=(-2, -1))
        return _sum_masked_time_density(density, trial_specs, self.epoch_indices)


_DEFAULT_LOSS_PRESET = load_loss_preset()
DEFAULT_TOP_WEIGHTS: dict[str, float] = _DEFAULT_LOSS_PRESET.top_weights.model_dump()
DEFAULT_GOAL_HIT_SUBWEIGHTS: dict[str, float] = (
    _DEFAULT_LOSS_PRESET.goal_hit_subweights.model_dump()
)
DEFAULT_GOAL_HIT_PARAMS: dict[str, Any] = _DEFAULT_LOSS_PRESET.goal_hit_params.model_dump()
DEFAULT_EFFECTOR_POS_LATE_PARAMS: dict[str, Any] = (
    _DEFAULT_LOSS_PRESET.effector_pos_late_params.model_dump()
)
DEFAULT_EFFECTOR_VEL_LATE_PARAMS: dict[str, Any] = (
    _DEFAULT_LOSS_PRESET.effector_vel_late_params.model_dump()
)
DEFAULT_EFFECTOR_POS_MID_PARAMS: dict[str, Any] = (
    _DEFAULT_LOSS_PRESET.effector_pos_mid_params.model_dump()
)
DEFAULT_EFFECTOR_VEL_MID_PARAMS: dict[str, Any] = (
    _DEFAULT_LOSS_PRESET.effector_vel_mid_params.model_dump()
)


def get_epoch_weights(
    trial_spec: TaskTrialSpec,
    *,
    end_epoch: Optional[int] = None,
    start_epoch: Optional[int] = None,
    start_offset: int = 0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """
    Returns a (T,) float weights vector that's 1.0 for t in [start, end) and 0.0 elsewhere.
    Works under jit even when `start`/`end` come from traced epoch_bounds.
    """
    bounds = trial_spec.timeline.epoch_bounds
    if bounds is None:
        raise ValueError("Trial spec supplies no epoch_bounds")

    n_steps = trial_spec.timeline.n_steps
    if n_steps is None:
        raise ValueError("Trial spec does not specify n_steps")
    n_steps -= 1

    start_idx = 0 if start_epoch is None else start_epoch
    end_idx = -1 if end_epoch is None else end_epoch

    # Build a fixed-length index vector once (needs T known at trace time, which is typical)
    t = jnp.arange(n_steps, dtype=jnp.int32)

    mask = (t >= bounds[start_idx] + start_offset) & (t < bounds[end_idx])
    return mask.astype(dtype)


during_hold = TargetSpec(time_mask=partial(get_epoch_weights, end_epoch=-2))
during_movement = TargetSpec(time_mask=partial(get_epoch_weights, start_epoch=-2))


def _add_delayed_pre_go_auxiliary_terms(
    terms: dict[str, AbstractLoss],
    weights: dict[str, float],
    user_loss_config: Any,
    user_outer_weights: Any,
    *,
    epoch_indices: tuple[int, ...],
    n_phys: int,
) -> None:
    """Append nonzero delayed-reach prep-only auxiliary terms."""

    start_pos_norm = str(
        _nsget(user_loss_config, "delayed_pre_go_start_pos_hold_norm", "l2") or "l2"
    )
    term_builders: dict[str, Callable[[], AbstractLoss]] = {
        "delayed_pre_go_force_filter_hold": lambda: PrepForceFilterHoldLoss(
            n_phys=n_phys,
            epoch_indices=epoch_indices,
        ),
        "delayed_pre_go_start_pos_hold": lambda: InitialEffectorPositionHoldLoss(
            norm=start_pos_norm,
            epoch_indices=epoch_indices,
        ),
        "delayed_pre_go_zero_vel_hold": lambda: PrepZeroVelocityHoldLoss(
            epoch_indices=epoch_indices,
        ),
    }
    for name, builder in term_builders.items():
        weight = _as_number(_nsget(user_outer_weights, name, None))
        if weight is not None and weight != 0.0:
            terms[name] = builder()
            weights[name] = float(weight)


def make_late_discount_from_epoch(
    offset_steps: int, max_factor: float = 3.0, smooth: str = "cosine", start_epoch: int = -2
):
    """Return a callable(spec_i)->(T,) that ramps from 1.0 to max_factor starting at
    (epoch -2 start) + offset_steps, over the remainder of the trial.

    smooth: "linear" or "cosine" (cosine has C1 continuity at endpoints).
    """

    def discount(spec):
        # T and epoch bounds are available on the per-trial spec
        T = spec.timeline.n_steps - 1
        t = jnp.arange(T, dtype=jnp.float32)

        # 1. Get the mask for epoch -2 (1 from start of epoch -2 to its end)
        epoch_m = get_epoch_weights(
            spec, start_epoch=start_epoch, end_epoch=None, dtype=jnp.float32
        )  # (T,)

        # 2. Start index: first 1 in that mask, then add the late offset
        start = jnp.argmax(epoch_m > 0).astype(jnp.int32) + jnp.int32(offset_steps)

        # 3. Fraction along the late region [start, T)
        denom = jnp.maximum(1.0, (jnp.asarray(T, jnp.float32) - start.astype(jnp.float32)))
        frac = jnp.clip((t - start.astype(jnp.float32)) / denom, 0.0, 1.0)

        # 4. Apply smoothing if desired
        if smooth == "cosine":
            frac = 0.5 - 0.5 * jnp.cos(jnp.pi * frac)  # 0 → 0, 1 → 1, smooth
        # elif smooth == "linear": keep as-is

        return 1.0 + (max_factor - 1.0) * frac  # (T,)

    return discount


def make_power_law_schedule(power: float = 6.0, normalization: str = "trial_end"):
    """Return a callable ``(spec_i) -> (T,) array`` computing ``(t / N) ** power``.

    The schedule assigns increasing weight to later timesteps, concentrating the
    loss signal near the end of the trial.  ``power=6`` matches C&S 2019 Eq. 15.

    Args:
        power: Exponent of the power law.  ``power=1`` is linear; ``power=6``
            puts ~98 % of the weight in the last 30 % of the trial.  Default 6.0.
        normalization: Denominator for the time fraction:
            - ``"trial_end"`` (default): divide by ``T - 1`` (the index of the
              final timestep), so the schedule runs exactly from ``0`` at
              ``t = 0`` to ``1`` at ``t = T - 1``.  Matches C&S Eq. 15.
            - ``"epoch_end"``: divide by the last timestep of the movement
              epoch (epoch -1 start, i.e. the final epoch boundary).  Less
              common; produces a scale that peaks earlier for long post-target
              hold periods.

    Returns:
        Callable ``(spec_i) -> (T,) float32 array`` shaped to the trial length.
    """

    def schedule(spec):
        T = spec.timeline.n_steps - 1
        t = jnp.arange(T, dtype=jnp.float32)

        if normalization == "trial_end":
            N = jnp.asarray(T - 1, jnp.float32)
        elif normalization == "epoch_end":
            # Use the start of the final epoch as the normalisation denominator.
            bounds = spec.timeline.epoch_bounds
            N = jnp.asarray(bounds[-1], jnp.float32) - 1.0
        else:
            raise ValueError(
                f"normalization must be 'trial_end' or 'epoch_end', got {normalization!r}"
            )

        N = jnp.maximum(N, 1.0)  # avoid divide-by-zero for very short trials
        return (t / N) ** power  # (T,)

    return schedule


def make_fixed_power_law_schedule(n_steps: int, power: float = 6.0) -> Array:
    """Return a transition-time power law for tasks without epoch metadata."""

    T = max(int(n_steps) - 1, 1)
    t = jnp.arange(T, dtype=jnp.float32)
    normalizer = jnp.maximum(jnp.asarray(T - 1, dtype=jnp.float32), 1.0)
    return (t / normalizer) ** power


def make_cs_eq15_stage_schedule(n_steps: int, power: float = 6.0) -> Array:
    """Return C&S Eq. 15 stage weights ``((t + 1) / T) ** power``.

    ``n_steps`` is the Feedbax rollout length, so the number of control/cost
    stages is ``n_steps - 1``.
    """

    T = max(int(n_steps) - 1, 1)
    t_plus_1 = jnp.arange(1, T + 1, dtype=jnp.float32)
    return jnp.minimum(1.0, (t_plus_1 / float(T)) ** power)


def make_movement_cs_eq15_stage_schedule(
    *,
    horizon_steps: int,
    start_epoch: int = -2,
    power: float = 6.0,
):
    """Return C&S Eq. 15 weights indexed by movement age, not trial age."""

    if horizon_steps <= 0:
        raise ValueError(f"horizon_steps must be positive, got {horizon_steps}")

    def schedule(spec):
        T = spec.timeline.n_steps
        bounds = spec.timeline.epoch_bounds
        if bounds is None:
            raise ValueError("Movement-epoch C&S schedule requires epoch bounds")
        t = jnp.arange(T, dtype=jnp.float32)
        start = jnp.asarray(bounds[..., start_epoch], dtype=jnp.float32)
        if start.ndim > 0:
            start = jnp.expand_dims(start, axis=-1)
        age = t - start
        weights = ((age + 1.0) / float(horizon_steps)) ** power
        weights = jnp.minimum(1.0, weights)
        return jnp.where(age >= 0.0, weights, 0.0).astype(jnp.float32)

    return schedule


def make_epoch_locked_ramp(
    *,
    duration_steps: int,
    start_epoch: int = -2,
    shape: str = "linear",
    power: float = 2.0,
):
    """Return a fixed-duration ramp starting at an epoch boundary.

    The returned weights are zero before the selected epoch starts, ramp from
    zero to one over ``duration_steps``, and remain one after the ramp completes.
    This is useful for movement-locked position costs because it does not leak
    position-error weight into the target-on/pre-go period.
    """
    if duration_steps <= 0:
        raise ValueError(f"duration_steps must be positive, got {duration_steps}")
    if shape not in {"linear", "cosine", "power"}:
        raise ValueError(f"shape must be 'linear', 'cosine', or 'power', got {shape!r}")

    def schedule(spec):
        T = spec.timeline.n_steps - 1
        bounds = spec.timeline.epoch_bounds
        if bounds is None:
            raise ValueError("Trial spec supplies no epoch_bounds")

        t = jnp.arange(T, dtype=jnp.float32)
        start = jnp.asarray(bounds[..., start_epoch], dtype=jnp.float32)
        if start.ndim > 0:
            start = jnp.expand_dims(start, axis=-1)
        frac = jnp.clip((t - start) / float(duration_steps), 0.0, 1.0)

        if shape == "cosine":
            frac = 0.5 - 0.5 * jnp.cos(jnp.pi * frac)
        elif shape == "power":
            frac = frac**power

        return frac.astype(jnp.float32)

    return schedule


def make_mid_period_ramp(
    start_step: int,
    end_step: int,
    init_weight: float,
    final_weight: float,
    start_epoch: int = -2,
):
    """Return a callable(spec_i)->(T,) that linearly ramps from init_weight to final_weight
    during a window relative to the start of the specified epoch.

    The returned discount is:
    - 0.0 before the window
    - linearly interpolates from init_weight to final_weight during the window
    - 0.0 after the window

    Args:
        start_step: Window start offset relative to epoch start (e.g., 0)
        end_step: Window end offset relative to epoch start (e.g., 80)
        init_weight: Weight at window start
        final_weight: Weight at window end
        start_epoch: Which epoch to offset from (default -2, the go cue)
    """

    def discount(spec):
        T = spec.timeline.n_steps - 1
        t = jnp.arange(T, dtype=jnp.float32)

        # Get the mask for the specified epoch
        epoch_m = get_epoch_weights(
            spec, start_epoch=start_epoch, end_epoch=None, dtype=jnp.float32
        )

        # Epoch start index
        epoch_start = jnp.argmax(epoch_m > 0).astype(jnp.int32)

        # Window boundaries in absolute time
        window_start = epoch_start + jnp.int32(start_step)
        window_end = epoch_start + jnp.int32(end_step)

        # Fraction through the window [0, 1]
        window_length = jnp.maximum(1.0, (window_end - window_start).astype(jnp.float32))
        frac = (t - window_start.astype(jnp.float32)) / window_length
        frac = jnp.clip(frac, 0.0, 1.0)

        # Linear interpolation from init_weight to final_weight
        ramp_value = init_weight + (final_weight - init_weight) * frac

        # Mask: only apply during the window
        in_window = (t >= window_start) & (t < window_end)

        return jnp.where(in_window, ramp_value, 0.0)

    return discount


def get_reach_loss(hps: TreeNamespace) -> CompositeLoss:
    """Construct a loss function for reaching task.

    Some terms are always included (e.g. penalty on neural network activity and output) while others
    depend on which mode is active based on weight configuration.

    Three modes are available:
    1. **Simple mode** (`effector_pos` weight > 0): Basic position error during movement + final
       velocity penalty. No mid/late terms.
    2. **Goal hit mode** (`goal_hit_in_window` weight > 0): Soft hit-window objective with optional
       mid/late terms.
    3. **Structured mode** (both above weights = 0): Fully configurable mid/late period terms.

    Example YAML config for structured mode with mid/late-period ramping:

      loss:
        weights:
          goal_hit_in_window: 0.0
          effector_pos: 0.0
          effector_pos_mid: 1.0
          effector_vel_mid: 1.0
          effector_pos_late: 1.0
          effector_vel_late: 1.0
          effector_hold_pos: 10.0
          effector_hold_vel: 10.0
          nn_output: 1e-6
          nn_hidden: 1e-6
        effector_pos_late:
          start_step_after_go: 80
          final_scale_factor: 1.0
        effector_vel_late:
          start_step_after_go: 80
          final_scale_factor: 1.0
        effector_pos_mid:
          start_step_after_go: 0
          end_step_after_go: 80
          ramp_init_weight: 0.0
          ramp_final_weight: 0.1
        effector_vel_mid:
          start_step_after_go: 0
          end_step_after_go: 80
          ramp_init_weight: 0.0
          ramp_final_weight: 0.1
        goal_hit_in_window:
          start_step_after_go: 60
          end_step_after_go: 80
          softmin_tau: 0.2
          post_pos_sigma_t: 5
          weights:
            pos: 1.0
            vel: 0.1
            post_pos: 1.0

    The mid-period terms (`effector_pos_mid`, `effector_vel_mid`) provide linear ramping penalties
    during a specified window after the go cue (active in goal-hit and structured modes only).

    The late-period terms (`effector_pos_late`, `effector_vel_late`) provide constant or increasing
    penalties starting at a specified time after the go cue, continuing until trial end (active in
    goal-hit and structured modes only, disabled in simple mode).

    TODO: Refactor all the defaults/merging logic into a preparatory function.
    """
    loss_objective = str(
        _nsget(hps, "loss.objective", CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE)
        or CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE
    )
    user_outer_weights = _nsget(hps, "loss.weights", {}) or {}
    task_type = str(getattr(hps.task, "type", ""))
    is_feedbax_delayed_center_out = (
        task_type == "cs_delayed_center_out_reach"
        or bool(_nsget(hps, "delayed_reach.enabled", False))
        or (
            task_type == "delayed_reach"
            and str(_nsget(hps, "task.preset", "")) == "delayed_center_out"
        )
    )
    pre_go_epoch_indices = (0,) if is_feedbax_delayed_center_out else (0, 1)

    if loss_objective == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE:
        plant_backend = str(_nsget(hps, "model.plant_backend", ""))
        if plant_backend != "cs_lss":
            raise ValueError(
                "loss.objective='full_analytical_qrf' requires model.plant_backend='cs_lss' "
                f"so the full 48D C&S state is available; got {plant_backend!r}."
            )
        no_integrator_state = bool(_nsget(hps, "model.no_integrator_state", False))
        _plant, schedule = (
            build_no_integrator_game() if no_integrator_state else build_canonical_game()
        )
        term = CsAnalyticalQrfLoss(
            Q=schedule.Q,
            R=schedule.R,
            Q_f=schedule.Q_f,
            target_pos=TARGET_POS,
            n_phys=6 if no_integrator_state else 8,
            delayed_movement_cost_tail_mode=str(
                _nsget(hps, "loss.delayed_movement_cost_tail_mode", "canonical_window")
                or "canonical_window"
            ),
        )
        trial_type_normalization = bool(
            _nsget(hps, "loss.delayed_trial_type_normalization.enabled", False)
        )
        if trial_type_normalization:
            terms: dict[str, AbstractLoss] = {
                f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_no_catch": (
                    DelayedReachTrialTypeNormalizedLoss(
                        base_loss=term,
                        trial_type="no_catch",
                        label=f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_no_catch",
                    )
                ),
                f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_catch": (
                    DelayedReachTrialTypeNormalizedLoss(
                        base_loss=term,
                        trial_type="catch",
                        label=f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_catch",
                    )
                ),
            }
            weights = {
                f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_no_catch": float(
                    _nsget(hps, "loss.delayed_trial_type_normalization.no_catch_weight", 1.0)
                ),
                f"{CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE}_catch": float(
                    _nsget(hps, "loss.delayed_trial_type_normalization.catch_weight", 1.0)
                ),
            }
        else:
            terms = {CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE: term}
            weights = {CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE: 1.0}
        nn_output_pre_go_weight = _as_number(_nsget(user_outer_weights, "nn_output_pre_go", None))
        if nn_output_pre_go_weight is not None and nn_output_pre_go_weight != 0.0:
            terms["nn_output_pre_go"] = EpochMaskedLoss(
                label="nn_output_pre_go",
                base_loss=TargetStateLoss(
                    label="nn_output_pre_go",
                    where=lambda state: state.efferent.output,
                    spec=target_zero,
                ),
                epoch_indices=pre_go_epoch_indices,
            )
            weights["nn_output_pre_go"] = float(nn_output_pre_go_weight)
        _add_delayed_pre_go_auxiliary_terms(
            terms,
            weights,
            _nsget(hps, "loss", None),
            user_outer_weights,
            epoch_indices=pre_go_epoch_indices,
            n_phys=6 if no_integrator_state else 8,
        )
        return CompositeLoss(
            label="reach_loss",
            terms=terms,
            weights=weights,
        )
    if loss_objective not in {
        CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
        CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    }:
        raise ValueError(
            f"Unknown loss.objective {loss_objective!r}; expected one of {CS_LOSS_OBJECTIVES}."
        )
    ablate_net_force_filter = loss_objective == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE
    if ablate_net_force_filter:
        plant_backend = str(_nsget(hps, "model.plant_backend", ""))
        if plant_backend != "cs_lss":
            raise ValueError(
                "loss.objective='partial_net_output_force_filter' requires "
                f"model.plant_backend='cs_lss'; got {plant_backend!r}."
            )

    terms: Mapping[str, AbstractLoss] = dict(
        nn_output=TargetStateLoss(
            "nn_output",
            where=(
                (lambda state: state.net.output)
                if ablate_net_force_filter
                else (lambda state: state.efferent.output)
            ),
            spec=target_zero,
        ),
        nn_hidden=TargetStateLoss(
            "nn_hidden",
            where=lambda state: state.net.hidden,
            spec=target_zero,
        ),
        # Hidden-state smoothness regulariser (Shahbazi et al. 2025 Eq. 1).
        # mean(||h_t - h_{t-1}||²) over rollout time axis; default weight 0
        # leaves baseline behaviour unchanged unless explicitly enabled in
        # `loss.weights.nn_hidden_derivative`. Bug: efc4d68
        nn_hidden_derivative=StateDerivativeLoss(
            label="nn_hidden_derivative",
            where=lambda state: state.net.hidden,
        ),
        # Output-jerk regulariser (Shahbazi et al. 2025 Eq. 1, weight 1e5 in
        # their setup). Discrete second-difference of effector velocity =
        # discrete jerk. Default weight 0 leaves baseline behaviour unchanged
        # unless explicitly enabled in `loss.weights.nn_output_jerk`.
        # Bug: efc4d68 (feedbax 7e1d257)
        nn_output_jerk=OutputJerkLoss(
            label="nn_output_jerk",
            where=lambda state: state.mechanics.effector.vel,
        ),
        # Anti-anticipation pre-go controller-output penalty. Wraps the
        # standard `nn_output` term (squared L2 of the controller force) in
        # an `EpochMaskedLoss` keyed to epochs 0+1 of the standard
        # `DelayedReaches` timeline ("hold" + "target_on"). The mask is 1
        # before the go cue and 0 after, so any non-zero contribution is
        # purely a pre-go anticipatory force. Default 0.0; activate via
        # `loss.weights.nn_output_pre_go` or `--nn-output-pre-go <w>` on
        # `train_minimax.py`. Bug: efc4d68 (feedbax 50507a9)
        nn_output_pre_go=EpochMaskedLoss(
            label="nn_output_pre_go",
            base_loss=TargetStateLoss(
                label="nn_output_pre_go",
                where=lambda state: state.efferent.output,
                spec=target_zero,
            ),
            epoch_indices=pre_go_epoch_indices,
        ),
        # Anti-preparation companion: same epoch mask wrapped around the
        # hidden-state first-difference term. Exposed so the comparator run
        # is a single CLI flag away. Default 0.0. Bug: efc4d68 (feedbax 50507a9)
        nn_hidden_derivative_pre_go=EpochMaskedLoss(
            label="nn_hidden_derivative_pre_go",
            base_loss=StateDerivativeLoss(
                label="nn_hidden_derivative_pre_go",
                where=lambda state: state.net.hidden,
            ),
            epoch_indices=pre_go_epoch_indices,
        ),
    )
    if ablate_net_force_filter:
        no_integrator_state = bool(_nsget(hps, "model.no_integrator_state", False))
        terms["mechanics_force_filter"] = CsForceFilterStateLoss(
            n_phys=6 if no_integrator_state else 8,
        )
    _add_delayed_pre_go_auxiliary_terms(
        terms,
        {},
        _nsget(hps, "loss", None),
        user_outer_weights,
        epoch_indices=pre_go_epoch_indices,
        n_phys=6 if bool(_nsget(hps, "model.no_integrator_state", False)) else 8,
    )

    # Power-law schedule parameters (Bug: 2e1a6ad).
    # "flat" (default) keeps existing uniform weighting; "powerlaw" applies
    # (t / (T-1))^power to each term's timestep weights.
    _pos_running_sched = _nsget(hps, "loss.effector_pos_running_schedule", "flat") or "flat"
    _hold_pos_sched = _nsget(hps, "loss.effector_hold_pos_schedule", "flat") or "flat"
    _powerlaw_power = float(_nsget(hps, "loss.position_powerlaw_power", 6.0) or 6.0)
    _movement_ramp_shape = _nsget(hps, "loss.movement_ramp_shape", "linear") or "linear"
    _movement_ramp_duration = int(_nsget(hps, "loss.movement_ramp_duration_steps", 60) or 60)
    _movement_ramp_power = float(_nsget(hps, "loss.movement_ramp_power", 2.0) or 2.0)
    is_simple_reach = task_type in {"simple_reach", "fixed_simple_reach"}
    is_cs_delayed_reach = is_feedbax_delayed_center_out
    is_transition_aligned_reach = is_simple_reach or is_cs_delayed_reach

    # "center_out_delayed_reach" is a subclass of DelayedReaches and shares the
    # same hold-period structure — match on suffix. Bug: 2e1a6ad.
    if "delayed_reach" in task_type or is_cs_delayed_reach:
        if _hold_pos_sched == "powerlaw":
            # (t/T-1)^power applied over the entire trial; the epoch mask still
            # comes from `during_hold` (time_mask), but the weight rises with t.
            # The hold epoch is early in the trial so the power-law weight is
            # very small there (<<1), placing essentially all the position-hold
            # emphasis on the late-trial timesteps — matching C&S 2019 Eq. 15.
            hold_pos_spec = during_hold & TargetSpec(
                discount=make_power_law_schedule(power=_powerlaw_power)
            )
        else:
            hold_pos_spec = during_hold

        terms["effector_hold_pos"] = TargetStateLoss(
            "effector_hold_pos",
            where=lambda state: state.mechanics.effector.pos,
            # norm=lambda x: jnp.sum(x**2, axis=-1),
            spec=hold_pos_spec,
        )
        terms["effector_hold_vel"] = TargetStateLoss(
            "effector_hold_vel",
            where=lambda state: state.mechanics.effector.vel,
            # norm=lambda x: jnp.sum(x**2, axis=-1),
            spec=target_zero & during_hold,
        )

    # if getattr(hps.loss, "stop_at_goal", False):
    #     terms["stop_at_goal"] = StopAtGoalLoss(**hps.loss.stop_at_goal)

    fix_readout_cfg = _nsget(hps, "loss.fix_readout_norm", None)
    if fix_readout_cfg:
        terms["fix_readout_norm"] = get_readout_norm_loss(**fix_readout_cfg)

    # Determine which mode we're in based on weights
    use_goal_hit = getattr(user_outer_weights, "goal_hit_in_window", 0.0) != 0.0
    use_simple_effector_pos = getattr(user_outer_weights, "effector_pos", 0.0) != 0.0

    # Read configs for late-period terms (used in goal_hit and structured modes)
    effector_pos_late_cfg = _nsget(hps, "loss.effector_pos_late", None)
    effector_pos_late_params = deep_merge(
        DEFAULT_EFFECTOR_POS_LATE_PARAMS, effector_pos_late_cfg or {}
    )
    pos_late_start = int(effector_pos_late_params["start_step_after_go"])
    pos_late_scale_factor = float(effector_pos_late_params["final_scale_factor"])

    effector_vel_late_cfg = _nsget(hps, "loss.effector_vel_late", None)
    effector_vel_late_params = deep_merge(
        DEFAULT_EFFECTOR_VEL_LATE_PARAMS, effector_vel_late_cfg or {}
    )
    vel_late_start = int(effector_vel_late_params["start_step_after_go"])
    vel_late_scale_factor = float(effector_vel_late_params["final_scale_factor"])

    # MODE 1: Simple mode (effector_pos weight > 0)
    # Creates simple effector_pos term + final velocity penalty only
    if use_simple_effector_pos:
        effector_pos_loss_cls = (
            SimpleReachPositionLoss if is_transition_aligned_reach else TargetStateLoss
        )
        effector_pos_spec = None if is_transition_aligned_reach else during_movement
        terms["effector_pos"] = effector_pos_loss_cls(
            "effector_pos",
            where=lambda state: state.mechanics.effector.pos,
            spec=effector_pos_spec,
        )
        # Final velocity penalty (only at final timestep)
        terms["effector_final_vel"] = TargetStateLoss(
            "effector_final_vel",
            where=lambda state: state.mechanics.effector.vel,
            spec=target_zero & target_final_state,
        )

    # MODE 2: Goal hit mode (goal_hit_in_window weight > 0)
    # Creates goal hit term + can have mid/late terms
    elif use_goal_hit:
        goal_hit_cfg = _nsget(hps, "loss.goal_hit_in_window", None)
        if goal_hit_cfg is not None:
            goal_hit_params = deep_merge(DEFAULT_GOAL_HIT_PARAMS, goal_hit_cfg or {})

            window_start = int(goal_hit_params["start_step_after_go"])
            window_end = int(goal_hit_params["end_step_after_go"])
            softmin_tau = float(goal_hit_params["softmin_tau"])
            post_pos_sigma_t = float(goal_hit_params["post_pos_sigma_t"])
            alpha_eps = float(goal_hit_params["alpha_eps"])

            goal_hit_terms = dict(
                pos=goal_hit_pos_loss_term_fn,
                vel=goal_hit_vel_loss_term_fn,
                post_pos=partial(post_hit_pos_loss_term_fn, sigma_t=post_pos_sigma_t),
            )

            # Merge default and user subweights
            goal_hit_subweights = deep_merge(
                DEFAULT_GOAL_HIT_SUBWEIGHTS,
                _nsget(hps, "loss.goal_hit_in_window.weights", {}),
            )

            # Exclude zero-weighted sub-terms
            goal_hit_subweights = _filter_nonzero(goal_hit_subweights)
            goal_hit_terms = {k: v for k, v in goal_hit_terms.items() if k in goal_hit_subweights}

            terms["goal_hit_in_window"] = FuncTermsLoss(
                label="goal_hit_in_window",
                build_context=GoalHitCtxBuilder(
                    window_bounds=(window_start, window_end), tau=softmin_tau, eps=alpha_eps
                ),
                terms=goal_hit_terms,
                weights=goal_hit_subweights,
            )

    # MODE 3: Structured mode (neither effector_pos nor goal_hit_in_window)
    # Can have mid/late terms for both position and velocity

    # Mid-period terms (active in goal_hit or structured modes, NOT in simple mode)
    if not use_simple_effector_pos:
        # Mid-period position ramping (optional)
        effector_pos_mid_cfg = _nsget(hps, "loss.effector_pos_mid", None)
        if effector_pos_mid_cfg is not None:
            effector_pos_mid_params = deep_merge(
                DEFAULT_EFFECTOR_POS_MID_PARAMS, effector_pos_mid_cfg or {}
            )
            pos_mid_start = int(effector_pos_mid_params["start_step_after_go"])
            pos_mid_end = int(effector_pos_mid_params["end_step_after_go"])
            pos_mid_init = float(effector_pos_mid_params["ramp_init_weight"])
            pos_mid_final = float(effector_pos_mid_params["ramp_final_weight"])

            terms["effector_pos_mid"] = TargetStateLoss(
                "effector_pos_mid",
                where=lambda state: state.mechanics.effector.pos,
                spec=TargetSpec(
                    discount=make_mid_period_ramp(
                        start_step=pos_mid_start,
                        end_step=pos_mid_end,
                        init_weight=pos_mid_init,
                        final_weight=pos_mid_final,
                        start_epoch=-2,
                    )
                ),
            )

        # Mid-period velocity ramping (optional)
        effector_vel_mid_cfg = _nsget(hps, "loss.effector_vel_mid", None)
        if effector_vel_mid_cfg is not None:
            effector_vel_mid_params = deep_merge(
                DEFAULT_EFFECTOR_VEL_MID_PARAMS, effector_vel_mid_cfg or {}
            )
            vel_mid_start = int(effector_vel_mid_params["start_step_after_go"])
            vel_mid_end = int(effector_vel_mid_params["end_step_after_go"])
            vel_mid_init = float(effector_vel_mid_params["ramp_init_weight"])
            vel_mid_final = float(effector_vel_mid_params["ramp_final_weight"])

            terms["effector_vel_mid"] = TargetStateLoss(
                "effector_vel_mid",
                where=lambda state: state.mechanics.effector.vel,
                spec=target_zero
                & TargetSpec(
                    discount=make_mid_period_ramp(
                        start_step=vel_mid_start,
                        end_step=vel_mid_end,
                        init_weight=vel_mid_init,
                        final_weight=vel_mid_final,
                        start_epoch=-2,
                    )
                ),
            )

        # Running position cost: penalizes position error during the entire movement period.
        # Unlike mid-period terms (which ramp), this applies a penalty from
        # go cue to trial end, with the time-profile determined by the schedule.
        # Bug: 2e1a6ad
        if getattr(user_outer_weights, "effector_pos_running", 0.0) != 0.0:
            effector_pos_running_loss_cls = (
                SimpleReachPositionLoss if is_transition_aligned_reach else TargetStateLoss
            )
            if _pos_running_sched == "cs_eq15_power6":
                if is_simple_reach:
                    running_spec = TargetSpec(
                        discount=make_cs_eq15_stage_schedule(hps.task.n_steps, power=6.0)
                    )
                elif is_cs_delayed_reach:
                    running_spec = TargetSpec(
                        discount=make_movement_cs_eq15_stage_schedule(
                            horizon_steps=60,
                            power=6.0,
                        )
                    )
                else:
                    running_spec = during_movement & TargetSpec(
                        discount=make_power_law_schedule(power=6.0)
                    )
            elif _pos_running_sched == "powerlaw":
                # Multiply the epoch mask (1 during movement, 0 before) by the
                # power-law discount (rises as (t/T-1)^power over the whole trial).
                if is_simple_reach:
                    running_spec = TargetSpec(
                        discount=make_fixed_power_law_schedule(
                            hps.task.n_steps,
                            power=_powerlaw_power,
                        )
                    )
                else:
                    running_spec = during_movement & TargetSpec(
                        discount=make_power_law_schedule(power=_powerlaw_power)
                    )
            elif _pos_running_sched == "movement_ramp":
                running_spec = TargetSpec(
                    discount=make_epoch_locked_ramp(
                        duration_steps=_movement_ramp_duration,
                        start_epoch=-2,
                        shape=_movement_ramp_shape,
                        power=_movement_ramp_power,
                    )
                )
            else:
                running_spec = TargetSpec() if is_transition_aligned_reach else during_movement

            terms["effector_pos_running"] = effector_pos_running_loss_cls(
                "effector_pos_running",
                where=lambda state: state.mechanics.effector.pos,
                spec=running_spec,
            )

        if getattr(user_outer_weights, "effector_vel_running", 0.0) != 0.0:
            effector_vel_running_loss_cls = (
                SimpleReachPositionLoss if is_transition_aligned_reach else TargetStateLoss
            )
            if _pos_running_sched == "cs_eq15_power6":
                if is_cs_delayed_reach:
                    vel_running_spec = target_zero & TargetSpec(
                        discount=make_movement_cs_eq15_stage_schedule(
                            horizon_steps=60,
                            power=6.0,
                        )
                    )
                else:
                    vel_running_spec = target_zero & TargetSpec(
                        discount=make_cs_eq15_stage_schedule(hps.task.n_steps, power=6.0)
                    )
            elif _pos_running_sched == "powerlaw":
                if is_simple_reach:
                    vel_running_spec = target_zero & TargetSpec(
                        discount=make_fixed_power_law_schedule(
                            hps.task.n_steps,
                            power=_powerlaw_power,
                        )
                    )
                else:
                    vel_running_spec = (
                        target_zero
                        & during_movement
                        & TargetSpec(discount=make_power_law_schedule(power=_powerlaw_power))
                    )
            else:
                vel_running_spec = (
                    target_zero if is_transition_aligned_reach else target_zero & during_movement
                )

            terms["effector_vel_running"] = effector_vel_running_loss_cls(
                "effector_vel_running",
                where=lambda state: state.mechanics.effector.vel,
                spec=vel_running_spec,
            )

        if getattr(user_outer_weights, "effector_terminal_pos", 0.0) != 0.0:
            effector_terminal_pos_loss_cls = (
                SimpleReachPositionLoss if is_transition_aligned_reach else TargetStateLoss
            )
            terms["effector_terminal_pos"] = effector_terminal_pos_loss_cls(
                "effector_terminal_pos",
                where=lambda state: state.mechanics.effector.pos,
                spec=(
                    target_final_state
                    if is_transition_aligned_reach
                    else during_movement & target_final_state
                ),
            )

        if getattr(user_outer_weights, "effector_terminal_vel", 0.0) != 0.0:
            effector_terminal_vel_loss_cls = (
                SimpleReachPositionLoss if is_transition_aligned_reach else TargetStateLoss
            )
            terms["effector_terminal_vel"] = effector_terminal_vel_loss_cls(
                "effector_terminal_vel",
                where=lambda state: state.mechanics.effector.vel,
                spec=target_zero & target_final_state,
            )

        # Late-period position term (optional, active in goal_hit or structured modes)
        if getattr(user_outer_weights, "effector_pos_late", 0.0) != 0.0:
            after_pos_late_window = TargetSpec(
                time_mask=partial(
                    get_epoch_weights,
                    start_epoch=-2,
                    start_offset=pos_late_start,
                )
            )
            terms["effector_pos_late"] = TargetStateLoss(
                "effector_pos_late",
                where=lambda state: state.mechanics.effector.pos,
                spec=(
                    after_pos_late_window
                    & TargetSpec(
                        discount=make_late_discount_from_epoch(
                            pos_late_start,
                            max_factor=pos_late_scale_factor,
                            smooth="cosine",
                            start_epoch=-2,
                        ),
                    )
                ),
            )

        # Late-period velocity term (entire late window, not just final timestep)
        if getattr(user_outer_weights, "effector_vel_late", 0.0) != 0.0:
            after_vel_late_window = TargetSpec(
                time_mask=partial(
                    get_epoch_weights,
                    start_epoch=-2,
                    start_offset=vel_late_start,
                )
            )
            terms["effector_vel_late"] = TargetStateLoss(
                "effector_vel_late",
                where=lambda state: state.mechanics.effector.vel,
                spec=(
                    target_zero
                    & after_vel_late_window
                    & TargetSpec(
                        discount=make_late_discount_from_epoch(
                            vel_late_start,
                            max_factor=vel_late_scale_factor,
                            smooth="cosine",
                            start_epoch=-2,
                        ),
                    )
                ),
            )

        # Terminal-step velocity penalty: fires only at t=T, the final timestep.
        # This is the historical `simple_reach_loss` shape (commit 3eea931,
        # feedbax e985e0e). A surgical "come-to-rest at the goal" signal that
        # acts as an identifying constraint, funnelling replicates toward a
        # single stopping strategy. In contrast, `effector_vel_late` spreads
        # the velocity penalty across a window, which is more permissive.
        # Default 0.0 preserves baseline behaviour; enable via
        # --effector-final-vel (suggested 1.0). Bug: 2bc95fd
        if getattr(user_outer_weights, "effector_final_vel", 0.0) != 0.0:
            terms["effector_final_vel"] = TargetStateLoss(
                "effector_final_vel",
                where=lambda state: state.mechanics.effector.vel,
                spec=target_zero & target_final_state,
            )

    # Defaults only for present terms
    defaults_for_present = _pick(DEFAULT_TOP_WEIGHTS, terms.keys())

    # Overlay user scalars; ignore dicts (e.g., the bundle's internal weights)
    outer_weights: dict[str, float] = {}
    for name in terms.keys():
        cand = _as_number(getattr(user_outer_weights, name, None))
        outer_weights[name] = float(
            cand if cand is not None else defaults_for_present.get(name, 1.0)
        )

    # Exclude zero-weighted terms
    outer_weights = _filter_nonzero(outer_weights)
    terms = {k: v for k, v in terms.items() if k in outer_weights}

    loss_fn = CompositeLoss(label="reach_loss", terms=terms, weights=outer_weights)
    return loss_fn


def get_adaptive_control_penalty_update(
    *,
    target_ratio: float,
    alpha: float,
    control_term: str = "nn_output",
    goal_term: str | list[str] = "effector_pos_late",
) -> Callable[[CompositeLoss, TermTree, PyTree], CompositeLoss]:
    """Create a loss update function that adaptively adjusts control penalty weights.

    Maintains a target ratio between control cost and goal error by updating the
    control term's weight according to:

        ρ_{k+1} = ρ_k * (J_x / (r* · J_u))^α

    where:
        - ρ is the weight on the control term
        - J_x is the goal error (after weighting, summed if multiple terms)
        - J_u is the control cost (after weighting)
        - r* is the target ratio (J_u / J_x)
        - α is the update rate

    Arguments:
        target_ratio: Desired ratio of control cost to goal error (J_u / J_x)
        alpha: Update rate in (0, 1]. Smaller values = slower adaptation.
        control_term: Name of the control cost term in the loss function
        goal_term: Name(s) of the goal error term(s) in the loss function. If a list,
                   all terms will be summed to compute total goal error.

    Returns:
        Update function with signature (loss_func, losses, grads) -> updated_loss_func
    """
    # Normalize goal_term to always be a list
    goal_terms = [goal_term] if isinstance(goal_term, str) else goal_term

    def loss_update_func(
        loss_func: CompositeLoss,
        losses: TermTree,
        grads: PyTree,  # Not used but required by signature
    ) -> CompositeLoss:
        # Extract term values (already weighted and aggregated over trials)
        # For ensembled training, losses[term].total has shape (replicates,)
        # Keep this shape to compute per-replicate weight updates

        # Sum multiple goal terms if provided (maintains replicate dimension)
        J_x = sum(losses[term].total for term in goal_terms if term in losses)
        J_u = losses[control_term].total  # Control cost (shape: (replicates,) or ())

        # Compute multiplicative update per replicate
        # Want: J_u ≈ target_ratio * J_x
        # So if J_u is too small, increase weight; if too large, decrease weight
        current_weight = loss_func.weights[control_term]
        ratio = J_x / (target_ratio * J_u + 1e-12)  # Add epsilon to avoid division by zero
        new_weight = current_weight * (ratio**alpha)

        # Clip to reasonable range to prevent runaway.
        new_weight = jnp.clip(new_weight, 1e-8, 1e-2)

        # Convert to Python float. This forces a device→host sync, but
        # loss_update_iterations is set to run infrequently (every ~100 iters)
        # so the amortized cost is negligible. We MUST use a Python float
        # because vmap stacks JAX array weights across replicas, breaking the
        # scalar weight invariant.
        new_weight = float(new_weight)

        new_weights = loss_func.weights.copy()
        new_weights[control_term] = new_weight

        return loss_func.with_weights(new_weights)

    return loss_update_func


def get_loss_update_func(hps: TreeNamespace):
    """Return loss update function configured from hyperparameters, or None if disabled.

    This is the standard interface that training modules should use to get a loss
    update function from the configuration.

    Arguments:
        hps: Hyperparameters namespace containing `loss_update` configuration

    Returns:
        Tuple of (update_func, start_iteration) where:
            - update_func: Callable[[CompositeLoss, TermTree, PyTree], CompositeLoss] or None
            - start_iteration: int indicating iteration to start applying updates (0 = from beginning)
    """
    loss_update_cfg = getattr(hps, "loss_update", None)
    if loss_update_cfg is None or not getattr(loss_update_cfg, "enabled", False):
        return None, 0  # (func, start_iteration) - 0 is a dummy value when func is None

    # Default goal_term: sum mid, late, and running position penalties for full movement error
    default_goal_term = ["effector_pos_mid", "effector_pos_late", "effector_pos_running"]
    goal_term = getattr(loss_update_cfg, "goal_term", default_goal_term)

    update_func = get_adaptive_control_penalty_update(
        target_ratio=loss_update_cfg.target_ratio,
        alpha=loss_update_cfg.alpha,
        control_term=getattr(loss_update_cfg, "control_term", "nn_output"),
        goal_term=goal_term,
    )

    start_iteration = getattr(loss_update_cfg, "start_iteration", 0)

    return update_func, start_iteration
