import logging
from collections.abc import Mapping
from functools import partial
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from feedbax import AbstractModel
from feedbax.loss import (
    AbstractLoss,
    CompositeLoss,
    FuncTermsLoss,
    StopAtGoalLoss,
    TargetSpec,
    TargetStateLoss,
    target_final_state,
    target_zero,
)
from feedbax.task import TaskTrialSpec
from feedbax.xabdeef.losses import simple_reach_loss

# from feedbax.xabdeef.losses import simple_reach_loss
from feedbax_experiments.training.loss import get_readout_norm_loss
from feedbax_experiments.types import TreeNamespace
from jax_cookbook.misc import window_take
from jaxtyping import Array, PyTree

logger = logging.getLogger(__name__)


def softmin_and_alpha(values, tau, axis=-1, eps=1e-12, keepdims=False):
    # values: (..., T) with T = |W|
    m = jnp.min(values, axis=axis, keepdims=True)  # shift for stability
    z = jnp.exp(-(values - m) / tau)
    S = jnp.sum(z, axis=axis, keepdims=True)
    softmin = -tau * jnp.log(S) + m
    alpha = z / (S + eps)  # normalized on W
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


DEFAULT_TOP_WEIGHTS: dict[str, float] = {
    # leaf terms
    "effector_pos": 1.0,
    "effector_hold_pos": 1.0,
    "effector_hold_vel": 1.0,
    "effector_final_vel": 1.0,
    "nn_output": 1e-5,
    "nn_hidden": 1e-5,
    # composite bundle (if enabled)
    "goal_hit_in_window": 1.0,
}

DEFAULT_GOAL_HIT_SUBWEIGHTS: dict[str, float] = {
    "pos": 1e-5,
    "vel": 2.0,
    "post_pos": 1.0,
    "late_pos": 0.0,
}

DEFAULT_GOAL_HIT_PARAMS: dict[str, Any] = {
    "window_bounds": (60, 80),
    "softmin_tau": 0.2,
    "post_pos_sigma_t": 5.0,
    "alpha_eps": 1e-12,
}


# def get_epochs_idxs(
#     trial_spec: TaskTrialSpec,
#     *,
#     end_epoch: Optional[int] = None,
#     start_epoch: Optional[int] = None,
# ):
#     if trial_spec.timeline.epoch_bounds is None:
#         # ? Or, could return empty array (no such epoch)
#         raise ValueError("Trial spec supplies no epoch_bounds")

#     if trial_spec.timeline.n_steps is None:
#         raise ValueError("Trial spec does not specify n_steps")

#     start, end = [  # mll
#         default if idx is None else trial_spec.timeline.epoch_bounds[idx]
#         for idx, default in [
#             [start_epoch, jnp.zeros((), dtype=int)],
#             [end_epoch, trial_spec.timeline.n_steps],
#         ]
#     ]
#     return jnp.arange(start, end)


def get_epoch_weights(
    trial_spec: TaskTrialSpec,
    *,
    end_epoch: Optional[int] = None,
    start_epoch: Optional[int] = None,
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

    # Compute start/end as traced scalars (OK)
    start = jnp.where(start_epoch is None, jnp.array(0, dtype=jnp.int32), bounds[start_epoch])
    end = jnp.where(end_epoch is None, jnp.array(n_steps, dtype=jnp.int32), bounds[end_epoch])

    # Build a fixed-length index vector once (needs T known at trace time, which is typical)
    t = jnp.arange(n_steps, dtype=jnp.int32)

    mask = (t >= start) & (t < end)
    return mask.astype(dtype)


during_hold = TargetSpec(time_mask=partial(get_epoch_weights, end_epoch=-2))
during_movement = TargetSpec(time_mask=partial(get_epoch_weights, start_epoch=-2))


def get_reach_loss(hps: TreeNamespace) -> CompositeLoss:
    """Construct a loss function for reaching task.

    Some terms are always included (e.g. penalty on neural network activity and output) while others
    (e.g. simple effector goal position error, versus hit-in-window error) depend on user config.

    Example of a typical YAML config fragment for the loss function:

      loss:
        weights:
          effector_pos: 1.0
          effector_final_vel: 1.0
          nn_output: 1e-5
          nn_hidden: 1e-5
          goal_hit_in_window: 1.0
        goal_hit_in_window:
          window_bounds: [60, 80]
          softmin_tau: 0.2
          post_pos_sigma_t: 5
          alpha_eps: 1e-12
          weights:
            pos: 1.0
            vel: 1.0
            post_pos: 1.0

    If `hps.loss.goal_hit_in_window` is present and `hps.loss.weights.goal_hit_in_window` is
    non-zero, then the goal-hit-in-window term is used; otherwise, the simple position error term
    is used.

    TODO: Refactor all the defaults/merging logic into a preparatory function.
    """
    user_outer_weights = _nsget(hps, "loss.weights", {}) or {}

    if (
        getattr(user_outer_weights, "effector_pos", 1.0) == 0.0
        and getattr(user_outer_weights, "goal_hit_in_window", 1.0) == 0.0
    ):
        logger.warning(
            "Both effector_pos and goal_hit_in_window loss term weights are zero; "
            "goal position error will NOT be penalized."
        )

    terms: Mapping[str, AbstractLoss] = dict(
        effector_final_vel=TargetStateLoss(
            "Effector final velocity",
            where=lambda state: state.mechanics.effector.vel,
            spec=target_zero & target_final_state,
        ),
        nn_output=TargetStateLoss(
            "Command",
            where=lambda state: state.efferent.output,
            spec=target_zero,
        ),
        nn_hidden=TargetStateLoss(
            "NN activity",
            where=lambda state: state.net.hidden,
            spec=target_zero,
        ),
    )

    if hps.task.type == "delayed_reach":
        terms["effector_hold_pos"] = TargetStateLoss(
            "Effector hold position",
            where=lambda state: state.mechanics.effector.pos,
            # norm=lambda x: jnp.sum(x**2, axis=-1),
            spec=during_hold,
        )
        terms["effector_hold_vel"] = TargetStateLoss(
            "Effector hold velocity",
            where=lambda state: state.mechanics.effector.vel,
            # norm=lambda x: jnp.sum(x**2, axis=-1),
            spec=target_zero & during_hold,
        )

    # if getattr(hps.loss, "stop_at_goal", False):
    #     terms["stop_at_goal"] = StopAtGoalLoss(**hps.loss.stop_at_goal)

    fix_readout_cfg = _nsget(hps, "loss.fix_readout_norm", None)
    if fix_readout_cfg:
        terms["fix_readout_norm"] = get_readout_norm_loss(**fix_readout_cfg)

    goal_hit_cfg = _nsget(hps, "loss.goal_hit_in_window", None)
    if goal_hit_cfg is not None and getattr(user_outer_weights, "goal_hit_in_window", 1.0) != 0.0:
        goal_hit_params = {**DEFAULT_GOAL_HIT_PARAMS, **(goal_hit_cfg or {})}

        window_bounds = tuple(goal_hit_params["window_bounds"])
        softmin_tau = float(goal_hit_params["softmin_tau"])
        post_pos_sigma_t = float(goal_hit_params["post_pos_sigma_t"])
        alpha_eps = float(goal_hit_params["alpha_eps"])

        goal_hit_terms = dict(
            pos=goal_hit_pos_loss_term_fn,
            vel=goal_hit_vel_loss_term_fn,
            post_pos=partial(post_hit_pos_loss_term_fn, sigma_t=post_pos_sigma_t),
            late_pos=goal_hit_late_pos_loss_term_fn,
        )

        # Merge default and user subweights
        goal_hit_subweights = {
            **DEFAULT_GOAL_HIT_SUBWEIGHTS,
            **_nsget(hps, "loss.goal_hit_in_window.weights", {}),
        }

        # Exclude zero-weighted sub-terms
        goal_hit_subweights = _filter_nonzero(goal_hit_subweights)
        goal_hit_terms = {k: v for k, v in goal_hit_terms.items() if k in goal_hit_subweights}

        terms["goal_hit_in_window"] = FuncTermsLoss(
            label="goal_hit_in_window",
            build_context=GoalHitCtxBuilder(
                window_bounds=window_bounds, tau=softmin_tau, eps=alpha_eps
            ),
            # Only include terms with nonzero weights
            terms=goal_hit_terms,
            weights=goal_hit_subweights,
        )
    else:
        terms["effector_pos"] = TargetStateLoss(
            "Effector position",
            where=lambda state: state.mechanics.effector.pos,
            norm=lambda x: jnp.sum(x**2, axis=-1),
            spec=during_movement,
            # norm=lambda *args, **kwargs: (
            #     # Euclidean distance
            #     jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2
            # ),
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
