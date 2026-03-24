import logging
from collections.abc import Callable, Mapping
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
    TermTree,
    target_final_state,
    target_zero,
)
from feedbax.task import TaskTrialSpec
from feedbax.xabdeef.losses import simple_reach_loss
from feedbax.misc import deep_merge
from feedbax.training.loss import get_readout_norm_loss
from feedbax.types import TreeNamespace
from jax_cookbook.misc import window_take
from jaxtyping import Array, PyTree

logger = logging.getLogger(__name__)


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


DEFAULT_TOP_WEIGHTS: dict[str, float] = {
    # leaf terms
    "effector_pos": 1.0,
    "effector_hold_pos": 1.0,
    "effector_hold_vel": 1.0,
    "effector_pos_mid": 1.0,
    "effector_vel_mid": 1.0,
    "effector_pos_late": 1.0,
    "effector_vel_late": 1.0,
    "effector_pos_running": 0.0,
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
    "start_step_after_go": 60,
    "end_step_after_go": 80,
    "softmin_tau": 0.2,
    "post_pos_sigma_t": 5.0,
    "alpha_eps": 1e-12,
}

DEFAULT_EFFECTOR_POS_LATE_PARAMS: dict[str, Any] = {
    "start_step_after_go": 80,
    "final_scale_factor": 1.0,
}

DEFAULT_EFFECTOR_VEL_LATE_PARAMS: dict[str, Any] = {
    "start_step_after_go": 80,
    "final_scale_factor": 1.0,
}

DEFAULT_EFFECTOR_POS_MID_PARAMS: dict[str, Any] = {
    "start_step_after_go": 0,
    "end_step_after_go": 80,
    "ramp_init_weight": 0.0,
    "ramp_final_weight": 0.1,
}

DEFAULT_EFFECTOR_VEL_MID_PARAMS: dict[str, Any] = {
    "start_step_after_go": 0,
    "end_step_after_go": 80,
    "ramp_init_weight": 0.0,
    "ramp_final_weight": 0.1,
}


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
    user_outer_weights = _nsget(hps, "loss.weights", {}) or {}

    terms: Mapping[str, AbstractLoss] = dict(
        nn_output=TargetStateLoss(
            "nn_output",
            where=lambda state: state.efferent.output,
            spec=target_zero,
        ),
        nn_hidden=TargetStateLoss(
            "nn_hidden",
            where=lambda state: state.net.hidden,
            spec=target_zero,
        ),
    )

    if hps.task.type == "delayed_reach":
        terms["effector_hold_pos"] = TargetStateLoss(
            "effector_hold_pos",
            where=lambda state: state.mechanics.effector.pos,
            # norm=lambda x: jnp.sum(x**2, axis=-1),
            spec=during_hold,
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
        terms["effector_pos"] = TargetStateLoss(
            "effector_pos",
            where=lambda state: state.mechanics.effector.pos,
            spec=during_movement,
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
        # Unlike mid-period terms (which ramp), this applies a uniform penalty from
        # go cue to trial end, encouraging the effector to track the target throughout.
        if getattr(user_outer_weights, "effector_pos_running", 0.0) != 0.0:
            terms["effector_pos_running"] = TargetStateLoss(
                "effector_pos_running",
                where=lambda state: state.mechanics.effector.pos,
                spec=during_movement,
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
        J_x = sum(
            losses[term].total for term in goal_terms if term in losses
        )
        J_u = losses[control_term].total  # Control cost (shape: (replicates,) or ())

        # Compute multiplicative update per replicate
        # Want: J_u ≈ target_ratio * J_x
        # So if J_u is too small, increase weight; if too large, decrease weight
        current_weight = loss_func.weights[control_term]
        ratio = J_x / (target_ratio * J_u + 1e-12)  # Add epsilon to avoid division by zero
        new_weight = current_weight * (ratio ** alpha)

        # Optional: clip to reasonable range to prevent runaway
        new_weight = jnp.clip(new_weight, 1e-8, 1e-2)

        # Update weights dict
        # new_weight is now a JAX array (scalar or shape (replicates,))
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
