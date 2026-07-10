"""Broad-epsilon randomized and PGD adversary training mechanisms."""
# ruff: noqa: F401, F403, F405

from __future__ import annotations

from rlrmp.train.training_configs import *  # noqa: F403

from dataclasses import dataclass
from typing import Any, Callable
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
from feedbax import TaskTrialSpec
from rlrmp.train.closed_loop_finite_adversary import (
    AFFINE_POLICY,
    FINITE_POLICY_BIAS_INPUT,
    FINITE_POLICY_GAINS_INPUT,
    finite_policy_step_epsilon,
)


@dataclass(frozen=True)
class _PgdAscentResult:
    objective_initial: jnp.ndarray
    objective_final_endpoint: jnp.ndarray
    best_candidate: Any
    best_objective: jnp.ndarray
    final_candidate: Any
    objective_nan_seen: jnp.ndarray
    objective_overflow_seen: jnp.ndarray


def _run_broad_epsilon_pgd_ascent(
    zero_candidate: Any,
    *,
    objective: Callable[[Any], jnp.ndarray],
    objective_and_grad: Callable[[Any], tuple[jnp.ndarray, Any]],
    proposal_from_gradient: Callable[[Any, Any], Any],
    mask_candidate: Callable[[Any], Any],
    cfg: PgdFullStateEpsilonTrainingConfig,
) -> _PgdAscentResult:
    """Run the shared broad-epsilon PGD/Adam ascent loop for one parameterization."""

    objective_initial, grad_initial = objective_and_grad(zero_candidate)
    grad_initial = mask_candidate(grad_initial)

    def select_best(best_candidate, best_objective, candidate, candidate_objective):
        improved = jnp.logical_and(
            jnp.isfinite(candidate_objective),
            candidate_objective > best_objective,
        )
        best_candidate = jax.tree.map(
            lambda best, current: jnp.where(improved, current, best),
            best_candidate,
            candidate,
        )
        best_objective = jnp.where(improved, candidate_objective, best_objective)
        return best_candidate, best_objective

    def run_projected_gradient_ascent():
        def body(_, state):
            (
                candidate_current,
                _current_objective,
                grad_current,
                best_candidate,
                best_objective,
                objective_nan_seen,
                objective_overflow_seen,
            ) = state
            proposal = proposal_from_gradient(candidate_current, grad_current)
            proposal_objective, proposal_grad = objective_and_grad(proposal)
            proposal_grad = mask_candidate(proposal_grad)
            objective_nan_seen = jnp.logical_or(
                objective_nan_seen,
                jnp.isnan(proposal_objective),
            )
            objective_overflow_seen = jnp.logical_or(
                objective_overflow_seen,
                jnp.isinf(proposal_objective),
            )
            best_candidate, best_objective = select_best(
                best_candidate,
                best_objective,
                proposal,
                proposal_objective,
            )
            return (
                proposal,
                proposal_objective,
                proposal_grad,
                best_candidate,
                best_objective,
                objective_nan_seen,
                objective_overflow_seen,
            )

        candidate_current = zero_candidate
        current_objective = objective_initial
        grad_current = grad_initial
        best_candidate = zero_candidate
        objective_best = objective_initial
        objective_nan_seen = jnp.isnan(objective_initial)
        objective_overflow_seen = jnp.isinf(objective_initial)
        if int(cfg.n_steps) > 1:
            (
                candidate_current,
                current_objective,
                grad_current,
                best_candidate,
                objective_best,
                objective_nan_seen,
                objective_overflow_seen,
            ) = jax.lax.fori_loop(
                0,
                int(cfg.n_steps) - 1,
                body,
                (
                    candidate_current,
                    current_objective,
                    grad_current,
                    best_candidate,
                    objective_best,
                    objective_nan_seen,
                    objective_overflow_seen,
                ),
            )

        final_candidate = proposal_from_gradient(candidate_current, grad_current)
        objective_final_endpoint = objective(final_candidate)
        objective_nan_seen = jnp.logical_or(
            objective_nan_seen,
            jnp.isnan(objective_final_endpoint),
        )
        objective_overflow_seen = jnp.logical_or(
            objective_overflow_seen,
            jnp.isinf(objective_final_endpoint),
        )
        best_candidate, objective_best = select_best(
            best_candidate,
            objective_best,
            final_candidate,
            objective_final_endpoint,
        )
        return (
            final_candidate,
            objective_final_endpoint,
            best_candidate,
            objective_best,
            objective_nan_seen,
            objective_overflow_seen,
        )

    def run_adam_ascent():
        optimizer = optax.adam(
            learning_rate=float(cfg.adam_learning_rate),
            b1=float(cfg.adam_b1),
            b2=float(cfg.adam_b2),
            eps=float(cfg.adam_eps),
        )
        opt_state0 = optimizer.init(zero_candidate)

        def body(_, state):
            (
                candidate_current,
                _current_objective,
                grad_current,
                opt_state,
                best_candidate,
                best_objective,
                objective_nan_seen,
                objective_overflow_seen,
            ) = state
            ascent_grad = jax.tree.map(lambda grad: -grad, grad_current)
            updates, opt_state = optimizer.update(ascent_grad, opt_state, candidate_current)
            proposal = mask_candidate(eqx.apply_updates(candidate_current, updates))
            proposal_objective, proposal_grad = objective_and_grad(proposal)
            proposal_grad = mask_candidate(proposal_grad)
            objective_nan_seen = jnp.logical_or(
                objective_nan_seen,
                jnp.isnan(proposal_objective),
            )
            objective_overflow_seen = jnp.logical_or(
                objective_overflow_seen,
                jnp.isinf(proposal_objective),
            )
            best_candidate, best_objective = select_best(
                best_candidate,
                best_objective,
                proposal,
                proposal_objective,
            )
            return (
                proposal,
                proposal_objective,
                proposal_grad,
                opt_state,
                best_candidate,
                best_objective,
                objective_nan_seen,
                objective_overflow_seen,
            )

        (
            final_candidate,
            objective_final_endpoint,
            _grad_final,
            _opt_state_final,
            best_candidate,
            objective_best,
            objective_nan_seen,
            objective_overflow_seen,
        ) = jax.lax.fori_loop(
            0,
            int(cfg.n_steps),
            body,
            (
                zero_candidate,
                objective_initial,
                grad_initial,
                opt_state0,
                zero_candidate,
                objective_initial,
                jnp.isnan(objective_initial),
                jnp.isinf(objective_initial),
            ),
        )
        return (
            final_candidate,
            objective_final_endpoint,
            best_candidate,
            objective_best,
            objective_nan_seen,
            objective_overflow_seen,
        )

    if cfg.inner_optimizer_method == BROAD_EPSILON_PGD_ADAM:
        (
            final_candidate,
            objective_final_endpoint,
            best_candidate,
            objective_best,
            objective_nan_seen,
            objective_overflow_seen,
        ) = run_adam_ascent()
    else:
        (
            final_candidate,
            objective_final_endpoint,
            best_candidate,
            objective_best,
            objective_nan_seen,
            objective_overflow_seen,
        ) = run_projected_gradient_ascent()
    return _PgdAscentResult(
        objective_initial=objective_initial,
        objective_final_endpoint=objective_final_endpoint,
        best_candidate=best_candidate,
        best_objective=objective_best,
        final_candidate=final_candidate,
        objective_nan_seen=objective_nan_seen,
        objective_overflow_seen=objective_overflow_seen,
    )


def run_broad_epsilon_pgd_inner_maximizer(
    task: Any,
    model: Any,
    trial_specs: TaskTrialSpec,
    loss_func: Any,
    keys_model: Any,
    config: Any,
    *,
    soft_energy_lambda_override: Any | None = None,
    return_diagnostics: bool = False,
) -> tuple[TaskTrialSpec, dict[str, jnp.ndarray]]:
    """Run the PGD inner maximizer and optionally return compact scalar diagnostics."""

    cfg = PgdFullStateEpsilonTrainingConfig.from_payload(config)
    if (
        soft_energy_lambda_override is not None
        and cfg.objective_kind != BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
    ):
        raise ValueError("soft_energy_lambda_override is only valid for soft-energy PGD.")
    if (
        soft_energy_lambda_override is not None
        and cfg.adversary_mechanism != BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
    ):
        raise ValueError(
            "soft_energy_lambda_override is only supported for the direct_epsilon PGD mechanism."
        )
    if cfg.adversary_mechanism in BROAD_EPSILON_PGD_FINITE_POLICY_MECHANISMS:
        return _run_finite_broad_epsilon_pgd_inner_maximizer(
            task,
            model,
            trial_specs,
            loss_func,
            keys_model,
            cfg,
            return_diagnostics=return_diagnostics,
        )
    specs = _ensure_broad_epsilon_input(trial_specs, epsilon_dim=cfg.epsilon_dim)
    base_epsilon = jnp.asarray(specs.inputs["epsilon"])
    radius = (
        None
        if _pgd_cap_free_direct_soft_energy(cfg)
        else _broad_epsilon_pgd_trust_radius(specs, cfg).astype(base_epsilon.dtype)
    )
    time_mask = _epsilon_time_mask(specs, base_epsilon, cfg.movement_epoch_only)
    delta = jnp.zeros_like(base_epsilon)
    soft_energy_lambda = (
        0.0
        if cfg.objective_kind != BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
        else (
            soft_energy_lambda_override
            if soft_energy_lambda_override is not None
            else cfg.soft_energy_lambda
        )
    )

    def objective_components(delta_candidate):
        masked_delta = delta_candidate * time_mask
        candidate = _set_input(specs, "epsilon", base_epsilon + masked_delta)
        candidate_states = task.eval_trials(model, candidate, keys_model)
        task_loss = jnp.asarray(loss_func(candidate_states, candidate, model).total)
        energy_per_trial = _epsilon_energy_per_trial(masked_delta)
        energy_mean = jnp.mean(energy_per_trial)
        if cfg.objective_kind != BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE:
            penalty = jnp.asarray(0.0, dtype=task_loss.dtype)
        else:
            penalty = jnp.asarray(soft_energy_lambda, dtype=task_loss.dtype) * energy_mean
        return task_loss, energy_per_trial, penalty, task_loss - penalty

    def objective(delta_candidate):
        return objective_components(delta_candidate)[-1]

    def objective_and_grad(delta_candidate):
        return jax.value_and_grad(objective)(delta_candidate)

    zero_delta = jnp.zeros_like(base_epsilon)
    if radius is None:
        step_scale = jnp.asarray(cfg.step_size_fraction, dtype=base_epsilon.dtype)
    else:
        step_scale = _expand_radius(
            radius * jnp.asarray(cfg.step_size_fraction, dtype=base_epsilon.dtype),
            base_epsilon.ndim,
        )

    def proposal_from_gradient(delta_current, grad_current):
        step = _normalize_flattened_per_trial(grad_current) * step_scale
        proposal = (delta_current + step) * time_mask
        if radius is None:
            return proposal
        return _project_flattened_per_trial_l2_ball(proposal, radius)

    ascent = _run_broad_epsilon_pgd_ascent(
        zero_delta,
        objective=objective,
        objective_and_grad=objective_and_grad,
        proposal_from_gradient=proposal_from_gradient,
        mask_candidate=lambda delta_candidate: delta_candidate * time_mask,
        cfg=cfg,
    )
    objective_initial = ascent.objective_initial
    final_delta = ascent.final_candidate
    objective_final_endpoint = ascent.objective_final_endpoint
    best_delta = ascent.best_candidate
    objective_best = ascent.best_objective
    objective_nan_seen = ascent.objective_nan_seen
    objective_overflow_seen = ascent.objective_overflow_seen
    delta = jax.lax.stop_gradient(best_delta * time_mask)
    updated = _set_input(specs, "epsilon", base_epsilon + delta)
    if not return_diagnostics:
        return updated, {}

    objective_selected = objective_best
    diagnostic_dtype = base_epsilon.dtype if radius is None else radius.dtype
    delta_norm = _flattened_per_trial_norm(delta).astype(diagnostic_dtype)
    delta_energy = _epsilon_energy_per_trial(delta)
    if radius is None:
        radius_mean = jnp.asarray(jnp.nan, dtype=diagnostic_dtype)
        radius_max = jnp.asarray(jnp.nan, dtype=diagnostic_dtype)
        ratio_mean = jnp.asarray(jnp.nan, dtype=diagnostic_dtype)
        ratio_max = jnp.asarray(jnp.nan, dtype=diagnostic_dtype)
        boundary_fraction = jnp.asarray(0.0, dtype=diagnostic_dtype)
    else:
        ratio = delta_norm / jnp.maximum(radius, jnp.asarray(1e-12, dtype=radius.dtype))
        boundary = ratio >= jnp.asarray(1.0 - 1e-4, dtype=ratio.dtype)
        radius_mean = jnp.mean(radius)
        radius_max = jnp.max(radius)
        ratio_mean = jnp.mean(ratio)
        ratio_max = jnp.max(ratio)
        boundary_fraction = jnp.mean(boundary.astype(radius.dtype))
    zero_task_loss, zero_energy, zero_penalty, zero_objective = objective_components(zero_delta)
    selected_task_loss, selected_energy, selected_penalty, selected_objective = (
        objective_components(delta)
    )
    final_task_loss, final_energy, final_penalty, final_objective = objective_components(
        final_delta
    )
    del zero_energy, selected_energy, final_energy
    objective_nonfinite_seen = jnp.logical_or(objective_nan_seen, objective_overflow_seen)
    projection_active = radius is not None
    diagnostics = {
        "radius_mean": radius_mean,
        "radius_max": radius_max,
        "epsilon_norm_mean": jnp.mean(delta_norm),
        "epsilon_norm_max": jnp.max(delta_norm),
        "epsilon_energy_mean": jnp.mean(delta_energy),
        "epsilon_energy_max": jnp.max(delta_energy),
        "epsilon_norm_radius_ratio_mean": ratio_mean,
        "epsilon_norm_radius_ratio_max": ratio_max,
        "inner_objective_before": jnp.asarray(objective_initial),
        "inner_objective_after": jnp.asarray(objective_selected),
        "inner_objective_improvement": jnp.asarray(objective_selected - objective_initial),
        "inner_objective_best": jnp.asarray(objective_best),
        "inner_objective_final_endpoint": jnp.asarray(objective_final_endpoint),
        "inner_objective_final_endpoint_gap": jnp.asarray(
            objective_best - objective_final_endpoint
        ),
        "raw_task_loss_zero": jnp.asarray(zero_task_loss),
        "raw_task_loss_selected": jnp.asarray(selected_task_loss),
        "raw_task_loss_final_endpoint": jnp.asarray(final_task_loss),
        "energy_penalty_term_zero": jnp.asarray(zero_penalty),
        "energy_penalty_term_selected": jnp.asarray(selected_penalty),
        "energy_penalty_term_final_endpoint": jnp.asarray(final_penalty),
        "penalized_objective_zero": jnp.asarray(zero_objective),
        "penalized_objective_selected": jnp.asarray(selected_objective),
        "penalized_objective_final_endpoint": jnp.asarray(final_objective),
        "selected_objective_gain_over_zero": jnp.asarray(objective_selected - zero_objective),
        "selected_vs_final_objective_gap": jnp.asarray(objective_best - objective_final_endpoint),
        "boundary_fraction": boundary_fraction,
        "cap_boundary_fraction": boundary_fraction,
        "safety_cap_boundary_fraction": boundary_fraction,
        "inner_objective_nan_seen": objective_nan_seen,
        "inner_objective_overflow_seen": objective_overflow_seen,
        "inner_objective_nonfinite_seen": objective_nonfinite_seen,
        "n_steps": jnp.asarray(cfg.n_steps, dtype=jnp.float32),
        "step_size_fraction_of_l2_radius": jnp.asarray(
            cfg.step_size_fraction,
            dtype=jnp.float32,
        ),
        "step_size_uses_radius": jnp.asarray(projection_active),
        "inner_optimizer_method_is_adam": jnp.asarray(
            cfg.inner_optimizer_method == BROAD_EPSILON_PGD_ADAM
        ),
        "adam_learning_rate": jnp.asarray(cfg.adam_learning_rate, dtype=jnp.float32),
        "objective_kind_is_soft_energy": jnp.asarray(
            cfg.objective_kind == BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
        ),
        "energy_lambda": jnp.asarray(soft_energy_lambda, dtype=jnp.float32),
        "energy_lambda_override_active": jnp.asarray(soft_energy_lambda_override is not None),
        "projection_active": jnp.asarray(projection_active),
        "radius_bound_mode": jnp.asarray(projection_active),
        "cap_free_soft_energy": jnp.asarray(_pgd_cap_free_direct_soft_energy(cfg)),
        "safety_cap_enabled": jnp.asarray(cfg.safety_cap_l2_radius_15cm is not None),
    }
    return updated, diagnostics


def _run_finite_broad_epsilon_pgd_inner_maximizer(
    task: Any,
    model: Any,
    trial_specs: TaskTrialSpec,
    loss_func: Any,
    keys_model: Any,
    cfg: PgdFullStateEpsilonTrainingConfig,
    *,
    return_diagnostics: bool = False,
) -> tuple[TaskTrialSpec, dict[str, jnp.ndarray]]:
    """Run PGD over finite closed-loop policy inputs for broad epsilon."""

    if cfg.objective_kind != BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE:
        raise ValueError(
            "Finite broad-epsilon PGD mechanisms currently require "
            "--broad-epsilon-pgd-objective soft_energy; hard-L2 projection is only "
            "defined for the direct_epsilon sequence mechanism."
        )
    specs = _ensure_broad_epsilon_input(trial_specs, epsilon_dim=cfg.epsilon_dim)
    base_epsilon = jnp.asarray(specs.inputs["epsilon"])
    radius = _broad_epsilon_pgd_trust_radius(specs, cfg).astype(base_epsilon.dtype)
    time_mask = _epsilon_time_mask(specs, base_epsilon, cfg.movement_epoch_only)
    horizon = int(base_epsilon.shape[-2])
    feature_dim = int(jnp.asarray(specs.inits["mechanics.vector"]).shape[-1])
    policy_mask = _shared_policy_time_mask(time_mask)
    zero_params = _zero_finite_policy_params(
        cfg,
        horizon=horizon,
        feature_dim=feature_dim,
        dtype=base_epsilon.dtype,
    )
    zero_params = _mask_finite_policy_params(zero_params, policy_mask)

    def candidate_specs(params):
        masked = _mask_finite_policy_params(params, policy_mask)
        candidate = _set_input(
            specs,
            FINITE_POLICY_GAINS_INPUT,
            _broadcast_finite_policy_params_to_batch(masked["gains"], base_epsilon),
        )
        if cfg.adversary_mechanism == AFFINE_POLICY:
            candidate = _set_input(
                candidate,
                FINITE_POLICY_BIAS_INPUT,
                _broadcast_finite_policy_params_to_batch(masked["bias"], base_epsilon),
            )
        return candidate

    def objective_components(params):
        candidate = candidate_specs(params)
        candidate_states = task.eval_trials(model, candidate, keys_model)
        task_loss = jnp.asarray(loss_func(candidate_states, candidate, model).total)
        epsilon_delta = _finite_policy_epsilon_from_rollout(
            candidate_states,
            candidate,
            cfg,
        )
        energy_per_trial = _epsilon_energy_per_trial(epsilon_delta)
        energy_mean = jnp.mean(energy_per_trial)
        penalty = jnp.asarray(cfg.soft_energy_lambda, dtype=task_loss.dtype) * energy_mean
        return task_loss, energy_per_trial, penalty, task_loss - penalty, epsilon_delta

    def objective(params):
        return objective_components(params)[-2]

    def objective_and_grad(params):
        return jax.value_and_grad(objective)(params)

    step_size = jnp.asarray(cfg.step_size_fraction, dtype=base_epsilon.dtype) * jnp.mean(radius)

    def proposal_from_gradient(params_current, grad_current):
        grad_norm = _finite_policy_tree_norm(grad_current)
        scaled = jax.tree.map(
            lambda param, grad: (
                param
                + step_size
                * grad
                / jnp.maximum(grad_norm, jnp.asarray(1e-12, dtype=step_size.dtype))
            ),
            params_current,
            grad_current,
        )
        return _mask_finite_policy_params(scaled, policy_mask)

    ascent = _run_broad_epsilon_pgd_ascent(
        zero_params,
        objective=objective,
        objective_and_grad=objective_and_grad,
        proposal_from_gradient=proposal_from_gradient,
        mask_candidate=lambda params: _mask_finite_policy_params(params, policy_mask),
        cfg=cfg,
    )
    objective_initial = ascent.objective_initial
    final_params = ascent.final_candidate
    objective_final_endpoint = ascent.objective_final_endpoint
    best_params = ascent.best_candidate
    objective_best = ascent.best_objective
    objective_nan_seen = ascent.objective_nan_seen
    objective_overflow_seen = ascent.objective_overflow_seen
    best_params = jax.tree.map(jax.lax.stop_gradient, best_params)
    updated = candidate_specs(best_params)
    if not return_diagnostics:
        return updated, {}

    objective_selected = objective_best
    zero_task_loss, _zero_energy, zero_penalty, zero_objective, zero_delta = objective_components(
        zero_params
    )
    selected_task_loss, _selected_energy, selected_penalty, selected_objective, delta = (
        objective_components(best_params)
    )
    final_task_loss, _final_energy, final_penalty, final_objective, final_delta = (
        objective_components(final_params)
    )
    delta_norm = _flattened_per_trial_norm(delta).astype(radius.dtype)
    delta_energy = _epsilon_energy_per_trial(delta)
    ratio = delta_norm / jnp.maximum(radius, jnp.asarray(1e-12, dtype=radius.dtype))
    boundary = ratio >= jnp.asarray(1.0 - 1e-4, dtype=ratio.dtype)
    objective_nonfinite_seen = jnp.logical_or(objective_nan_seen, objective_overflow_seen)
    diagnostics = {
        "radius_mean": jnp.mean(radius),
        "radius_max": jnp.max(radius),
        "epsilon_norm_mean": jnp.mean(delta_norm),
        "epsilon_norm_max": jnp.max(delta_norm),
        "epsilon_energy_mean": jnp.mean(delta_energy),
        "epsilon_energy_max": jnp.max(delta_energy),
        "epsilon_norm_radius_ratio_mean": jnp.mean(ratio),
        "epsilon_norm_radius_ratio_max": jnp.max(ratio),
        "inner_objective_before": jnp.asarray(objective_initial),
        "inner_objective_after": jnp.asarray(objective_selected),
        "inner_objective_improvement": jnp.asarray(objective_selected - objective_initial),
        "inner_objective_best": jnp.asarray(objective_best),
        "inner_objective_final_endpoint": jnp.asarray(objective_final_endpoint),
        "inner_objective_final_endpoint_gap": jnp.asarray(
            objective_best - objective_final_endpoint
        ),
        "raw_task_loss_zero": jnp.asarray(zero_task_loss),
        "raw_task_loss_selected": jnp.asarray(selected_task_loss),
        "raw_task_loss_final_endpoint": jnp.asarray(final_task_loss),
        "energy_penalty_term_zero": jnp.asarray(zero_penalty),
        "energy_penalty_term_selected": jnp.asarray(selected_penalty),
        "energy_penalty_term_final_endpoint": jnp.asarray(final_penalty),
        "penalized_objective_zero": jnp.asarray(zero_objective),
        "penalized_objective_selected": jnp.asarray(selected_objective),
        "penalized_objective_final_endpoint": jnp.asarray(final_objective),
        "selected_objective_gain_over_zero": jnp.asarray(objective_selected - zero_objective),
        "selected_vs_final_objective_gap": jnp.asarray(objective_best - objective_final_endpoint),
        "boundary_fraction": jnp.mean(boundary.astype(radius.dtype)),
        "cap_boundary_fraction": jnp.mean(boundary.astype(radius.dtype)),
        "safety_cap_boundary_fraction": jnp.mean(boundary.astype(radius.dtype)),
        "inner_objective_nan_seen": objective_nan_seen,
        "inner_objective_overflow_seen": objective_overflow_seen,
        "inner_objective_nonfinite_seen": objective_nonfinite_seen,
        "n_steps": jnp.asarray(cfg.n_steps, dtype=jnp.float32),
        "step_size_fraction_of_l2_radius": jnp.asarray(
            cfg.step_size_fraction,
            dtype=jnp.float32,
        ),
        "inner_optimizer_method_is_adam": jnp.asarray(
            cfg.inner_optimizer_method == BROAD_EPSILON_PGD_ADAM
        ),
        "adam_learning_rate": jnp.asarray(cfg.adam_learning_rate, dtype=jnp.float32),
        "objective_kind_is_soft_energy": jnp.asarray(True),
        "energy_lambda": jnp.asarray(
            cfg.soft_energy_lambda or 0.0,
            dtype=jnp.float32,
        ),
        "finite_policy_class_is_affine": jnp.asarray(cfg.adversary_mechanism == AFFINE_POLICY),
        "finite_policy_delta_zero_energy_mean": jnp.mean(_epsilon_energy_per_trial(zero_delta)),
        "finite_policy_final_endpoint_energy_mean": jnp.mean(
            _epsilon_energy_per_trial(final_delta)
        ),
    }
    return updated, diagnostics


def _zero_finite_policy_params(
    cfg: PgdFullStateEpsilonTrainingConfig,
    *,
    horizon: int,
    feature_dim: int,
    dtype: Any,
) -> dict[str, jnp.ndarray]:
    params = {
        "gains": jnp.zeros((int(horizon), int(cfg.epsilon_dim), int(feature_dim)), dtype=dtype)
    }
    if cfg.adversary_mechanism == AFFINE_POLICY:
        params["bias"] = jnp.zeros((int(horizon), int(cfg.epsilon_dim)), dtype=dtype)
    return params


def _mask_finite_policy_params(
    params: dict[str, jnp.ndarray],
    policy_mask: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    masked = {"gains": params["gains"] * policy_mask[:, None, None]}
    if "bias" in params:
        masked["bias"] = params["bias"] * policy_mask[:, None]
    return masked


def _shared_policy_time_mask(time_mask: jnp.ndarray) -> jnp.ndarray:
    mask = jnp.asarray(time_mask)
    time_axis = mask.ndim - 2
    reduce_axes = tuple(axis for axis in range(mask.ndim) if axis != time_axis)
    if not reduce_axes:
        return mask
    return jnp.max(mask, axis=reduce_axes)


def _broadcast_finite_policy_params_to_batch(
    values: jnp.ndarray,
    base_epsilon: jnp.ndarray,
) -> jnp.ndarray:
    batch_shape = base_epsilon.shape[:-2]
    if not batch_shape:
        return values
    return jnp.broadcast_to(values, (*batch_shape, *values.shape))


def _finite_policy_tree_norm(tree: Any) -> jnp.ndarray:
    flat, _ = ravel_pytree(tree)
    return jnp.linalg.norm(flat)


def _finite_policy_epsilon_from_rollout(
    states: Any,
    trial_specs: TaskTrialSpec,
    cfg: PgdFullStateEpsilonTrainingConfig,
) -> jnp.ndarray:
    vectors = jnp.asarray(states.mechanics.vector)
    gains = jnp.asarray(trial_specs.inputs[FINITE_POLICY_GAINS_INPUT], dtype=vectors.dtype)
    horizon = int(gains.shape[-3])
    init = jnp.asarray(trial_specs.inits["mechanics.vector"], dtype=vectors.dtype)
    while init.ndim < vectors.ndim - 1:
        init = jnp.expand_dims(init, axis=0)
    init = jnp.broadcast_to(init, (*vectors.shape[:-2], vectors.shape[-1]))
    pre_step_vectors = jnp.concatenate(
        [init[..., None, :], vectors[..., : max(horizon - 1, 0), :]],
        axis=-2,
    )
    target = jnp.asarray(
        trial_specs.inputs.get(
            "target",
            trial_specs.targets["mechanics.effector.pos"].value,
        ),
        dtype=vectors.dtype,
    )[..., :horizon, :]
    bias = (
        jnp.asarray(trial_specs.inputs[FINITE_POLICY_BIAS_INPUT], dtype=vectors.dtype)
        if cfg.adversary_mechanism == AFFINE_POLICY
        else None
    )
    return finite_policy_step_epsilon(
        pre_step_vectors,
        target_position=target,
        gain_t=gains,
        bias_t=bias,
        physical_block_size=int(cfg.epsilon_dim),
    )


def _epsilon_energy_per_trial(epsilon: jnp.ndarray) -> jnp.ndarray:
    """Return squared-L2 epsilon energy per leading trial entry."""

    energy_axes = tuple(range(max(epsilon.ndim - 2, 0), epsilon.ndim))
    return jnp.sum(jnp.square(epsilon), axis=energy_axes)


def _ensure_broad_epsilon_input(
    trial_specs: TaskTrialSpec,
    *,
    epsilon_dim: int = BROAD_EPSILON_DIM,
) -> TaskTrialSpec:
    """Ensure an epsilon input exists and is broadcast to the trial batch."""

    if "epsilon" not in trial_specs.inputs:
        zeros = jnp.zeros(
            (*_batch_shape(trial_specs), int(trial_specs.timeline.n_steps), int(epsilon_dim)),
            dtype=jnp.float32,
        )
        trial_specs = _set_input(trial_specs, "epsilon", zeros)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    if epsilon.shape[-1] != int(epsilon_dim):
        raise ValueError(
            f"PGD broad full-state epsilon expects a {int(epsilon_dim)}D process "
            "epsilon input; "
            f"got trailing dimension {epsilon.shape[-1]}."
        )
    batch_shape = _batch_shape(trial_specs)
    if batch_shape and epsilon.shape[: len(batch_shape)] != batch_shape:
        epsilon = jnp.broadcast_to(epsilon, (*batch_shape, *epsilon.shape[-2:]))
        trial_specs = _set_input(trial_specs, "epsilon", epsilon)
    return trial_specs


def _flattened_per_trial_norm(x: jnp.ndarray) -> jnp.ndarray:
    axes = tuple(range(max(x.ndim - 2, 0), x.ndim))
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axes))


def _flattened_per_trial_safe_norm(x: jnp.ndarray) -> jnp.ndarray:
    axes = tuple(range(max(x.ndim - 2, 0), x.ndim))
    squared = jnp.sum(jnp.square(x), axis=axes)
    return jnp.sqrt(jnp.maximum(squared, jnp.asarray(1e-24, dtype=x.dtype)))


def _expand_radius(radius: jnp.ndarray, ndim: int) -> jnp.ndarray:
    while radius.ndim < ndim:
        radius = jnp.expand_dims(radius, axis=-1)
    return radius


def _normalize_flattened_per_trial(x: jnp.ndarray) -> jnp.ndarray:
    norms = _expand_radius(_flattened_per_trial_safe_norm(x), x.ndim)
    return x / jnp.maximum(norms, jnp.asarray(1e-12, dtype=x.dtype))


def _project_flattened_per_trial_l2_ball(
    x: jnp.ndarray,
    radius: jnp.ndarray,
) -> jnp.ndarray:
    radius_expanded = _expand_radius(radius.astype(x.dtype), x.ndim)
    norms = _expand_radius(_flattened_per_trial_safe_norm(x).astype(x.dtype), x.ndim)
    scale = jnp.minimum(
        1.0,
        radius_expanded / jnp.maximum(norms, jnp.asarray(1e-12, dtype=x.dtype)),
    )
    return x * scale


def _epsilon_time_mask(
    trial_specs: TaskTrialSpec,
    epsilon: jnp.ndarray,
    movement_epoch_only: bool,
) -> jnp.ndarray:
    """Return a broadcastable epsilon mask over ``[..., time, component]``."""

    if not movement_epoch_only:
        return jnp.ones_like(epsilon)
    bounds = trial_specs.timeline.epoch_bounds
    if bounds is None:
        raise ValueError("movement-epoch broad-epsilon masking requires epoch bounds.")
    bounds = jnp.asarray(bounds)
    t = jnp.arange(epsilon.shape[-2], dtype=bounds.dtype)
    if bounds.ndim == 1:
        time_mask = (t >= bounds[-2]) & (t < bounds[-1])
    else:
        start = bounds[..., -2]
        end = bounds[..., -1]
        time_mask = (t >= jnp.expand_dims(start, -1)) & (t < jnp.expand_dims(end, -1))
    while time_mask.ndim < epsilon.ndim - 1:
        time_mask = jnp.expand_dims(time_mask, axis=0)
    return jnp.expand_dims(time_mask.astype(epsilon.dtype), axis=-1)


def _resolve_sisu_condition_input(
    trial_specs: TaskTrialSpec,
    config: PgdFullStateEpsilonTrainingConfig,
) -> str:
    if config.sisu_condition_input != "auto":
        if config.sisu_condition_input not in trial_specs.inputs:
            raise ValueError(
                f"SISU-conditioned PGD budget requested input "
                f"{config.sisu_condition_input!r}, but the trial has inputs "
                f"{sorted(trial_specs.inputs)}."
            )
        return config.sisu_condition_input
    for name in ("sisu", "input"):
        if name in trial_specs.inputs:
            return name
    raise ValueError(
        "SISU-conditioned PGD budget requires trial_specs.inputs['sisu'] or "
        "trial_specs.inputs['input']."
    )


def _sisu_condition_values(
    trial_specs: TaskTrialSpec,
    config: PgdFullStateEpsilonTrainingConfig,
) -> jnp.ndarray:
    input_name = _resolve_sisu_condition_input(trial_specs, config)
    values = jnp.asarray(trial_specs.inputs[input_name], dtype=jnp.float32)
    batch_shape = _batch_shape(trial_specs)
    if values.shape == batch_shape:
        return values
    reduce_axes = tuple(range(len(batch_shape), values.ndim))
    if not reduce_axes:
        return values
    return jnp.mean(values, axis=reduce_axes)


def _broad_epsilon_l2_radius(
    trial_specs: TaskTrialSpec,
    config: BroadFullStateEpsilonTrainingConfig,
) -> jnp.ndarray:
    """Return per-trial L2 radius for broad full-state epsilon sampling."""

    if (
        isinstance(config, PgdFullStateEpsilonTrainingConfig)
        and config.budget_schedule == BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE
    ):
        radius = jnp.asarray(config.sisu_max_l2_radius, dtype=jnp.float32)
    else:
        radius = jnp.asarray(config.reference_l2_radius, dtype=jnp.float32)
    if not config.reach_length_scaling:
        scaled_radius = jnp.broadcast_to(radius, _batch_shape(trial_specs))
    else:
        reach_length = _trial_reach_length_m(trial_specs)
        scaled_radius = radius * (
            reach_length / jnp.asarray(config.nominal_reach_length_m, dtype=reach_length.dtype)
        )
    if (
        isinstance(config, PgdFullStateEpsilonTrainingConfig)
        and config.budget_schedule == BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE
    ):
        sisu = jnp.clip(_sisu_condition_values(trial_specs, config), 0.0, 1.0)
        return scaled_radius * jnp.sqrt(sisu.astype(scaled_radius.dtype))
    return scaled_radius


def _broad_epsilon_pgd_trust_radius(
    trial_specs: TaskTrialSpec,
    config: PgdFullStateEpsilonTrainingConfig,
) -> jnp.ndarray:
    """Return the projection radius or soft-PGD stabilization cap."""

    if config.objective_kind != BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE:
        return _broad_epsilon_l2_radius(trial_specs, config)
    radius = jnp.asarray(config.safety_cap_l2_radius, dtype=jnp.float32)
    if not config.reach_length_scaling:
        return jnp.broadcast_to(radius, _batch_shape(trial_specs))
    reach_length = _trial_reach_length_m(trial_specs)
    return radius * (
        reach_length / jnp.asarray(config.nominal_reach_length_m, dtype=reach_length.dtype)
    )


def _trial_reach_length_m(trial_specs: TaskTrialSpec) -> jnp.ndarray:
    target_pos = _trial_target_position_m(trial_specs)
    init_vector = jnp.asarray(trial_specs.inits["mechanics.vector"])
    init_pos = init_vector[..., :2]
    try:
        delta = target_pos - init_pos
    except TypeError:
        delta = target_pos
    return jnp.linalg.norm(delta, axis=-1)


def _trial_target_position_m(trial_specs: TaskTrialSpec) -> jnp.ndarray:
    target_spec = trial_specs.targets["mechanics.effector.pos"]
    target = jnp.asarray(target_spec.value)
    if target.ndim >= 2:
        return target[..., -1, :]
    return target


def _set_input(trial_specs: TaskTrialSpec, key: str, value: Any) -> TaskTrialSpec:
    inputs = dict(trial_specs.inputs)
    inputs[key] = value
    return eqx.tree_at(lambda ts: ts.inputs, trial_specs, inputs)


def _batch_shape(trial_specs: TaskTrialSpec) -> tuple[int, ...]:
    target = trial_specs.targets["mechanics.effector.pos"].value
    return tuple(target.shape[:-2]) if target.ndim >= 3 else ()


__all__ = [
    "_PgdAscentResult",
    "_batch_shape",
    "_broad_epsilon_l2_radius",
    "_broad_epsilon_pgd_trust_radius",
    "_broadcast_finite_policy_params_to_batch",
    "_ensure_broad_epsilon_input",
    "_epsilon_energy_per_trial",
    "_epsilon_time_mask",
    "_expand_radius",
    "_finite_policy_epsilon_from_rollout",
    "_finite_policy_tree_norm",
    "_flattened_per_trial_norm",
    "_flattened_per_trial_safe_norm",
    "_mask_finite_policy_params",
    "_normalize_flattened_per_trial",
    "_project_flattened_per_trial_l2_ball",
    "_resolve_sisu_condition_input",
    "_run_broad_epsilon_pgd_ascent",
    "_run_finite_broad_epsilon_pgd_inner_maximizer",
    "_set_input",
    "_shared_policy_time_mask",
    "_sisu_condition_values",
    "_trial_reach_length_m",
    "_trial_target_position_m",
    "_zero_finite_policy_params",
    "run_broad_epsilon_pgd_inner_maximizer",
]
