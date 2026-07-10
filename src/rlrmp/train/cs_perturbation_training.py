"""Fixed-target perturbation-generalized training config for C&S GRU runs."""
# ruff: noqa: F401, F403, F405

from __future__ import annotations

from rlrmp.train.fixed_target_perturbation_training import (
    _add_graph_channel_calibrated_random_pulse,
    _add_graph_channel_random_pulse,
    _add_movement_onset_state_offset_random_components,
    _add_process_epsilon_calibrated_random_pulse,
    _add_process_epsilon_random_pulse,
    _add_target_aligned_lateral_calibrated_random_pulse,
    _calibrated_initial_amount,
    _calibrated_timing_basis,
    _calibrated_timing_indexed_amounts,
    _closed_loop_amplitudes_by_timing,
    _closed_loop_table,
    _closed_loop_table_reach_m,
    _command_input_direction_pulse,
    _controller_visible_component_amounts,
    _controller_visible_timing_starts,
    _expand_to_rank,
    _family_mask,
    _movement_start_index,
    _offset_initial_random_components,
    _plant_timing_starts,
    _process_epsilon_sensitivity_table,
    _pulse_tensor_from_start,
    _random_amplitude_level,
    _random_pulse_tensor,
    _random_sign,
    _randomized_payload_width,
    _sample_pulse_start,
    _target_aligned_lateral_direction_pulse,
    _target_peak_delta_x_m,
    _trial_reach_length_m_from_peak,
    _zero_graph_payload,
    add_zero_graph_channel_inputs,
    apply_training_perturbation_mixture,
)

from rlrmp.train.broad_epsilon_training import (
    _PgdAscentResult,
    _batch_shape,
    _broad_epsilon_l2_radius,
    _broad_epsilon_pgd_trust_radius,
    _broadcast_finite_policy_params_to_batch,
    _ensure_broad_epsilon_input,
    _epsilon_energy_per_trial,
    _epsilon_time_mask,
    _expand_radius,
    _finite_policy_epsilon_from_rollout,
    _finite_policy_tree_norm,
    _flattened_per_trial_norm,
    _flattened_per_trial_safe_norm,
    _mask_finite_policy_params,
    _normalize_flattened_per_trial,
    _project_flattened_per_trial_l2_ball,
    _resolve_sisu_condition_input,
    _run_broad_epsilon_pgd_ascent,
    _run_finite_broad_epsilon_pgd_inner_maximizer,
    _set_input,
    _shared_policy_time_mask,
    _sisu_condition_values,
    _trial_reach_length_m,
    _trial_target_position_m,
    _zero_finite_policy_params,
    run_broad_epsilon_pgd_inner_maximizer,
)

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
from feedbax import AbstractTask, TaskTrialSpec, WhereDict
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
)
from jaxtyping import PRNGKeyArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlrmp.data_products.broad_epsilon import (
    BROAD_EPSILON_PRODUCT_ROLE,
    BROAD_EPSILON_PRODUCT_SCHEMA_VERSION,
    load_broad_epsilon_anchors,
)
from rlrmp.data_products.calibration import (
    CALIBRATION_PRODUCT_RELPATH,
    CALIBRATION_PRODUCT_ROLE,
    CALIBRATION_PRODUCT_SCHEMA_VERSION,
    load_open_loop_calibration,
    load_perturbation_calibration_defaults,
)
from rlrmp.data_products.envelope import consumed_identity_from_loader
from rlrmp.model.feedbax_channel_adapters import (
    additive_channel_payload_dim,
    additive_channel_provenance,
    materialize_additive_channel_adapters_on_graph,
)
from rlrmp.model.feedback_descriptors import (
    COMPONENT_FORCE_FILTER,
    resolve_controller_feedback_view,
)
from rlrmp.model.cs_lss_gru import (
    CS_H0_CONTEXT_INPUT,
    FINITE_EPSILON_POLICY_GRAPH_COMPONENT,
    FINITE_EPSILON_POLICY_NODE_LABEL,
)
from rlrmp.runtime.params_models import register_params_model
from rlrmp.train.closed_loop_finite_adversary import (
    AFFINE_POLICY,
    FINITE_POLICY_BIAS_INPUT,
    FINITE_POLICY_GAINS_INPUT,
    LINEAR_NO_BIAS_POLICY,
    finite_policy_step_epsilon,
    target_centered_full_state_features,
    zero_finite_affine_policy,
    zero_finite_linear_no_bias_policy,
)

from rlrmp.train.training_configs import *  # noqa: F403


# Historical replay/provenance value only. New living rows must pass any radius
# explicitly with caller-owned provenance instead of inheriting this constant.


# The broad-epsilon per-level closed-loop budget anchors are no longer baked here.
# They are adopted from their analytical game-card / adversary-equivalence sources
# (cb98e58 moderate, a7dad8a strong), persisted as a governed data product under
# results/ea6ccb4/data_products/, and loaded fail-closed by identity via
# rlrmp.data_products.broad_epsilon.load_broad_epsilon_anchors(). See issue ea6ccb4.


register_perturbation_training_params_models()


class TargetRelativeMultiTargetTrainingTaskAdapter(AbstractTask):
    """Rewrite static target trials and expose controller-visible target input."""

    task: object
    config: Any

    def __getattr__(self, name: str):
        return getattr(self.task, name)

    @property
    def loss_func(self):
        return self.task.loss_func

    @property
    def n_steps(self) -> int:
        return int(self.task.n_steps)

    @property
    def seed_validation(self) -> int:
        return int(self.task.seed_validation)

    @property
    def intervention_specs(self):
        return self.task.intervention_specs

    @property
    def input_dependencies(self):
        return self.task.input_dependencies

    def add_input(self, name: str, input_fn, exist_ok: bool = True):
        return eqx.tree_at(
            lambda adapter: adapter.task,
            self,
            self.task.add_input(name, input_fn, exist_ok=exist_ok),
        )

    def get_train_trial(self, key: PRNGKeyArray, batch_info=None) -> TaskTrialSpec:
        return self.get_train_trial_with_intervenor_params(key, batch_info)

    def get_train_trial_with_intervenor_params(
        self,
        key: PRNGKeyArray,
        batch_info=None,
    ) -> TaskTrialSpec:
        base = self.task.get_train_trial_with_intervenor_params(key, batch_info)
        return apply_training_target_distribution(base, self.config, key)

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        del key
        return apply_validation_target_distribution(self.task.validation_trials, self.config)

    @property
    def n_validation_trials(self) -> int:
        return len(
            TargetRelativeMultiTargetTrainingConfig.from_payload(self.config).validation_targets_m
        )

    def validation_plots(self, states, trial_specs=None):
        return self.task.validation_plots(states, trial_specs=trial_specs)

    @property
    def validation_trials(self) -> TaskTrialSpec:
        return apply_validation_target_distribution(self.task.validation_trials, self.config)


class BroadFullStateEpsilonTrainingTaskAdapter(AbstractTask):
    """Inject randomized full-state C&S epsilon after target sampling."""

    task: object
    config: Any

    def __getattr__(self, name: str):
        return getattr(self.task, name)

    @property
    def loss_func(self):
        return self.task.loss_func

    @property
    def n_steps(self) -> int:
        return int(self.task.n_steps)

    @property
    def seed_validation(self) -> int:
        return int(self.task.seed_validation)

    @property
    def intervention_specs(self):
        return self.task.intervention_specs

    @property
    def input_dependencies(self):
        return self.task.input_dependencies

    def add_input(self, name: str, input_fn, exist_ok: bool = True):
        return eqx.tree_at(
            lambda adapter: adapter.task,
            self,
            self.task.add_input(name, input_fn, exist_ok=exist_ok),
        )

    def get_train_trial(self, key: PRNGKeyArray, batch_info=None) -> TaskTrialSpec:
        return self.get_train_trial_with_intervenor_params(key, batch_info)

    def get_train_trial_with_intervenor_params(
        self,
        key: PRNGKeyArray,
        batch_info=None,
    ) -> TaskTrialSpec:
        key_base, key_epsilon = jr.split(key)
        base = self.task.get_train_trial_with_intervenor_params(key_base, batch_info)
        return apply_broad_epsilon_training(base, self.config, key_epsilon)

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        return self.task.get_validation_trials(key)

    @property
    def n_validation_trials(self) -> int:
        return int(self.task.n_validation_trials)

    def validation_plots(self, states, trial_specs=None):
        return self.task.validation_plots(states, trial_specs=trial_specs)

    @property
    def validation_trials(self) -> TaskTrialSpec:
        return self.task.validation_trials


class FixedTargetPerturbationTrainingTaskAdapter(AbstractTask):
    """Apply fixed-target perturbation mixture and validation bins to a task."""

    task: object
    config: Any
    validation_bin: str | None = None

    def __getattr__(self, name: str):
        return getattr(self.task, name)

    @property
    def loss_func(self):
        return self.task.loss_func

    @property
    def n_steps(self) -> int:
        return int(self.task.n_steps)

    @property
    def seed_validation(self) -> int:
        return int(self.task.seed_validation)

    @property
    def intervention_specs(self):
        return self.task.intervention_specs

    @property
    def input_dependencies(self):
        return self.task.input_dependencies

    def add_input(self, name: str, input_fn, exist_ok: bool = True):
        return eqx.tree_at(
            lambda adapter: adapter.task,
            self,
            self.task.add_input(name, input_fn, exist_ok=exist_ok),
        )

    def get_train_trial(self, key: PRNGKeyArray, batch_info=None) -> TaskTrialSpec:
        return self.task.get_train_trial(key, batch_info)

    def get_train_trial_with_intervenor_params(
        self,
        key: PRNGKeyArray,
        batch_info=None,
    ) -> TaskTrialSpec:
        key_trial, key_pert = jr.split(key)
        base = self.task.get_train_trial_with_intervenor_params(key_trial, batch_info)
        return apply_training_perturbation_mixture(base, self.config, key_pert, batch_info)

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        base = self.task.get_validation_trials(key)
        return apply_validation_bin(base, self.config, self.validation_bin or "nominal")

    @property
    def n_validation_trials(self) -> int:
        return int(self.task.n_validation_trials)

    def validation_plots(self, states, trial_specs=None):
        return self.task.validation_plots(states, trial_specs=trial_specs)

    @property
    def validation_trials(self) -> TaskTrialSpec:
        return apply_validation_bin(
            self.task.validation_trials,
            self.config,
            self.validation_bin or "nominal",
        )


def consumed_calibration_budget_identities(
    *,
    calibration_consumed: bool,
    broad_epsilon_consumed: bool,
) -> list[dict[str, str]]:
    """Return consumed data-product identities for the emitted run spec.

    Each entry is a ``{role, schema, hash}`` record snapshotting the typed
    identity of a calibration/budget data product the run consumes at runtime.
    The open-loop calibration product is consumed whenever calibrated-timing
    amplitude wiring is active; the broad-epsilon budget-anchor product is
    consumed whenever a broad full-state epsilon lane (random or PGD) is active.
    """

    identities: list[dict[str, str]] = []
    if calibration_consumed:
        identities.append(
            consumed_identity_from_loader(
                load_product=load_open_loop_calibration,
                role=CALIBRATION_PRODUCT_ROLE,
                schema=CALIBRATION_PRODUCT_SCHEMA_VERSION,
            )
        )
    if broad_epsilon_consumed:
        identities.append(
            consumed_identity_from_loader(
                load_product=load_broad_epsilon_anchors,
                role=BROAD_EPSILON_PRODUCT_ROLE,
                schema=BROAD_EPSILON_PRODUCT_SCHEMA_VERSION,
            )
        )
    return identities


def make_memoryless_policy_adversary(
    config: Any,
    *,
    key: PRNGKeyArray,
) -> MemorylessFullStateEpsilonPolicy:
    """Initialize the memoryless full-state epsilon adversary policy."""

    cfg = PolicyFullStateEpsilonTrainingConfig.from_payload(config)
    return MemorylessFullStateEpsilonPolicy(
        state_feature_dim=cfg.state_feature_dim,
        epsilon_dim=cfg.epsilon_dim,
        width=cfg.width,
        depth=cfg.depth,
        key=key,
    )


def make_policy_adversary(
    config: Any,
    *,
    key: PRNGKeyArray,
    horizon: int | None = None,
) -> Any:
    """Initialize the configured policy adversary."""

    cfg = PolicyFullStateEpsilonTrainingConfig.from_payload(config)
    if cfg.policy_class == POLICY_ADVERSARY_MEMORYLESS_MLP:
        return make_memoryless_policy_adversary(cfg, key=key)
    del key
    if horizon is None:
        raise ValueError("Finite policy adversaries require an explicit horizon.")
    if cfg.policy_class == LINEAR_NO_BIAS_POLICY:
        return zero_finite_linear_no_bias_policy(
            horizon=int(horizon),
            feature_dim=int(cfg.state_feature_dim),
            epsilon_dim=int(cfg.epsilon_dim),
        )
    if cfg.policy_class == AFFINE_POLICY:
        return zero_finite_affine_policy(
            horizon=int(horizon),
            feature_dim=int(cfg.state_feature_dim),
            epsilon_dim=int(cfg.epsilon_dim),
        )
    raise ValueError(f"unknown policy adversary policy_class {cfg.policy_class!r}")


def make_broad_epsilon_pgd_pre_step(config: Any) -> Callable | None:
    """Return a Feedbax pre-step hook for training-time broad-epsilon PGD."""

    cfg = PgdFullStateEpsilonTrainingConfig.from_payload(config)
    if not cfg.enabled:
        return None

    def pre_step_fn(task, model, trial_specs, loss_func, keys_model):
        specs, _ = run_broad_epsilon_pgd_inner_maximizer(
            task,
            model,
            trial_specs,
            loss_func,
            keys_model,
            cfg,
            return_diagnostics=False,
        )
        return specs

    return pre_step_fn


class PolicyAdversaryPreStep(eqx.Module):
    """Feedbax pre-step hook carrying policy weights as dynamic JAX leaves."""

    policy: Any
    config: Any = eqx.field(static=True)

    def __call__(self, task, model, trial_specs, loss_func, keys_model):
        del loss_func
        updated, _diagnostics = policy_adversary_trial_specs(
            self.policy,
            task,
            model,
            trial_specs,
            keys_model,
            self.config,
            stop_gradient_epsilon=True,
        )
        return updated


def make_policy_adversary_pre_step(policy: Any, config: Any) -> PolicyAdversaryPreStep:
    """Return a stable PyTree pre-step hook for learned policy-adversary training."""

    return PolicyAdversaryPreStep(policy=policy, config=config)


def policy_adversary_trial_specs(
    policy: Any,
    task: Any,
    model: Any,
    trial_specs: TaskTrialSpec,
    keys_model: Any,
    config: Any,
    *,
    stop_gradient_epsilon: bool = False,
) -> tuple[TaskTrialSpec, dict[str, jnp.ndarray]]:
    """Apply a learned policy adversary and return low-overhead diagnostics."""

    cfg = PolicyFullStateEpsilonTrainingConfig.from_payload(config)
    if not cfg.enabled:
        return trial_specs, {}
    specs = _ensure_broad_epsilon_input(trial_specs, epsilon_dim=cfg.epsilon_dim)
    base_epsilon = jnp.asarray(specs.inputs["epsilon"])
    clean_states = task.eval_trials(model, specs, keys_model)
    raw_delta = _policy_adversary_raw_delta(policy, clean_states, specs, cfg)
    time_mask = _epsilon_time_mask(specs, base_epsilon, cfg.movement_epoch_only)
    radius = _broad_epsilon_l2_radius(specs, cfg).astype(base_epsilon.dtype)
    delta = _project_flattened_per_trial_l2_ball(raw_delta * time_mask, radius) * time_mask
    if stop_gradient_epsilon:
        delta = jax.lax.stop_gradient(delta)
    updated = _set_input(specs, "epsilon", base_epsilon + delta)
    diagnostics = policy_adversary_projection_diagnostics(delta, radius, mode=cfg.mode)
    return updated, diagnostics


def policy_adversary_objective(
    policy: Any,
    task: Any,
    model: Any,
    trial_specs: TaskTrialSpec,
    loss_func: Any,
    keys_model: Any,
    config: Any,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return the adversary maximization objective and scalar diagnostics."""

    cfg = PolicyFullStateEpsilonTrainingConfig.from_payload(config)
    updated, diagnostics = policy_adversary_trial_specs(
        policy,
        task,
        model,
        trial_specs,
        keys_model,
        cfg,
    )
    states = task.eval_trials(model, updated, keys_model)
    losses = loss_func(states, updated, model)
    controller_loss = jnp.asarray(losses.total)
    energy = diagnostics.get("epsilon_energy_mean", jnp.asarray(0.0, dtype=controller_loss.dtype))
    stabilizer = (
        jnp.asarray(cfg.energy_penalty_gamma, dtype=controller_loss.dtype) * energy
        if cfg.mode == POLICY_ADVERSARY_ENERGY_MODE
        else jnp.asarray(0.0, dtype=controller_loss.dtype)
    )
    objective = controller_loss - stabilizer
    diagnostics = {
        **diagnostics,
        "controller_loss": controller_loss,
        "adversary_objective": objective,
        "energy_penalty_gamma": jnp.asarray(cfg.energy_penalty_gamma, dtype=jnp.float32),
        "stabilizer_term": stabilizer,
        "mode_is_energy": jnp.asarray(cfg.mode == POLICY_ADVERSARY_ENERGY_MODE),
    }
    return objective, diagnostics


def _policy_adversary_raw_delta(
    policy: Any,
    clean_states: Any,
    trial_specs: TaskTrialSpec,
    cfg: PolicyFullStateEpsilonTrainingConfig,
) -> jnp.ndarray:
    mechanics = jnp.asarray(clean_states.mechanics.vector)[..., : cfg.state_feature_dim]
    if cfg.policy_class == POLICY_ADVERSARY_MEMORYLESS_MLP:
        return _memoryless_policy_sequence(policy, mechanics)
    target = _trial_target_position_m(trial_specs)
    state_features = target_centered_full_state_features(mechanics, target_position=target)
    return policy(state_features)


def policy_adversary_projection_diagnostics(
    delta: jnp.ndarray,
    radius: jnp.ndarray,
    *,
    mode: str,
) -> dict[str, jnp.ndarray]:
    """Return scalar projection diagnostics for a policy epsilon sequence."""

    delta_norm = _flattened_per_trial_norm(delta).astype(radius.dtype)
    energy = jnp.sum(jnp.square(delta), axis=tuple(range(max(delta.ndim - 2, 0), delta.ndim)))
    ratio = delta_norm / jnp.maximum(radius, jnp.asarray(1e-12, dtype=radius.dtype))
    boundary = ratio >= jnp.asarray(1.0 - 1e-4, dtype=ratio.dtype)
    return {
        "radius_mean": jnp.mean(radius),
        "radius_max": jnp.max(radius),
        "epsilon_norm_mean": jnp.mean(delta_norm),
        "epsilon_norm_max": jnp.max(delta_norm),
        "epsilon_norm_radius_ratio_mean": jnp.mean(ratio),
        "epsilon_norm_radius_ratio_max": jnp.max(ratio),
        "epsilon_energy_mean": jnp.mean(energy),
        "epsilon_energy_max": jnp.max(energy),
        "boundary_fraction": jnp.mean(boundary.astype(radius.dtype)),
        "mode_is_plain": jnp.asarray(mode == POLICY_ADVERSARY_PLAIN_MODE),
    }


def _memoryless_policy_sequence(
    policy: MemorylessFullStateEpsilonPolicy,
    state_features: jnp.ndarray,
) -> jnp.ndarray:
    flat = state_features.reshape((-1, state_features.shape[-1]))
    flat_epsilon = eqx.filter_vmap(policy)(flat)
    return flat_epsilon.reshape((*state_features.shape[:-1], flat_epsilon.shape[-1]))


def _expand_bool_like(mask: jnp.ndarray | bool, values: jnp.ndarray) -> jnp.ndarray:
    mask_array = jnp.asarray(mask)
    while mask_array.ndim < values.ndim:
        mask_array = jnp.expand_dims(mask_array, axis=-1)
    return mask_array


def _sample_sisu_budget_conditioning_input(
    trial_specs: TaskTrialSpec,
    config: PgdFullStateEpsilonTrainingConfig,
    key_source: Any,
) -> TaskTrialSpec:
    """Sample per-trial SISU levels when PGD uses a SISU budget schedule."""

    if config.budget_schedule != BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE:
        return trial_specs
    input_name = _resolve_sisu_condition_input(trial_specs, config)
    current = jnp.asarray(trial_specs.inputs[input_name])
    key = _extract_prng_key(key_source)
    levels = jnp.asarray(config.sisu_levels, dtype=current.dtype)
    probabilities = jnp.asarray(
        _sisu_level_probabilities(config.sisu_levels, config.sisu_exact_zero_mass),
        dtype=current.dtype,
    )
    sampled = jr.choice(key, levels, shape=_batch_shape(trial_specs), p=probabilities)
    sampled = jnp.broadcast_to(_expand_radius(sampled, current.ndim), current.shape)
    return _set_input(trial_specs, input_name, sampled)


def _extract_prng_key(key_source: Any) -> PRNGKeyArray:
    if hasattr(key_source, "shape") and tuple(key_source.shape) == (2,):
        return key_source
    for leaf in jax.tree.leaves(key_source):
        if hasattr(leaf, "shape") and tuple(leaf.shape) == (2,):
            return leaf
    raise ValueError("SISU-conditioned PGD budget sampling requires a PRNG key.")


def install_perturbation_training_graph_adapters(
    model: Any,
    *,
    force_filter_feedback: bool = False,
) -> Any:
    """Install the fixed external additive channel adapters on a C&S GRU graph."""

    return materialize_additive_channel_adapters_on_graph(
        model,
        tuple(graph_adapter_specs(force_filter_feedback=force_filter_feedback).values()),
    )


def apply_training_target_distribution(
    trial_specs: TaskTrialSpec,
    config: Any,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    """Apply one PRNG-driven static target draw from the seen target set."""

    cfg = TargetRelativeMultiTargetTrainingConfig.from_payload(config)
    targets = jnp.asarray(cfg.seen_targets_m, dtype=jnp.float32)
    batch_shape = _batch_shape(trial_specs)
    index = jr.randint(key, batch_shape, 0, targets.shape[0])
    target = targets[index]
    return _with_static_target(trial_specs, target, metadata=None)


def apply_validation_target_distribution(
    trial_specs: TaskTrialSpec,
    config: Any,
) -> TaskTrialSpec:
    """Return validation trials covering original, seen, and held-out targets."""

    cfg = TargetRelativeMultiTargetTrainingConfig.from_payload(config)
    targets = jnp.asarray(cfg.validation_targets_m, dtype=jnp.float32)
    trial_specs = _with_static_target(trial_specs, targets, metadata=None)
    extra = dict(trial_specs.extra or {})
    extra["target_relative_multitarget_bins"] = target_relative_validation_bins(cfg)
    extra["target_relative_input_contract"] = target_relative_input_contract(
        force_filter_feedback=cfg.force_filter_feedback
    )
    return TaskTrialSpec(
        inits=WhereDict(trial_specs.inits),
        inputs=trial_specs.inputs,
        targets=trial_specs.targets,
        intervene=trial_specs.intervene,
        timeline=trial_specs.timeline,
        extra=extra,
    )


def apply_broad_epsilon_training(
    trial_specs: TaskTrialSpec,
    config: Any,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    """Add a per-trial L2-projected C&S epsilon sequence to training inputs."""

    cfg = BroadFullStateEpsilonTrainingConfig.from_payload(config)
    if not cfg.enabled:
        return trial_specs
    if "epsilon" not in trial_specs.inputs:
        zeros = jnp.zeros(
            (*_batch_shape(trial_specs), int(trial_specs.timeline.n_steps), int(cfg.epsilon_dim)),
            dtype=jnp.float32,
        )
        trial_specs = _set_input(trial_specs, "epsilon", zeros)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    if epsilon.shape[-1] != int(cfg.epsilon_dim):
        raise ValueError(
            f"Broad full-state epsilon expects a {int(cfg.epsilon_dim)}D process "
            "epsilon input; "
            f"got trailing dimension {epsilon.shape[-1]}."
        )
    batch_shape = _batch_shape(trial_specs)
    if batch_shape and epsilon.shape[: len(batch_shape)] != batch_shape:
        epsilon = jnp.broadcast_to(epsilon, (*batch_shape, *epsilon.shape[-2:]))
    time_mask = _epsilon_time_mask(trial_specs, epsilon, cfg.movement_epoch_only)
    draws = jr.normal(key, epsilon.shape, dtype=epsilon.dtype) * time_mask
    flat_axes = tuple(range(max(epsilon.ndim - 2, 0), epsilon.ndim))
    norms = jnp.sqrt(jnp.sum(jnp.square(draws), axis=flat_axes))
    radius = _broad_epsilon_l2_radius(trial_specs, cfg).astype(epsilon.dtype)
    while radius.ndim < epsilon.ndim:
        radius = jnp.expand_dims(radius, axis=-1)
    while norms.ndim < epsilon.ndim:
        norms = jnp.expand_dims(norms, axis=-1)
    broad = draws * (radius / jnp.maximum(norms, jnp.asarray(1e-12, dtype=epsilon.dtype)))
    return _set_input(trial_specs, "epsilon", epsilon + broad)


def apply_validation_bin(
    trial_specs: TaskTrialSpec,
    config: Any,
    bin_name: str,
) -> TaskTrialSpec:
    """Apply one deterministic validation perturbation bin."""

    cfg = (
        FixedTargetPerturbationTrainingConfig.from_payload(config)
        if not isinstance(config, FixedTargetPerturbationTrainingConfig)
        else config
    )
    if bin_name == "nominal":
        return _with_perturbation_metadata(
            trial_specs,
            "nominal",
            force_filter_feedback=cfg.force_filter_feedback,
        )
    if bin_name == "mild_combined":
        trial_specs = _apply_single_bin(
            trial_specs,
            cfg,
            "initial_position",
            cfg.combined_amplitude_scale,
        )
        trial_specs = _apply_single_bin(
            trial_specs,
            cfg,
            "command_input",
            cfg.combined_amplitude_scale,
        )
        return _with_perturbation_metadata(
            trial_specs,
            "mild_combined",
            families=("initial_position", "command_input"),
            force_filter_feedback=cfg.force_filter_feedback,
        )
    if bin_name not in (*SINGLE_FAMILY_BINS, *INACTIVE_LEGACY_PERTURBATION_BINS):
        raise ValueError(f"Unknown perturbation validation bin {bin_name!r}.")
    return _apply_single_bin(trial_specs, cfg, bin_name, 1.0)


def validation_bin_manifest(config: Any) -> dict[str, Any]:
    """Return validation-bin metadata for run specs and sidecars."""

    cfg = (
        FixedTargetPerturbationTrainingConfig.from_payload(config)
        if not isinstance(config, FixedTargetPerturbationTrainingConfig)
        else config
    )
    selection_role = (
        "aggregate rollout loss over predeclared held-out perturbation bins selects "
        "checkpoints; analytical action, I/O, and Jacobian metrics are audit-only"
        if cfg.enabled
        else "nominal rollout validation loss selects checkpoints"
    )
    validation_role = (
        "generalized_held_out_perturbation_rollout_loss"
        if cfg.enabled
        else "nominal_rollout_validation_loss"
    )
    return {
        "schema_version": "rlrmp.cs_fixed_target_perturbation_validation_bins.v1",
        "validation_role": validation_role,
        "selection_role": selection_role,
        "nominal_quality_role": (
            "nominal bin remains a reported quality sidecar/gate and is not an "
            "analytical-fidelity selector"
        ),
        "bins": [
            {
                "bin": bin_name,
                "families": _bin_families(bin_name),
                "target_stream_mutated": False,
            }
            for bin_name in VALIDATION_BINS
        ],
        "config": cfg.to_json(),
    }


def target_relative_validation_manifest(config: Any) -> dict[str, Any]:
    """Return target-relative validation-bin metadata for run specs."""

    cfg = TargetRelativeMultiTargetTrainingConfig.from_payload(config)
    return {
        "schema_version": "rlrmp.cs_target_relative_multitarget_validation_bins.v1",
        "validation_role": "target_relative_multitarget_rollout_loss",
        "selection_role": (
            "rollout loss over original-anchor, seen-target, held-out-target, and "
            "perturbation-emphasis bins selects checkpoints; analytical action and "
            "I/O metrics remain audit-only"
        ),
        "target_centered_scoring": "trial_static_target",
        "bins": target_relative_validation_bins(cfg),
        "input_contract": target_relative_input_contract(
            force_filter_feedback=cfg.force_filter_feedback
        ),
        "config": cfg.to_json(),
    }


def _apply_single_bin(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: PerturbationBin,
    amplitude_scale: float,
) -> TaskTrialSpec:
    return _with_perturbation_metadata(
        _apply_single_bin_raw(trial_specs, config, bin_name, amplitude_scale),
        bin_name,
        force_filter_feedback=config.force_filter_feedback,
    )


def _apply_single_bin_raw(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: PerturbationBin,
    amplitude_scale: float,
) -> TaskTrialSpec:
    if bin_name == "initial_position":
        amount = (
            _single_bin_amount(
                trial_specs,
                config,
                "initial_position",
            )
            * amplitude_scale
        )
        if config.calibrated_timing and config.movement_age_timing:
            return _add_movement_onset_state_offset_pulse(
                trial_specs,
                component=1,
                amount=amount,
            )
        return _offset_initial_vector(trial_specs, axis=1, amount=amount)
    if bin_name == "initial_velocity":
        amount = (
            _single_bin_amount(
                trial_specs,
                config,
                "initial_velocity",
            )
            * amplitude_scale
        )
        if config.calibrated_timing and config.movement_age_timing:
            return _add_movement_onset_state_offset_pulse(
                trial_specs,
                component=3,
                amount=amount,
            )
        return _offset_initial_vector(trial_specs, axis=3, amount=amount)
    if bin_name == "process_epsilon":
        return _add_process_epsilon_pulse(
            trial_specs,
            amount=_single_bin_amount(
                trial_specs,
                config,
                "process_epsilon",
            )
            * amplitude_scale,
            start=_deterministic_validation_start(trial_specs, config, "process_epsilon"),
            duration=config.pulse_duration_steps,
        )
    if bin_name == "command_input":
        return _add_graph_channel_pulse(
            trial_specs,
            GRAPH_ADAPTER_SPECS["command_input"],
            amount=_single_bin_amount(
                trial_specs,
                config,
                "command_input",
            )
            * amplitude_scale,
            start=_deterministic_validation_start(trial_specs, config, "command_input"),
            duration=config.pulse_duration_steps,
        )
    if bin_name == "sensory_feedback":
        specs = graph_adapter_specs(force_filter_feedback=config.force_filter_feedback)
        return _add_graph_channel_pulse(
            trial_specs,
            specs["sensory_feedback"],
            amount=_single_bin_amount(
                trial_specs,
                config,
                "sensory_feedback",
            )
            * amplitude_scale,
            start=_deterministic_validation_start(trial_specs, config, "sensory_feedback"),
            duration=(
                config.pulse_duration_steps
                if config.calibrated_timing
                else trial_specs.timeline.n_steps
            ),
        )
    if bin_name == "delayed_observation":
        specs = graph_adapter_specs(force_filter_feedback=config.force_filter_feedback)
        return _add_graph_channel_pulse(
            trial_specs,
            specs["delayed_observation"],
            amount=_single_bin_amount(
                trial_specs,
                config,
                "delayed_observation",
            )
            * amplitude_scale,
            start=_deterministic_validation_start(trial_specs, config, "delayed_observation"),
            duration=(
                config.pulse_duration_steps
                if config.calibrated_timing
                else trial_specs.timeline.n_steps
            ),
        )
    raise ValueError(f"Unsupported perturbation bin {bin_name!r}.")


def _single_bin_amount(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: PerturbationBin,
) -> float | jnp.ndarray:
    if not config.calibrated_timing:
        if bin_name == "initial_position":
            return config.initial_position_offset_m
        if bin_name == "initial_velocity":
            return config.initial_velocity_offset_m_s
        if bin_name == "process_epsilon":
            return config.process_epsilon_scale
        if bin_name == "command_input":
            return config.command_input_pulse_n
        if bin_name == "sensory_feedback":
            return config.sensory_feedback_offset_m
        if bin_name == "delayed_observation":
            return config.delayed_observation_offset_m
    target_peak_delta_x = _target_peak_delta_x_m(trial_specs, config)
    if bin_name == "initial_position":
        return target_peak_delta_x
    if bin_name == "initial_velocity":
        sensitivity = load_open_loop_calibration()["initial_velocity_offset"]["initial_condition"]
        return target_peak_delta_x / sensitivity
    if bin_name == "process_epsilon":
        sensitivity = load_open_loop_calibration()["process_epsilon_force_state_xy"][
            TIMING_LABELS_PLANT[0]
        ]
        return target_peak_delta_x / sensitivity
    if bin_name == "command_input":
        if _calibration_uses_closed_loop(config, "command_input_pulse"):
            reach_scale = _trial_reach_length_m(trial_specs) / jnp.asarray(
                _closed_loop_table_reach_m(config),
                dtype=jnp.float32,
            )
            amount = _closed_loop_amplitudes_by_timing(
                config,
                family="command_input_pulse",
                timing_labels=(TIMING_LABELS_PLANT[0],),
                component="random_force_pulse_cardinal_basis",
                reducer="mean",
            )[0]
            return amount * reach_scale
        sensitivity = load_open_loop_calibration()["command_input_pulse"][TIMING_LABELS_PLANT[0]]
        return target_peak_delta_x / sensitivity
    if bin_name == "sensory_feedback" and _calibration_uses_closed_loop(
        config,
        "sensory_feedback_offset",
    ):
        reach_scale = _trial_reach_length_m(trial_specs) / jnp.asarray(
            _closed_loop_table_reach_m(config),
            dtype=jnp.float32,
        )
        amount = _closed_loop_amplitudes_by_timing(
            config,
            family="sensory_feedback_offset",
            timing_labels=(TIMING_LABELS_CONTROLLER_VISIBLE[0],),
            component="position",
            axis="x",
        )[0]
        return amount * reach_scale
    if bin_name in {"sensory_feedback", "delayed_observation"}:
        return target_peak_delta_x
    raise ValueError(f"Unsupported perturbation bin {bin_name!r}.")


def _cycle_amplitude(
    index: jnp.ndarray,
    *,
    single_indices: tuple[int, ...],
    combined_indices: tuple[int, ...],
    cfg: FixedTargetPerturbationTrainingConfig,
) -> jnp.ndarray:
    single = jnp.zeros_like(index, dtype=jnp.float32)
    for value in single_indices:
        single = single + (index == value).astype(jnp.float32)
    combined = jnp.zeros_like(index, dtype=jnp.float32)
    for value in combined_indices:
        combined = combined + (index == value).astype(jnp.float32)
    return single.astype(jnp.float32) + combined.astype(jnp.float32) * float(
        cfg.combined_amplitude_scale
    )


def _offset_initial_vector(
    trial_specs: TaskTrialSpec,
    *,
    axis: int,
    amount: float,
) -> TaskTrialSpec:
    vector = jnp.asarray(trial_specs.inits["mechanics.vector"])
    updated = vector.at[..., axis].add(jnp.asarray(amount, dtype=vector.dtype))
    return eqx.tree_at(lambda ts: ts.inits["mechanics.vector"], trial_specs, updated)


def _add_movement_onset_state_offset_pulse(
    trial_specs: TaskTrialSpec,
    *,
    component: int,
    amount: float | jnp.ndarray,
) -> TaskTrialSpec:
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    batch_shape = _batch_shape(trial_specs)
    component_index = jnp.full(batch_shape, int(component), dtype=jnp.int32)
    pulse = _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        width=epsilon.shape[-1],
        component=component_index,
        amount=jnp.asarray(amount, dtype=epsilon.dtype),
        duration=1,
        start=_movement_start_index(trial_specs, batch_shape=batch_shape),
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, epsilon + pulse)


def _add_process_epsilon_pulse(
    trial_specs: TaskTrialSpec,
    *,
    amount: float | jnp.ndarray,
    start: int | jnp.ndarray,
    duration: int,
) -> TaskTrialSpec:
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    batch_shape = _batch_shape(trial_specs)
    component = jnp.full(batch_shape, 5, dtype=jnp.int32)
    updated = epsilon + _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        width=epsilon.shape[-1],
        component=component,
        amount=jnp.asarray(amount, dtype=epsilon.dtype),
        duration=duration,
        start=jnp.asarray(start, dtype=jnp.int32),
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, updated)


def _add_graph_channel_pulse(
    trial_specs: TaskTrialSpec,
    spec: AdditiveGraphChannelAdapterSpec,
    *,
    amount: float | jnp.ndarray,
    start: int | jnp.ndarray,
    duration: int,
) -> TaskTrialSpec:
    payload = _zero_graph_payload(trial_specs, spec)
    batch_shape = _batch_shape(trial_specs)
    component = jnp.zeros(batch_shape, dtype=jnp.int32)
    updated = payload + _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=payload.shape[-2],
        width=payload.shape[-1],
        component=component,
        amount=jnp.asarray(amount, dtype=payload.dtype),
        duration=duration,
        start=jnp.asarray(start, dtype=jnp.int32),
        dtype=payload.dtype,
    )
    return _set_input(trial_specs, spec.input_key, updated)


def _deterministic_validation_start(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: PerturbationBin,
) -> int | jnp.ndarray:
    if not config.calibrated_timing:
        return 0 if bin_name in CONTROLLER_VISIBLE_TIMED_BINS else config.pulse_start_step
    if bin_name in PLANT_TIMED_BINS:
        start = _plant_timing_starts()[0]
    elif bin_name in CONTROLLER_VISIBLE_TIMED_BINS:
        start = _controller_visible_timing_starts()[0]
    else:
        start = 0
    if not config.movement_age_timing:
        return start
    return _movement_start_index(trial_specs) + jnp.asarray(start, dtype=jnp.int32)


def _with_static_target(
    trial_specs: TaskTrialSpec,
    target: jnp.ndarray,
    *,
    metadata: dict[str, Any] | None,
) -> TaskTrialSpec:
    target_array = jnp.asarray(target)
    n_steps = int(trial_specs.timeline.n_steps)
    batch_shape = target_array.shape[:-1]
    target_sequence = jnp.broadcast_to(
        jnp.expand_dims(target_array, axis=-2),
        (*batch_shape, n_steps, 2),
    )
    loss_target_sequence = _catch_preserving_loss_target_sequence(
        trial_specs,
        target_sequence=target_sequence,
        batch_shape=batch_shape,
        n_steps=n_steps,
    )
    target_spec = trial_specs.targets["mechanics.effector.pos"]
    updated_target_spec = eqx.tree_at(
        lambda spec: spec.value,
        target_spec,
        loss_target_sequence,
    )
    updated_target_spec = jax.tree.map(
        lambda leaf: _broadcast_trial_array(leaf, batch_shape),
        updated_target_spec,
    )
    targets = dict(trial_specs.targets)
    targets["mechanics.effector.pos"] = updated_target_spec
    inits = {
        key: _broadcast_trial_array(value, batch_shape)
        for key, value in dict(trial_specs.inits).items()
    }
    inputs = dict(trial_specs.inputs)
    if "effector_target" in inputs and hasattr(inputs["effector_target"], "pos"):
        inputs["effector_target"] = eqx.tree_at(
            lambda state: state.pos,
            inputs["effector_target"],
            loss_target_sequence,
        )
        inputs["effector_target"] = jax.tree.map(
            lambda leaf: _broadcast_trial_array(leaf, batch_shape),
            inputs["effector_target"],
        )
    if "task" in inputs and hasattr(inputs["task"], "effector_target"):
        task_inputs = inputs["task"]
        if hasattr(task_inputs.effector_target, "pos"):
            task_inputs = eqx.tree_at(
                lambda task: task.effector_target.pos,
                task_inputs,
                loss_target_sequence,
            )
            task_inputs = jax.tree.map(
                lambda leaf: _broadcast_trial_array(leaf, batch_shape),
                task_inputs,
            )
            inputs["task"] = task_inputs
    inputs["target"] = target_sequence
    if CS_H0_CONTEXT_INPUT in inputs:
        inputs[CS_H0_CONTEXT_INPUT] = _target_relative_h0_context(
            trial_specs,
            target_sequence=target_sequence,
            context_dim=int(jnp.asarray(inputs[CS_H0_CONTEXT_INPUT]).shape[-1]),
            batch_shape=batch_shape,
        )
    inputs = {
        key: (
            value
            if key in {"target", "effector_target", "task", CS_H0_CONTEXT_INPUT}
            else _broadcast_trial_input_array(key, value, batch_shape)
        )
        for key, value in inputs.items()
    }
    timeline = jax.tree.map(
        lambda leaf: _broadcast_trial_array(leaf, batch_shape),
        trial_specs.timeline,
    )
    intervene = jax.tree.map(
        lambda leaf: _broadcast_trial_array(leaf, batch_shape),
        trial_specs.intervene,
    )
    extra = _broadcast_trial_extra(trial_specs.extra, batch_shape)
    if metadata is not None:
        extra = {**dict(extra or {}), **metadata}
    return TaskTrialSpec(
        inits=WhereDict(inits),
        inputs=inputs,
        targets=WhereDict(targets),
        intervene=intervene,
        timeline=timeline,
        extra=extra,
    )


def _target_relative_h0_context(
    trial_specs: TaskTrialSpec,
    *,
    target_sequence: jnp.ndarray,
    context_dim: int,
    batch_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Return the first controller-visible target-relative feedback for native h0."""

    init = _initial_lss_physical_state(trial_specs, batch_shape=batch_shape)
    target_delta = target_sequence[..., 0, :] - init[..., 0:2]
    neg_velocity = -init[..., 2:4]
    pieces = [target_delta, neg_velocity]
    if int(context_dim) == 6:
        feedback_view = resolve_controller_feedback_view(
            None,
            feedback_dim=6,
            values=init[..., :6],
            source="cs_perturbation_training_h0_context",
        )
        pieces.append(feedback_view.component(COMPONENT_FORCE_FILTER).values)
    elif int(context_dim) != 4:
        raise ValueError(f"Unsupported h0 context dimension {context_dim}; expected 4 or 6.")
    return jnp.concatenate(pieces, axis=-1)


def _initial_lss_physical_state(
    trial_specs: TaskTrialSpec,
    *,
    batch_shape: tuple[int, ...],
) -> jnp.ndarray:
    init = jnp.asarray(trial_specs.inits["mechanics.vector"])
    if init.shape[: len(batch_shape)] != batch_shape:
        init = _broadcast_trial_array(init, batch_shape)
    if init.shape[-1] < 6:
        raise ValueError("Native h0 context requires at least 6 physical initial-state entries.")
    return init[..., :6]


def _catch_preserving_loss_target_sequence(
    trial_specs: TaskTrialSpec,
    *,
    target_sequence: jnp.ndarray,
    batch_shape: tuple[int, ...],
    n_steps: int,
) -> jnp.ndarray:
    """Return scored target sequence, preserving no-go catch trials if present."""

    catch_mask = _catch_mask_from_trial_contract(trial_specs, batch_shape)
    if catch_mask is None:
        return target_sequence
    init_sequence = _initial_position_sequence(
        trial_specs,
        batch_shape=batch_shape,
        n_steps=n_steps,
        dtype=target_sequence.dtype,
    )
    return jnp.where(
        _expand_to_rank(catch_mask, target_sequence.ndim),
        init_sequence,
        target_sequence,
    )


def _catch_mask_from_trial_contract(
    trial_specs: TaskTrialSpec,
    batch_shape: tuple[int, ...],
) -> jnp.ndarray | None:
    """Return per-trial catch mask from explicit catch metadata or task structure."""

    if trial_specs.extra is not None and "is_catch_trial" in trial_specs.extra:
        return _broadcast_catch_mask(trial_specs.extra["is_catch_trial"], batch_shape)

    task_inputs = _task_inputs_from_trial_inputs(trial_specs.inputs)
    if hasattr(task_inputs, "hold"):
        hold = jnp.asarray(task_inputs.hold)
        if hold.ndim > 0 and hold.shape[-1] == 1:
            hold = jnp.squeeze(hold, axis=-1)
        reduce_axes = tuple(range(len(batch_shape), hold.ndim))
        catch_mask = jnp.all(hold > 0.5, axis=reduce_axes) if reduce_axes else hold > 0.5
        return _broadcast_catch_mask(catch_mask, batch_shape)

    input_key = _declared_go_cue_input_key(trial_specs.extra)
    if input_key is None:
        return None
    inputs = dict(trial_specs.inputs)
    if input_key not in inputs:
        raise ValueError(
            f"Trial metadata declares {input_key!r} as a go-cue input, "
            f"but trial_specs.inputs has keys {sorted(inputs)}."
        )
    return _catch_mask_from_declared_go_input(inputs[input_key], batch_shape)


def _task_inputs_from_trial_inputs(inputs: Mapping[str, Any]) -> Any:
    return inputs["task"] if "task" in inputs else inputs


def _declared_go_cue_input_key(extra: Mapping[str, Any] | None) -> str | None:
    if extra is None:
        return None
    if "go_cue_input_key" in extra:
        return str(extra["go_cue_input_key"])
    roles = extra.get("input_roles")
    if isinstance(roles, Mapping):
        for key, role in roles.items():
            role_name = role.get("role") if isinstance(role, Mapping) else role
            if role_name in {"go_cue", "delayed_go_cue"}:
                return str(key)
    return None


def _catch_mask_from_declared_go_input(
    go_input: Any,
    batch_shape: tuple[int, ...],
) -> jnp.ndarray | None:
    go = jnp.asarray(go_input)
    if go.ndim == 0:
        return None
    if go.ndim > len(batch_shape) and go.shape[-1] == 1:
        go = jnp.squeeze(go, axis=-1)
    reduce_axes = tuple(range(len(batch_shape), go.ndim))
    any_go = jnp.any(go > 0.5, axis=reduce_axes) if reduce_axes else go > 0.5
    catch_mask = jnp.logical_not(any_go)
    return _broadcast_catch_mask(catch_mask, batch_shape)


def _broadcast_catch_mask(catch_mask: Any, batch_shape: tuple[int, ...]) -> jnp.ndarray:
    catch = jnp.asarray(catch_mask, dtype=bool)
    while catch.ndim > len(batch_shape) and catch.shape[-1] == 1:
        catch = jnp.squeeze(catch, axis=-1)
    return jnp.broadcast_to(catch, batch_shape)


def _initial_position_sequence(
    trial_specs: TaskTrialSpec,
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    dtype: Any,
) -> jnp.ndarray:
    """Return the initial effector position broadcast as a time sequence."""

    init_pos = None
    for value in dict(trial_specs.inits).values():
        pos = getattr(value, "pos", None)
        if pos is not None:
            init_pos = jnp.asarray(pos, dtype=dtype)
            break
        if eqx.is_array(value):
            array = jnp.asarray(value, dtype=dtype)
            if array.ndim >= 1 and array.shape[-1] >= 2:
                init_pos = array[..., :2]
                break
    if init_pos is None:
        raise ValueError("Catch-preserving target replacement requires an initial position.")
    init_pos = _broadcast_trial_array(init_pos, batch_shape)
    return jnp.broadcast_to(
        jnp.expand_dims(jnp.asarray(init_pos, dtype=dtype), axis=-2),
        (*batch_shape, int(n_steps), 2),
    )


def _broadcast_trial_array(value: Any, batch_shape: tuple[int, ...]) -> Any:
    if not batch_shape:
        return value
    if not eqx.is_array(value):
        return value
    array = jnp.asarray(value)
    if array.ndim == 0:
        return value
    if array.shape[: len(batch_shape)] == batch_shape:
        return value
    tail = array.shape[-1:] if array.ndim <= 2 else array.shape[-2:]
    try:
        return jnp.broadcast_to(array, (*batch_shape, *tail))
    except ValueError:
        return value


def _broadcast_trial_input_array(key: str, value: Any, batch_shape: tuple[int, ...]) -> Any:
    if key == FINITE_POLICY_GAINS_INPUT:
        return _broadcast_trial_array_with_suffix_rank(value, batch_shape, suffix_rank=3)
    if key == FINITE_POLICY_BIAS_INPUT:
        return _broadcast_trial_array_with_suffix_rank(value, batch_shape, suffix_rank=2)
    return _broadcast_trial_array(value, batch_shape)


def _broadcast_trial_array_with_suffix_rank(
    value: Any,
    batch_shape: tuple[int, ...],
    *,
    suffix_rank: int,
) -> Any:
    if not batch_shape or not eqx.is_array(value):
        return value
    array = jnp.asarray(value)
    if array.ndim < int(suffix_rank):
        return value
    if array.shape[: len(batch_shape)] == batch_shape:
        return value
    suffix = array.shape[-int(suffix_rank) :]
    try:
        return jnp.broadcast_to(array, (*batch_shape, *suffix))
    except ValueError:
        return value


def _broadcast_trial_extra(
    extra: Mapping[str, Any] | None,
    batch_shape: tuple[int, ...],
) -> dict[str, Any] | None:
    """Broadcast array-valued TaskTrialSpec metadata to a rewritten trial bank."""

    if extra is None:
        return None
    if not batch_shape:
        return dict(extra)
    result: dict[str, Any] = {}
    for key, value in dict(extra).items():
        if not eqx.is_array(value):
            result[key] = value
            continue
        array = jnp.asarray(value)
        if array.shape[: len(batch_shape)] == batch_shape:
            result[key] = value
        elif array.ndim <= 1 and array.size == 1:
            result[key] = jnp.broadcast_to(jnp.reshape(array, ()), batch_shape)
        else:
            result[key] = _broadcast_trial_array(value, batch_shape)
    return result


def _with_perturbation_metadata(
    trial_specs: TaskTrialSpec,
    bin_name: str,
    *,
    families: tuple[str, ...] | None = None,
    force_filter_feedback: bool = False,
) -> TaskTrialSpec:
    trial_specs = add_zero_graph_channel_inputs(
        trial_specs,
        force_filter_feedback=force_filter_feedback,
    )
    extra = dict(trial_specs.extra or {})
    extra["perturbation_training_bin"] = bin_name
    extra["perturbation_training_families"] = list(families or _bin_families(bin_name))
    return TaskTrialSpec(
        inits=WhereDict(trial_specs.inits),
        inputs=trial_specs.inputs,
        targets=trial_specs.targets,
        intervene=trial_specs.intervene,
        timeline=trial_specs.timeline,
        extra=extra,
    )


def _bin_families(bin_name: str) -> tuple[str, ...]:
    if bin_name == "nominal":
        return ()
    if bin_name == "mild_combined":
        return MILD_COMBINED_FAMILIES
    return (bin_name,)
