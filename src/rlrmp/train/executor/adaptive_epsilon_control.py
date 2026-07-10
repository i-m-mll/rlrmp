"""Adaptive-epsilon controller updates, damage probes, and chunk execution."""
# ruff: noqa: F401

from __future__ import annotations

from rlrmp.train.executor.cs_supervised import (
    CsSupervisedExternalObjectiveLoss,
    CsSupervisedExternalObjectiveLossService,
    CsSupervisedNativeChunkRecord,
    CsSupervisedNativeRuntime,
    DEFAULT_CHECKPOINT_INTERVAL_BATCHES,
    DEFAULT_DELAYED_GO_CUE_MAX_STEP,
    DEFAULT_DELAYED_GO_CUE_MIN_STEP,
    DEFAULT_DELAYED_P_CATCH_TRIAL,
    DEFAULT_OUTPUT_DIR,
    GradientDiagnosticsState,
    RUN_SPEC_OVERRIDE_CATEGORIES,
    RUN_SPEC_RUNTIME_OVERRIDE_KEYS,
    RunSpecExecutionContext,
    UpdateDiagnosticsState,
    VolumeCommit,
    _BOOLEAN_OPTIONAL_FIELDS,
    _CLI_ALIASES,
    _CLI_HELP,
    _adaptive_epsilon_curriculum_enabled,
    _add_model_field_argument,
    _apply_config_parser_defaults,
    _args_values_from_run_spec,
    _axis_removed_shape,
    _build_config_generated_parser,
    _build_optimizer,
    _checkpoint_writes_by_completed_batch,
    _cli_values_match,
    _combine_history_diagnostic_chunks,
    _commit_volume,
    _config_choices,
    _config_default,
    _cs_model_slot,
    _cs_optimizer_slot,
    _cs_supervised_native_run_id,
    _cs_supervised_native_supported,
    _cs_supervised_resume_slot_transform,
    _diagnostic_series,
    _dict_value,
    _emit_checkpoint_progress,
    _empty_diagnostic_series,
    _explicit_cli_overrides,
    _family_amplitude,
    _find_diagnostics_state,
    _gradient_diagnostics_arrays,
    _gradient_diagnostics_transform,
    _has_time_axis,
    _history_diagnostics_arrays,
    _hps_from_run_spec,
    _initial_training_state,
    _is_bool_annotation,
    _latest_checkpoint_write,
    _latest_loss_scalars,
    _latest_pgd_progress_scalars,
    _latest_scalar,
    _learning_rate_schedule,
    _literal_values,
    _loss_tree_arrays,
    _loss_tree_total_array,
    _materialize_adaptive_epsilon_native_result,
    _materialize_cs_supervised_native_result,
    _materialize_policy_adversary_native_result,
    _native_resume_history_base,
    _optimizer_diagnostic_series,
    _optimizer_diagnostic_series_range,
    _optional_arg_type,
    _prepend_existing_training_diagnostics,
    _pulse_value,
    _resize_diagnostic_series,
    _resize_optimizer_diagnostics_for_batches,
    _resolve_full_train_launch_context,
    _run_adaptive_epsilon_native_from_context,
    _run_cs_supervised_native_from_context,
    _run_full_training_from_context,
    _run_policy_adversary_native_from_context,
    _run_spec_override_category,
    _sanitize_array_name,
    _slice_axis,
    _spec_result_from_execution_context,
    _stitch_training_diagnostic_array,
    _trainable_parameter_tree,
    _tree_global_norm,
    _update_diagnostics_arrays,
    _update_diagnostics_transform,
    _validate_composed_training_spec_payload,
    _where_train,
    build_cs_supervised_native_initial_slots,
    build_parser,
    build_run_spec_execution_context,
    get_model_parameters,
    load_validated_run_spec,
    logger,
    make_delayed_cosine_schedule,
    resolve_run_spec_execution_args,
    run_full_training,
    write_training_diagnostics_sidecar,
)
from rlrmp.train.run_spec_authoring import (
    TRAINING_DIAGNOSTICS_MANIFEST,
    TRAINING_DIAGNOSTICS_NPZ,
    _adversarial_phase,
    _broad_epsilon_pgd_finite_policy_inputs,
    _broad_epsilon_pgd_mechanism,
    _broad_epsilon_pgd_training_enabled,
    _broad_epsilon_training_enabled,
    _checkpoint_metadata,
    _controller_feedback_basis,
    _controller_feedback_descriptors,
    _controller_feedback_dim,
    _delayed_pre_go_auxiliary_terms_metadata,
    _delayed_reach_enabled,
    _fidelity_status,
    _get_dependency_metadata,
    _get_git_metadata,
    _get_gpu_metadata,
    _get_runtime_metadata,
    _initial_hidden_encoder_enabled,
    _initial_hidden_encoder_metadata,
    _json_dumps,
    _loss_spec,
    _nominal_only,
    _optimizer_metadata,
    _perturbation_training_enabled,
    _policy_adversary_policy_class,
    _policy_adversary_training_enabled,
    _run_mode,
    _run_spec_path_for_write,
    _should_write_graph_spec,
    _sisu_conditioned_pgd_input_key,
    _stochastic_runtime_contract,
    _target_relative_multitarget_enabled,
    _task_spec,
    _training_diagnostics_metadata,
    _training_distribution_metadata,
    _training_mode,
    _validation_bins_metadata,
    _write_graph_bundle_for_backend,
    build_game_card_provenance,
    build_graph_bundle,
    build_loss_game_card_provenance,
    build_model_structure_summary,
    build_run_spec,
    build_training_run_graph_spec,
    derive_spec_dir,
    derive_spec_path,
    write_run_spec,
)
import math
import time
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Callable, Literal, NamedTuple, Union, get_args, get_origin
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
import optax
from feedbax import TaskTrialSpec
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.intervene.schedule import TimeSeriesParam
from feedbax.runtime.batch import BatchInfo
from feedbax.runtime.graph import init_state_from_component
from feedbax.runtime.iteration import run_component
from feedbax.tasks import (
    extract_timeseries_params,
    infer_n_steps,
    prepare_inputs,
    set_state_by_path,
    where_key_to_path,
)
from jax_cookbook.tree import array_set as tree_set
from jax_cookbook.tree import filter_spec_leaves
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
    BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
    BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE,
    BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    BROAD_EPSILON_PGD_TRAINING_MODE,
    BROAD_EPSILON_TRAINING_MODE,
    DEFAULT_TARGET_SUPPORT_PROFILE,
    LEGACY_PERTURBATION_TRAINING_MODE,
    PERTURBATION_TRAINING_MODE,
    POLICY_ADVERSARY_MEMORYLESS_MLP,
    POLICY_ADVERSARY_PLAIN_MODE,
    POLICY_ADVERSARY_TRAINING_MODE,
    FINITE_POLICY_BIAS_INPUT,
    FINITE_POLICY_GAINS_INPUT,
    TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
    TARGET_RELATIVE_MULTITARGET_TRAINING_MODE,
    BroadFullStateEpsilonTrainingConfig,
    FixedTargetPerturbationTrainingConfig,
    PgdFullStateEpsilonTrainingConfig,
    PolicyFullStateEpsilonTrainingConfig,
    add_zero_graph_channel_inputs,
    consumed_calibration_budget_identities,
    make_broad_epsilon_pgd_pre_step,
    make_policy_adversary_pre_step,
    _batch_shape,
    policy_adversary_objective,
    run_broad_epsilon_pgd_inner_maximizer,
    target_relative_target_support_config,
    target_relative_validation_manifest,
    validation_bin_manifest,
)
from rlrmp.train.progress import batch_log_every, format_batch_line, should_log_batch
from rlrmp.train.executor.checkpoints import (
    ADAPTIVE_EPSILON_ZERO_ADVERSARY_GAIN_TOLERANCE,
    ADAPTIVE_EPSILON_ZERO_ADVERSARY_STOP_REASON,
    AdaptiveEpsilonState,
    SCHEMA_VERSION,
    TrainingState,
    _atomic_write_json,
    _initial_adaptive_epsilon_zero_guard,
    _load_latest_checkpoint_materialization,
    _normalize_adaptive_epsilon_zero_guard,
    _plain,
    _save_pytree,
    latest_checkpoint_path,
    load_latest_checkpoint as load_latest_checkpoint,
    save_training_checkpoint,
)


class TrainingHistory(eqx.Module):
    """Batch-indexable training history retained for RLRMP training summaries."""

    loss: Any
    loss_validation: Any
    learning_rate: Any
    model_parameters: Any | None = None
    trial_specs: dict[int, Any] = field(default_factory=dict)


def init_training_history(
    loss_func: Any,
    n_batches: int,
    n_replicates: int,
    *,
    ensembled: bool,
    start_batch: int = 0,
    task: Any | None = None,
) -> TrainingHistory:
    """Initialize the RLRMP history shape used by native executor loops."""

    batch_dims = (
        (int(n_batches) - int(start_batch), int(n_replicates))
        if ensembled
        else (int(n_batches) - int(start_batch),)
    )
    validation_loss = task.loss_func if task is not None else loss_func
    return TrainingHistory(
        loss=loss_func.skeleton(batch_dims),
        loss_validation=validation_loss.skeleton(batch_dims),
        learning_rate=jnp.empty(batch_dims),
    )


def _adaptive_epsilon_damage_target(config: Any, batch_index: int) -> float:
    schedule = getattr(config, "damage_schedule")
    start = float(schedule.start)
    peak = float(schedule.peak)
    final = float(schedule.final)
    ramp_batches = int(schedule.ramp_batches)
    anneal_batches = int(schedule.anneal_batches)
    batch = max(0, int(batch_index))
    if ramp_batches > 0 and batch < ramp_batches:
        frac = batch / float(ramp_batches)
        return start + frac * (peak - start)
    if anneal_batches > 0 and batch < ramp_batches + anneal_batches:
        frac = (batch - ramp_batches) / float(anneal_batches)
        cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
        return final + cosine * (peak - final)
    return final


def _adaptive_epsilon_outer_weight(config: Any, batch_index: int) -> float:
    schedule = getattr(config, "outer_adversarial_weight")
    start = float(schedule.start)
    final = float(schedule.final)
    ramp_batches = int(schedule.ramp_batches)
    batch = max(0, int(batch_index))
    if ramp_batches < 1:
        return final
    frac = min(1.0, batch / float(ramp_batches))
    return start + frac * (final - start)


def _initial_adaptive_epsilon_state(
    hps: TreeNamespace,
    *,
    schedule_start_batch: int = 0,
) -> AdaptiveEpsilonState | None:
    if not _adaptive_epsilon_curriculum_enabled(hps):
        return None
    cfg = PgdFullStateEpsilonTrainingConfig.from_payload(hps.broad_epsilon_pgd_training)
    if cfg.soft_energy_lambda is None:
        raise ValueError("Adaptive epsilon curriculum requires a resolved positive energy lambda.")
    return AdaptiveEpsilonState(
        lambda_value=float(cfg.soft_energy_lambda),
        schedule_start_batch=max(0, int(schedule_start_batch)),
    )


def _adaptive_epsilon_schedule_batch(
    adaptive_state: AdaptiveEpsilonState,
    global_batch: int,
) -> int:
    return max(0, int(global_batch) - int(adaptive_state.schedule_start_batch))


def _run_adaptive_epsilon_training_chunk(
    *,
    trainer: optax.GradientTransformation,
    task: Any,
    model: Any,
    optimizer_state: Any,
    adaptive_state: AdaptiveEpsilonState | None,
    hps: TreeNamespace,
    where_train: Callable[[Any], Any],
    key: Any,
    start_batch: int,
    chunk_batches: int,
    log_progress: bool,
) -> tuple[Any, Any, Any, AdaptiveEpsilonState, dict[str, np.ndarray]]:
    """Run one paired clean/adversarial adaptive direct-epsilon training chunk."""

    if chunk_batches < 1:
        raise ValueError("chunk_batches must be positive")
    if adaptive_state is None:
        adaptive_state = _initial_adaptive_epsilon_state(hps)
    if adaptive_state is None:
        raise ValueError("Adaptive epsilon state is required for adaptive training.")

    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
    batch_size = int(hps.batch_size)
    where_train_spec = filter_spec_leaves(model, where_train)
    flat_model, treedef_model = jtu.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jtu.tree_flatten(optimizer_state)

    def _ensemble_in_axis(leaf):
        if _is_replicate_axis_array(leaf, n_replicates):
            return 0
        return None

    flat_model_arr_spec = jt.map(_ensemble_in_axis, flat_model)
    train_step = eqx.filter_vmap(
        _adaptive_epsilon_train_step,
        in_axes=(
            None,
            None,
            None,
            flat_model_arr_spec,
            None,
            0,
            None,
            None,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
        ),
        out_axes=(
            eqx.if_array(0),
            0,
            flat_model_arr_spec,
            eqx.if_array(0),
            eqx.if_array(0),
            eqx.if_array(0),
        ),
    )
    history = init_training_history(
        task.loss_func,
        chunk_batches,
        n_replicates,
        ensembled=True,
        start_batch=0,
        task=task,
    )
    keys = jr.split(key, chunk_batches)
    eval_batch_info = BatchInfo(
        size=batch_size,
        start=jnp.asarray(0),
        current=jnp.asarray(0),
        total=jnp.asarray(hps.n_batches_condition),
    )
    eval_keys = jr.split(
        jr.fold_in(jr.PRNGKey(int(getattr(task, "seed_validation", 0))), 847503),
        n_replicates,
    )
    default_force_filter_feedback = bool(_config_default("force_filter_feedback"))
    model_force_filter_feedback = bool(
        getattr(hps.model, "force_filter_feedback", default_force_filter_feedback)
    )
    perturbation_force_filter_feedback = bool(
        getattr(
            hps.perturbation_training,
            "force_filter_feedback",
            default_force_filter_feedback,
        )
    )
    eval_trial_specs, eval_keys_init, eval_keys_model = eqx.filter_vmap(
        partial(
            _sample_adaptive_epsilon_damage_eval_batch,
            task,
            batch_info=eval_batch_info,
            batch_size=batch_size,
            include_graph_adapter_inputs=_perturbation_training_enabled(hps),
            force_filter_feedback=perturbation_force_filter_feedback,
        )
    )(eval_keys)
    eval_trial_specs_arr_spec = jt.map(_ensemble_in_axis, eval_trial_specs)
    damage_eval_step = eqx.filter_vmap(
        _adaptive_epsilon_damage_eval_step,
        in_axes=(
            None,
            None,
            flat_model_arr_spec,
            None,
            None,
            None,
            eval_trial_specs_arr_spec,
            0,
            0,
            None,
            None,
        ),
        out_axes=eqx.if_array(0),
    )
    progress_every = batch_log_every(int(hps.n_batches_condition))
    chunk_started = time.perf_counter()
    diagnostic_series: dict[str, list[np.ndarray]] = {}

    for local_batch in range(chunk_batches):
        global_batch = start_batch + local_batch
        schedule_batch = _adaptive_epsilon_schedule_batch(adaptive_state, global_batch)
        key_train, key_eval = jr.split(keys[local_batch], 2)
        target_damage = _adaptive_epsilon_damage_target(
            hps.adaptive_epsilon_curriculum,
            schedule_batch,
        )
        outer_weight = _adaptive_epsilon_outer_weight(
            hps.adaptive_epsilon_curriculum,
            schedule_batch,
        )
        batch_info = BatchInfo(
            size=batch_size,
            start=jnp.asarray(0),
            current=jnp.asarray(global_batch),
            total=jnp.asarray(hps.n_batches_condition),
        )
        key_train = jr.split(key_train, n_replicates)
        losses, _trial_specs, flat_model, flat_opt_state, _grads, diagnostics = train_step(
            task,
            task.loss_func,
            batch_info,
            flat_model,
            treedef_model,
            flat_opt_state,
            treedef_opt_state,
            where_train_spec,
            trainer,
            hps.broad_epsilon_pgd_training,
            key_train,
            jnp.asarray(adaptive_state.lambda_value, dtype=jnp.float32),
            jnp.asarray(outer_weight, dtype=jnp.float32),
            model_force_filter_feedback,
            hps.adaptive_epsilon_curriculum.controller_training_mode,
        )
        eval_diagnostics = damage_eval_step(
            task,
            task.loss_func,
            flat_model,
            treedef_model,
            hps.broad_epsilon_pgd_training,
            model_force_filter_feedback,
            eval_trial_specs,
            eval_keys_init,
            eval_keys_model,
            jnp.asarray(adaptive_state.lambda_value, dtype=jnp.float32),
            jnp.asarray(outer_weight, dtype=jnp.float32),
        )
        adaptive_update_damage_raw = float(
            np.asarray(jax.device_get(jnp.mean(eval_diagnostics["adaptive_update_damage_raw"])))
        )
        adaptive_update_clean_loss_total = float(
            np.asarray(
                jax.device_get(jnp.mean(eval_diagnostics["adaptive_update_clean_loss_total"]))
            )
        )
        adaptive_state, update_diagnostics = _update_adaptive_epsilon_state(
            adaptive_state,
            hps.adaptive_epsilon_curriculum,
            batch_index=global_batch,
            target_damage=target_damage,
            measured_damage=adaptive_update_damage_raw,
            measured_clean_loss=adaptive_update_clean_loss_total,
        )
        host_diagnostics = {
            name: np.asarray(jax.device_get(value))
            for name, value in diagnostics.items()
            if eqx.is_array(value) or np.isscalar(value)
        }
        host_diagnostics.update(
            {
                name: np.asarray(jax.device_get(value))
                for name, value in eval_diagnostics.items()
                if eqx.is_array(value) or np.isscalar(value)
            }
        )
        host_diagnostics.update(update_diagnostics)
        host_diagnostics["target_damage"] = np.asarray(target_damage, dtype=np.float32)
        host_diagnostics["outer_weight"] = np.asarray(outer_weight, dtype=np.float32)
        host_diagnostics["lambda_value"] = np.asarray(
            adaptive_state.lambda_value,
            dtype=np.float32,
        )
        host_diagnostics["global_batch"] = np.asarray(global_batch, dtype=np.float32)
        host_diagnostics["schedule_batch"] = np.asarray(schedule_batch, dtype=np.float32)
        _append_adaptive_epsilon_diagnostics(diagnostic_series, host_diagnostics)

        history = eqx.tree_at(
            lambda history: history.loss,
            history,
            tree_set(
                history.loss,
                losses.map(lambda arr: jnp.mean(arr, axis=-1)),
                local_batch,
            ),
        )
        opt_state_for_history = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)
        learning_rate = None
        if (hyperparams := getattr(opt_state_for_history, "hyperparams", None)) is not None:
            learning_rate = float(jax.device_get(hyperparams["learning_rate"]))
            history = eqx.tree_at(
                lambda history: history.learning_rate,
                history,
                history.learning_rate.at[local_batch].set(hyperparams["learning_rate"]),
            )
        if log_progress and should_log_batch(
            global_batch,
            int(hps.n_batches_condition),
            every=progress_every,
        ):
            loss_mean = losses.map(jnp.mean)
            clean_loss = float(
                np.asarray(host_diagnostics["adaptive_update_clean_loss_total"]).mean()
            )
            epsilon_scale = float(
                np.asarray(host_diagnostics["adaptive_update_epsilon_scale_used"]).mean()
            )
            print(
                format_batch_line(
                    "adaptive_epsilon",
                    global_batch,
                    int(hps.n_batches_condition),
                    loss=float(jax.device_get(loss_mean.total)),
                    clean_loss=clean_loss,
                    damage=adaptive_update_damage_raw,
                    epsilon_scale=epsilon_scale,
                    target=target_damage,
                    **{"lambda": float(adaptive_state.lambda_value)},
                    lr=learning_rate if learning_rate is not None else float("nan"),
                    outer=outer_weight,
                    elapsed=time.perf_counter() - chunk_started,
                ),
                flush=True,
            )

    model = jtu.tree_unflatten(treedef_model, flat_model)
    optimizer_state = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)
    states_validation, losses_validation = task.eval_ensemble_with_loss(
        model,
        n_replicates,
        key_eval,
        ensemble_random_trials=True,
    )
    del states_validation
    history = eqx.tree_at(
        lambda history: history.loss_validation,
        history,
        tree_set(
            history.loss_validation,
            losses_validation.map(lambda arr: jnp.mean(arr, axis=-1)),
            chunk_batches - 1,
        ),
    )
    return (
        model,
        history,
        optimizer_state,
        adaptive_state,
        _adaptive_epsilon_diagnostics_arrays(diagnostic_series),
    )


def _adaptive_epsilon_train_step(
    task: Any,
    loss_func: Any,
    batch_info: BatchInfo,
    flat_model: Any,
    treedef_model: Any,
    flat_opt_state: Any,
    treedef_opt_state: Any,
    where_train_spec: Any,
    optimizer: optax.GradientTransformation,
    pgd_config: Any,
    key: Any,
    energy_lambda: Any,
    outer_weight: Any,
    force_filter_feedback: bool,
    controller_training_mode: str,
) -> tuple[Any, Any, Any, Any, Any, dict[str, jnp.ndarray]]:
    from rlrmp.train.adaptive_epsilon_native import _adaptive_epsilon_train_step as native_step

    return native_step(
        task,
        loss_func,
        batch_info,
        flat_model,
        treedef_model,
        flat_opt_state,
        treedef_opt_state,
        where_train_spec,
        optimizer,
        pgd_config,
        key,
        energy_lambda,
        outer_weight,
        force_filter_feedback,
        controller_training_mode,
    )


def _scale_direct_epsilon_trial_specs(
    *,
    clean_specs: "TaskTrialSpec",
    adv_specs: "TaskTrialSpec",
    epsilon_scale: Any,
) -> "TaskTrialSpec":
    clean_epsilon = clean_specs.inputs["epsilon"]
    adv_epsilon = adv_specs.inputs["epsilon"]
    scaled_inputs = dict(adv_specs.inputs)
    scaled_inputs["epsilon"] = clean_epsilon + epsilon_scale * (adv_epsilon - clean_epsilon)
    return replace(adv_specs, inputs=scaled_inputs)


def _sample_adaptive_epsilon_damage_eval_batch(
    task: Any,
    key: Any,
    *,
    batch_info: BatchInfo,
    batch_size: int,
    include_graph_adapter_inputs: bool = False,
    force_filter_feedback: bool = False,
) -> tuple["TaskTrialSpec", Any, Any]:
    """Sample the fixed nominal batch used only for adaptive lambda damage updates."""

    key_trials, key_init, key_model = jr.split(key, 3)
    keys_trials = jr.split(key_trials, batch_size)
    trial_specs = eqx.filter_vmap(
        partial(
            _sample_nominal_trial_with_inactive_interventions,
            task,
            batch_info=batch_info,
        )
    )(keys_trials)
    if include_graph_adapter_inputs:
        trial_specs = add_zero_graph_channel_inputs(
            trial_specs,
            force_filter_feedback=force_filter_feedback,
        )
    return trial_specs, jr.split(key_init, batch_size), jr.split(key_model, batch_size)


def _sample_nominal_trial_with_inactive_interventions(
    task: Any,
    key: Any,
    *,
    batch_info: BatchInfo,
) -> "TaskTrialSpec":
    trial_spec = task.get_train_trial(key, batch_info=batch_info)
    intervention_trial_spec = task.get_train_trial_with_intervenor_params(
        key,
        batch_info=batch_info,
    )
    if not intervention_trial_spec.intervene:
        return trial_spec
    return replace(
        trial_spec,
        intervene=_inactive_interventions(intervention_trial_spec.intervene),
    )


def _inactive_interventions(intervene: Any) -> Any:
    def inactive_params(params: Any) -> Any:
        if not hasattr(params, "active"):
            return params
        active = getattr(params, "active")
        if isinstance(active, TimeSeriesParam):
            inactive = TimeSeriesParam(jnp.zeros_like(jnp.asarray(active.value), dtype=bool))
        else:
            inactive = jnp.zeros_like(jnp.asarray(active), dtype=bool)
        return eqx.tree_at(lambda item: item.active, params, inactive)

    return {label: inactive_params(params) for label, params in intervene.items()}


@eqx.filter_jit
def _adaptive_epsilon_damage_eval_step(
    task: Any,
    loss_func: Any,
    flat_model: Any,
    treedef_model: Any,
    pgd_config: Any,
    force_filter_feedback: bool,
    trial_specs: "TaskTrialSpec",
    keys_init: Any,
    keys_model: Any,
    energy_lambda: Any,
    epsilon_scale: Any,
) -> dict[str, jnp.ndarray]:
    """Measure paired clean/adversarial damage on the fixed nominal update batch."""

    model = jtu.tree_unflatten(treedef_model, flat_model)
    trial_specs = add_zero_graph_channel_inputs(
        trial_specs,
        force_filter_feedback=force_filter_feedback,
    )
    trial_specs = _with_default_intervention_inputs(model, trial_specs)
    adv_specs, inner_diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        task,
        model,
        trial_specs,
        loss_func,
        keys_model,
        config=pgd_config,
        soft_energy_lambda_override=energy_lambda,
        return_diagnostics=True,
    )
    adv_specs = jt.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        adv_specs,
    )
    applied_specs = _scale_direct_epsilon_trial_specs(
        clean_specs=trial_specs,
        adv_specs=adv_specs,
        epsilon_scale=epsilon_scale,
    )
    init_states = eqx.filter_vmap(lambda _: init_state_from_component(model))(keys_init)
    init_states = eqx.filter_vmap(
        lambda state, trial_spec: _apply_trial_spec_initial_state(model, state, trial_spec)
    )(init_states, trial_specs)
    clean_states = _eval_trial_specs_for_training(
        model,
        trial_specs,
        init_states,
        keys_model,
    )
    adv_states = _eval_trial_specs_for_training(
        model,
        adv_specs,
        init_states,
        keys_model,
    )
    applied_states = _eval_trial_specs_for_training(
        model,
        applied_specs,
        init_states,
        keys_model,
    )
    clean_losses = loss_func(clean_states, trial_specs, model)
    adv_losses = loss_func(adv_states, adv_specs, model)
    applied_losses = loss_func(applied_states, applied_specs, model)
    ratio_eps = jnp.asarray(1e-12, dtype=clean_losses.total.dtype)
    full_strength_damage = adv_losses.total - clean_losses.total
    applied_scaled_damage = applied_losses.total - clean_losses.total
    diagnostics = {
        "adaptive_update_damage_raw": jnp.asarray(full_strength_damage),
        "adaptive_update_full_strength_damage_raw": jnp.asarray(full_strength_damage),
        "adaptive_update_applied_scaled_damage_raw": jnp.asarray(applied_scaled_damage),
        "adaptive_update_damage_ratio_raw": jnp.asarray(
            full_strength_damage / jnp.maximum(clean_losses.total, ratio_eps)
        ),
        "adaptive_update_applied_scaled_damage_ratio_raw": jnp.asarray(
            applied_scaled_damage / jnp.maximum(clean_losses.total, ratio_eps)
        ),
        "adaptive_update_clean_loss_total": jnp.asarray(clean_losses.total),
        "adaptive_update_adversarial_loss_total": jnp.asarray(adv_losses.total),
        "adaptive_update_full_strength_adversarial_loss_total": jnp.asarray(adv_losses.total),
        "adaptive_update_applied_scaled_loss_total": jnp.asarray(applied_losses.total),
        "adaptive_update_energy_lambda_used": jnp.asarray(energy_lambda, dtype=jnp.float32),
        "adaptive_update_epsilon_scale_used": jnp.asarray(epsilon_scale, dtype=jnp.float32),
    }
    diagnostics.update(
        {f"adaptive_update_inner_{name}": value for name, value in inner_diagnostics.items()}
    )
    return diagnostics


def _with_default_intervention_inputs(model: Any, trial_specs: "TaskTrialSpec") -> "TaskTrialSpec":
    if not hasattr(model, "intervention_state_indices"):
        return trial_specs
    intervention_indices = model.intervention_state_indices()
    if not intervention_indices:
        return trial_specs

    inputs = dict(trial_specs.inputs)
    batch_shape = _batch_shape(trial_specs)
    n_steps = int(trial_specs.timeline.n_steps)
    init_state = init_state_from_component(model)
    changed = False
    for label, idx in intervention_indices.items():
        input_key = f"intervene:{label}"
        if input_key in inputs:
            continue
        params = init_state.get(idx)

        def broadcast_leaf(leaf: Any) -> jnp.ndarray:
            arr = jnp.asarray(leaf)
            return jnp.broadcast_to(arr, (*batch_shape, n_steps, *arr.shape))

        inputs[input_key] = jt.map(broadcast_leaf, params)
        changed = True
    if not changed:
        return trial_specs
    return replace(trial_specs, inputs=inputs)


def _apply_trial_spec_initial_state(model: Any, state: Any, trial_spec: Any) -> Any:
    for where_substate, init_substate in trial_spec.inits.items():
        path = where_key_to_path(where_substate)
        state = set_state_by_path(model, state, path, init_substate)

    if trial_spec.intervene:
        intervention_indices = model.intervention_state_indices()
        for label, params in trial_spec.intervene.items():
            if label not in intervention_indices:
                raise ValueError(f"Unknown intervention label '{label}'")
            idx = intervention_indices[label]
            current = state.get(idx)

            def _merge_leaf(p, c):
                if isinstance(p, TimeSeriesParam):
                    return c
                if p is None:
                    return c
                return p

            merged = jt.map(
                _merge_leaf,
                params,
                current,
                is_leaf=lambda x: x is None or isinstance(x, TimeSeriesParam),
            )
            state = _state_set_matching_dtypes(state, idx, merged)

    return model.state_consistency_update(state)


def _eval_trial_specs_for_training(
    model: Any, trial_specs: Any, init_states: Any, keys: Any
) -> Any:
    def _run_trial(trial_spec, init_state, key):
        inputs = prepare_inputs(model, trial_spec.inputs)
        n_steps = infer_n_steps(inputs, getattr(trial_spec, "timeline", None))
        if trial_spec.intervene:
            intervene_inputs = _extract_intervene_inputs(trial_spec.intervene, model)
            if intervene_inputs:
                inputs = {**inputs, **intervene_inputs}
        _, _, state_history = run_component(
            model,
            inputs,
            init_state,
            key=key,
            n_steps=n_steps,
        )
        return jt.map(lambda x: x[1:] if x is not None else x, state_history)

    return eqx.filter_vmap(_run_trial)(trial_specs, init_states, keys)


def _extract_intervene_inputs(intervene: Any, model: Any) -> dict[str, Any]:
    indices = model.intervention_state_indices()
    result = {}
    for label, params in intervene.items():
        if label not in indices:
            continue
        idx = indices[label]
        tv_params = extract_timeseries_params(params, idx.init)
        if tv_params is not None:
            result[f"intervene:{label}"] = tv_params
    return result


def _cast_to_state_dtypes(new_value: Any, current_value: Any) -> Any:
    def _cast_leaf(new_leaf, current_leaf):
        if eqx.is_array(new_leaf) and eqx.is_array(current_leaf):
            if getattr(new_leaf, "dtype", None) != getattr(current_leaf, "dtype", None):
                return jnp.asarray(new_leaf, dtype=current_leaf.dtype)
        return new_leaf

    return jt.map(_cast_leaf, new_value, current_value)


def _state_set_matching_dtypes(state: Any, idx: Any, new_value: Any) -> Any:
    current_value = state.get(idx)
    return state.set(idx, _cast_to_state_dtypes(new_value, current_value))


def _update_adaptive_epsilon_state(
    state: AdaptiveEpsilonState,
    config: Any,
    *,
    batch_index: int,
    target_damage: float,
    measured_damage: float,
    measured_clean_loss: float,
) -> tuple[AdaptiveEpsilonState, dict[str, np.ndarray]]:
    update_cfg = getattr(config, "lambda_update")
    alpha = float(update_cfg.ema_alpha)
    completed_batches = int(batch_index) + 1
    interval = int(update_cfg.interval_batches)
    update_due = completed_batches % interval == 0
    target = float(target_damage)
    ratio_eps = 1e-12
    frozen_for_burn_in = bool(getattr(update_cfg, "freeze_until_burn_in", True)) and target <= 0.0
    if frozen_for_burn_in:
        damage_ema = state.damage_ema
        clean_loss_ema = state.clean_loss_ema
    else:
        damage_ema = (
            float(measured_damage)
            if state.damage_ema is None
            else (1.0 - alpha) * float(state.damage_ema) + alpha * float(measured_damage)
        )
        clean_loss_ema = (
            float(measured_clean_loss)
            if state.clean_loss_ema is None
            else (1.0 - alpha) * float(state.clean_loss_ema) + alpha * float(measured_clean_loss)
        )
    measured_damage_ratio = float(measured_damage) / max(float(measured_clean_loss), ratio_eps)
    damage_ratio_ema = (
        None
        if damage_ema is None or clean_loss_ema is None
        else float(damage_ema) / max(float(clean_loss_ema), ratio_eps)
    )
    signal_for_math = ratio_eps if damage_ratio_ema is None else max(damage_ratio_ema, ratio_eps)
    current_log_damage_ema = (
        math.log(signal_for_math)
        if damage_ratio_ema is not None and signal_for_math > 0.0
        else None
    )
    raw_gain = _adaptive_epsilon_probe_gain(
        previous_log_damage=state.pending_log_damage_ema,
        current_log_damage=current_log_damage_ema,
        lambda_log_step=state.pending_lambda_log_step,
    )
    gain_estimate = _adaptive_epsilon_running_gain(
        previous=state.gain_estimate,
        sample=raw_gain,
        alpha=float(getattr(update_cfg, "gain_ema_alpha", alpha)),
    )
    gain_samples = state.gain_samples + (1 if raw_gain is not None else 0)
    ema_noise_floor = _adaptive_epsilon_running_noise_floor(
        previous=state.ema_noise_floor,
        previous_log_damage=state.last_log_damage_ema,
        current_log_damage=current_log_damage_ema,
        alpha=float(getattr(update_cfg, "gain_ema_alpha", alpha)),
    )
    relative_error = (signal_for_math - target) / max(target, ratio_eps) if target > 0.0 else 0.0
    log_ratio_error = math.log(signal_for_math / max(target, ratio_eps)) if target > 0.0 else 0.0
    deadband = float(update_cfg.deadband_frac)
    hysteresis = getattr(update_cfg, "hysteresis_frac", None)
    update_threshold = deadband
    if (
        hysteresis is not None
        and state.last_lambda_step_sign != 0
        and relative_error * float(state.last_lambda_step_sign) < 0.0
    ):
        update_threshold = max(update_threshold, float(hysteresis))
    lambda_value = float(state.lambda_value)
    updated = False
    log_step = 0.0
    eta = float(update_cfg.eta)
    eta_eff = _adaptive_epsilon_effective_eta(update_cfg, eta=eta, gain_estimate=gain_estimate)
    if (
        update_due
        and not frozen_for_burn_in
        and target > 0.0
        and abs(relative_error) > update_threshold
    ):
        max_log_step = float(update_cfg.max_log_step)
        log_step = max(-max_log_step, min(max_log_step, eta_eff * log_ratio_error))
        lambda_value *= math.exp(log_step)
        lambda_value = max(
            lambda_value,
            _adaptive_epsilon_lambda_min(update_cfg, analytical_seed=state.lambda_value),
        )
        lambda_max = getattr(update_cfg, "lambda_max", None)
        if lambda_max is not None:
            lambda_value = min(lambda_value, float(lambda_max))
        updated = True
    step_sign = 0
    if updated and log_step != 0.0:
        step_sign = 1 if log_step > 0.0 else -1
    lambda_step_count = state.lambda_step_count + (1 if step_sign != 0 else 0)
    lambda_step_alternations = state.lambda_step_alternations
    if step_sign != 0 and state.last_lambda_step_sign != 0:
        lambda_step_alternations += int(step_sign != state.last_lambda_step_sign)
    sign_alternation_fraction = (
        lambda_step_alternations / float(lambda_step_count - 1) if lambda_step_count > 1 else 0.0
    )
    next_state = AdaptiveEpsilonState(
        lambda_value=lambda_value,
        damage_ema=damage_ema,
        clean_loss_ema=clean_loss_ema,
        last_update_batch=int(batch_index) if updated else state.last_update_batch,
        update_count=state.update_count + (1 if updated else 0),
        schedule_start_batch=state.schedule_start_batch,
        zero_adversary_guard=state.zero_adversary_guard,
        gain_estimate=gain_estimate,
        gain_samples=gain_samples,
        pending_lambda_log_step=log_step
        if updated and current_log_damage_ema is not None
        else None,
        pending_log_damage_ema=(
            current_log_damage_ema if updated and current_log_damage_ema is not None else None
        ),
        last_log_damage_ema=(
            current_log_damage_ema
            if current_log_damage_ema is not None
            else state.last_log_damage_ema
        ),
        ema_noise_floor=ema_noise_floor,
        last_lambda_step_sign=step_sign or state.last_lambda_step_sign,
        lambda_step_count=lambda_step_count,
        lambda_step_alternations=lambda_step_alternations,
    )
    return next_state, {
        "damage_ema": np.asarray(np.nan if damage_ema is None else damage_ema, dtype=np.float32),
        "clean_loss_ema": np.asarray(
            np.nan if clean_loss_ema is None else clean_loss_ema,
            dtype=np.float32,
        ),
        "measured_damage_ratio": np.asarray(measured_damage_ratio, dtype=np.float32),
        "damage_ratio_ema": np.asarray(
            np.nan if damage_ratio_ema is None else damage_ratio_ema,
            dtype=np.float32,
        ),
        "target_damage_ratio": np.asarray(target, dtype=np.float32),
        "relative_error": np.asarray(relative_error, dtype=np.float32),
        "log_ratio_error": np.asarray(log_ratio_error, dtype=np.float32),
        "lambda_updated": np.asarray(updated, dtype=bool),
        "lambda_log_step": np.asarray(log_step, dtype=np.float32),
        "lambda_update_eta_eff": np.asarray(eta_eff, dtype=np.float32),
        "update_due": np.asarray(update_due, dtype=bool),
        "update_count": np.asarray(next_state.update_count, dtype=np.float32),
        "burn_in_frozen": np.asarray(frozen_for_burn_in, dtype=bool),
        "deadband_threshold": np.asarray(update_threshold, dtype=np.float32),
        "gain_normalization_enabled": np.asarray(
            bool(getattr(update_cfg, "gain_normalization", False)),
            dtype=bool,
        ),
        "gain_probe_raw": np.asarray(np.nan if raw_gain is None else raw_gain, dtype=np.float32),
        "gain_hat": np.asarray(
            np.nan if next_state.gain_estimate is None else next_state.gain_estimate,
            dtype=np.float32,
        ),
        "gain_samples": np.asarray(next_state.gain_samples, dtype=np.float32),
        "sign_alternation_fraction": np.asarray(sign_alternation_fraction, dtype=np.float32),
        "ema_noise_floor": np.asarray(
            np.nan if next_state.ema_noise_floor is None else next_state.ema_noise_floor,
            dtype=np.float32,
        ),
    }


def _adaptive_epsilon_probe_gain(
    *,
    previous_log_damage: float | None,
    current_log_damage: float | None,
    lambda_log_step: float | None,
) -> float | None:
    if previous_log_damage is None or current_log_damage is None or lambda_log_step is None:
        return None
    if not all(math.isfinite(value) for value in (previous_log_damage, current_log_damage)):
        return None
    if not math.isfinite(lambda_log_step) or abs(lambda_log_step) < 1e-12:
        return None
    gain = -(current_log_damage - previous_log_damage) / lambda_log_step
    if not math.isfinite(gain) or gain <= 0.0:
        return None
    return gain


def _adaptive_epsilon_running_gain(
    *,
    previous: float | None,
    sample: float | None,
    alpha: float,
) -> float | None:
    if sample is None:
        return previous
    if previous is None or not math.isfinite(previous):
        return float(sample)
    return (1.0 - alpha) * float(previous) + alpha * float(sample)


def _adaptive_epsilon_running_noise_floor(
    *,
    previous: float | None,
    previous_log_damage: float | None,
    current_log_damage: float | None,
    alpha: float,
) -> float | None:
    if previous_log_damage is None or current_log_damage is None:
        return previous
    sample = abs(current_log_damage - previous_log_damage)
    if not math.isfinite(sample):
        return previous
    if previous is None or not math.isfinite(previous):
        return sample
    return (1.0 - alpha) * float(previous) + alpha * sample


def _adaptive_epsilon_effective_eta(
    update_cfg: Any,
    *,
    eta: float,
    gain_estimate: float | None,
) -> float:
    if not bool(getattr(update_cfg, "gain_normalization", False)):
        return eta
    if gain_estimate is None or not math.isfinite(gain_estimate) or gain_estimate <= 0.0:
        return eta
    gain = min(
        float(getattr(update_cfg, "gain_max", 8.0)),
        max(float(getattr(update_cfg, "gain_min", 0.25)), float(gain_estimate)),
    )
    return eta / gain


def _adaptive_epsilon_lambda_min(update_cfg: Any, *, analytical_seed: float) -> float:
    explicit = getattr(update_cfg, "lambda_min", None)
    if explicit is not None:
        return float(explicit)
    return max(1e-12, 1e-3 * float(analytical_seed))


def _append_adaptive_epsilon_diagnostics(
    series: dict[str, list[np.ndarray]],
    diagnostics: dict[str, np.ndarray],
) -> None:
    for name, value in diagnostics.items():
        series.setdefault(name, []).append(np.asarray(value))


def _adaptive_epsilon_diagnostics_arrays(
    series: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        f"adaptive_epsilon_{name}": np.stack(values, axis=0)
        for name, values in sorted(series.items())
    }


def _is_replicate_axis_array(leaf: Any, n_replicates: int) -> bool:
    return (
        eqx.is_array(leaf)
        and leaf.ndim > 0
        and int(getattr(leaf, "shape", (0,))[0]) == int(n_replicates)
    )


__all__ = [
    "TrainingHistory",
    "_adaptive_epsilon_damage_eval_step",
    "_adaptive_epsilon_damage_target",
    "_adaptive_epsilon_diagnostics_arrays",
    "_adaptive_epsilon_effective_eta",
    "_adaptive_epsilon_lambda_min",
    "_adaptive_epsilon_outer_weight",
    "_adaptive_epsilon_probe_gain",
    "_adaptive_epsilon_running_gain",
    "_adaptive_epsilon_running_noise_floor",
    "_adaptive_epsilon_schedule_batch",
    "_adaptive_epsilon_train_step",
    "_append_adaptive_epsilon_diagnostics",
    "_apply_trial_spec_initial_state",
    "_cast_to_state_dtypes",
    "_eval_trial_specs_for_training",
    "_extract_intervene_inputs",
    "_inactive_interventions",
    "_initial_adaptive_epsilon_state",
    "_is_replicate_axis_array",
    "_run_adaptive_epsilon_training_chunk",
    "_sample_adaptive_epsilon_damage_eval_batch",
    "_sample_nominal_trial_with_inactive_interventions",
    "_scale_direct_epsilon_trial_specs",
    "_state_set_matching_dtypes",
    "_update_adaptive_epsilon_state",
    "_with_default_intervention_inputs",
    "init_training_history",
]
