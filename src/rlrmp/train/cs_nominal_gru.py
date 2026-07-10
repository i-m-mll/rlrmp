"""Stochastic C&S-fidelity GRU run-spec construction and training.

This module prepares nominal, hold-free C&S-aligned GRU runs for issue
``30f2313``. The default CLI mode writes only the lightweight run spec and
GraphSpec; ``--full-train`` performs the explicitly launched training path.
"""
# ruff: noqa: F401

from __future__ import annotations

from rlrmp.train.executor.adaptive_epsilon_control import (
    TrainingHistory,
    _adaptive_epsilon_damage_eval_step,
    _adaptive_epsilon_damage_target,
    _adaptive_epsilon_diagnostics_arrays,
    _adaptive_epsilon_effective_eta,
    _adaptive_epsilon_lambda_min,
    _adaptive_epsilon_outer_weight,
    _adaptive_epsilon_probe_gain,
    _adaptive_epsilon_running_gain,
    _adaptive_epsilon_running_noise_floor,
    _adaptive_epsilon_schedule_batch,
    _adaptive_epsilon_train_step,
    _append_adaptive_epsilon_diagnostics,
    _apply_trial_spec_initial_state,
    _cast_to_state_dtypes,
    _eval_trial_specs_for_training,
    _extract_intervene_inputs,
    _inactive_interventions,
    _initial_adaptive_epsilon_state,
    _is_replicate_axis_array,
    _run_adaptive_epsilon_training_chunk,
    _sample_adaptive_epsilon_damage_eval_batch,
    _sample_nominal_trial_with_inactive_interventions,
    _scale_direct_epsilon_trial_specs,
    _state_set_matching_dtypes,
    _update_adaptive_epsilon_state,
    _with_default_intervention_inputs,
    init_training_history,
)

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

from rlrmp.train.config_materialization import (
    ADAPTIVE_EPSILON_TRAINING_MODES,
    ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER,
    CS_DELAYED_REACH_TASK_PRESET,
    CS_DELAYED_REACH_TASK_TYPE,
    CS_FEEDBAX_N_STEPS,
    CS_REGULARIZED_NN_HIDDEN,
    CS_STAGE_COUNT,
    DEFAULT_STOCHASTIC_PRESET,
    DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON,
    DELAYED_MOVEMENT_COST_TAIL_MODES,
    DELAYED_REACH_TRAINING_MODE,
    LEGACY_CS_DELAYED_REACH_TASK_TYPE,
    StochasticPreset,
    _adaptive_epsilon_curriculum_config_from_args,
    _apply_smoke_overrides,
    _config_namespace,
    _config_payload_from_args,
    _delayed_reach_contract_from_args,
    _initial_hidden_encoder_config,
    _resolve_auto_bool,
    _training_diagnostics_enabled,
    build_hps,
    cs_nominal_gru_config_from_args,
    stochastic_preset,
)

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
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
from feedbax.runtime.parameter_constraints import project_component_parameters
from feedbax.objectives.loss import AbstractLoss
from feedbax.objectives.service import LossService, LoweredObjective
from feedbax.objectives.spec import ObjectiveExecutionRequirements
from feedbax.tasks import (
    extract_timeseries_params,
    infer_n_steps,
    prepare_inputs,
    set_state_by_path,
    where_key_to_path,
)
from jax_cookbook.tree import array_set as tree_set
from jax_cookbook.tree import filter_spec_leaves

from rlrmp.runtime.jax_config import assert_jax_x64_disabled
from rlrmp.runtime.params_models import register_params_model
from rlrmp.analysis.math.cs_game_card import (
    INIT_POS,
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID,
    TARGET_POS,
    build_canonical_game,
    build_no_integrator_game,
)
from rlrmp.analysis.math.cs_released_simulation import (
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    default_cs_noise_covariances,
)
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig
from rlrmp.model.cs_lss_gru import (
    CS_H0_CONTEXT_DIM,
    CS_H0_ENCODER_INIT,
)
from rlrmp.model.feedback_descriptors import (
    DESCRIPTOR_PAYLOAD_KEY,
    controller_feedback_descriptor_payload,
)
from rlrmp.model.feedbax_graph import (
    EXECUTION_BACKEND,
    GRAPH_PLANT_INTERVENOR_NODE,
    RLRMPFeedbaxGraphBundle,
    build_runtime_rlrmp_feedbax_graph_bundle,
    build_point_mass_sensorimotor_graph_spec,
    write_graph_spec_bundle,
)
from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
)
from rlrmp.io import compact_json_dumps
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.runtime.run_specs import validate_nominal_gru_run_spec
from rlrmp.runtime.training_run_specs import (
    CS_SUPERVISED_METHOD_REF,
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    RLRMP_RUN_SPEC_PAYLOAD_KEY,
    attach_composed_training_specs,
    attach_post_run_provenance,
    assert_runtime_graph_matches_training_spec,
    feedbax_training_run_spec_from_payload,
)
from rlrmp.runtime.spec_migrations import (
    RUN_SPEC_KIND,
    RUN_SPEC_SCHEMA_ID,
    RUN_SPEC_SCHEMA_VERSION,
    accept_rlrmp_spec_payload,
)
from rlrmp.model.stochastic_runtime import (
    graphspec_noise_contract,
    stochastic_runtime_config_from_model,
)
from rlrmp.train.executor.adapters import ChunkKernelAdapter, RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.executor.initial_slots import RlrmpRuntime, split_initial_keys
from rlrmp.train.executor.slots import (
    COMPLETED_BATCHES,
    HISTORY_CHUNK_BYTES,
    MODEL,
    OPTIMIZER,
    PRNG,
    TRAIN_LOSS,
)
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
from rlrmp.train.resume_control import emit_launch_continuation, resolve_launch_continuation
from rlrmp.train.task_model import (
    CS_LSS_PLANT_BACKEND,
    LEGACY_CAUSAL_PLANT_BACKEND,
    setup_task_model_pair,
)
from rlrmp.model.trainable import staged_network_trainable_parts, staged_network_trainable_paths


from rlrmp.train.training_configs import (
    ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
    CS_CONTROL_SCALE,
    CS_POSITION_SCALE,
    CS_VELOCITY_SCALE,
    CsNominalGruConfig,
    DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW,
    ISSUE_ID,
)
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


DEFAULT_EXPERIMENT = ISSUE_ID
DEFAULT_RUN = "cs_stochastic_gru__no_hidden_penalty"

CS_NOMINAL_GRU_PARAMS_REF = "rlrmp.train.cs_nominal_gru"


def register_training_config_params_models(*, replace: bool = True) -> None:
    """Register training params models for run-matrix validation."""

    register_params_model(CS_NOMINAL_GRU_PARAMS_REF, CsNominalGruConfig, replace=replace)
    register_params_model(CS_SUPERVISED_METHOD_REF, CsNominalGruConfig, replace=replace)


register_training_config_params_models()


def resolve_run_spec_args(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Return executable CLI arguments replayed from a modern nominal-GRU run spec.

    The checked-in spec owns run identity, graph identity, checkpoint policy,
    artifact routes, and scientific payload. CLI values may only supply
    runtime-only execution controls such as ``--resume`` and
    ``--stop-after-batches``.
    """

    run_spec_path = getattr(args, "run_spec", None)
    if run_spec_path is None:
        return args
    parser = parser or build_parser()
    payload_path, payload = load_validated_run_spec(run_spec_path)
    return resolve_run_spec_execution_args(
        args,
        run_spec_path=payload_path,
        run_spec=payload,
        parser=parser,
    )


def cs_supervised_update_kernels(payload: Any) -> Mapping[str, Any]:
    """Return Feedbax update kernels for the native C&S supervised method."""

    return {
        "rlrmp.cs_supervised.train_chunk": ChunkKernelAdapter(
            chunk_fn=_cs_supervised_train_chunk,
            reads=(MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES),
            writes=(
                MODEL,
                OPTIMIZER,
                PRNG,
                COMPLETED_BATCHES,
                TRAIN_LOSS,
                HISTORY_CHUNK_BYTES,
            ),
            metric_slots=(TRAIN_LOSS,),
            name="cs-supervised train chunk",
        ).to_kernel(payload)
    }


def _cs_model_from_slot(slot: Any, runtime: CsSupervisedNativeRuntime) -> Any:
    arrays = jt.unflatten(jt.structure(runtime.model_array_template), tuple(slot))
    return eqx.combine(arrays, runtime.pair.model)


def _cs_optimizer_from_slot(slot: Any, runtime: CsSupervisedNativeRuntime) -> Any:
    return jt.unflatten(jt.structure(runtime.optimizer_template), tuple(slot))


def _cs_runtime_model(
    chunk_slots: Mapping[str, Any],
    runtime: CsSupervisedNativeRuntime,
    *,
    completed_batches: int,
) -> Any:
    if runtime.current_model is not None and runtime.current_completed_batches == completed_batches:
        return runtime.current_model
    return _cs_model_from_slot(chunk_slots[MODEL], runtime)


def _cs_runtime_optimizer(
    chunk_slots: Mapping[str, Any],
    runtime: CsSupervisedNativeRuntime,
    *,
    completed_batches: int,
) -> Any:
    if (
        runtime.current_optimizer_state is not None
        and runtime.current_completed_batches == completed_batches
    ):
        return runtime.current_optimizer_state
    return _cs_optimizer_from_slot(chunk_slots[OPTIMIZER], runtime)


def _cs_supervised_train_chunk(
    runtime: RlrmpRuntime,
    payload: Any,
    chunk_slots: Mapping[str, Any],
    coordinate: Any,
) -> Mapping[str, Any]:
    del payload, coordinate
    native = runtime.component("cs_supervised")
    if not isinstance(native, CsSupervisedNativeRuntime):
        raise TypeError("missing cs_supervised native runtime")
    args = native.args
    completed_batches = int(chunk_slots[COMPLETED_BATCHES])
    remaining = int(args.n_train_batches) - completed_batches
    chunk_batches = min(int(args.checkpoint_interval_batches), remaining)
    if runtime.stop_after_batches is not None:
        chunk_batches = min(chunk_batches, int(runtime.stop_after_batches) - completed_batches)
    if chunk_batches < 1:
        chunk_batches = 0
        history_chunk = None
        model = _cs_runtime_model(chunk_slots, native, completed_batches=completed_batches)
        optimizer_state = _cs_runtime_optimizer(
            chunk_slots,
            native,
            completed_batches=completed_batches,
        )
        key_next = chunk_slots[PRNG]
        pgd_diagnostics: dict[str, np.ndarray] = {}
    else:
        key_chunk, key_next = jr.split(chunk_slots[PRNG])
        model_in = _cs_runtime_model(chunk_slots, native, completed_batches=completed_batches)
        optimizer_in = _cs_runtime_optimizer(
            chunk_slots,
            native,
            completed_batches=completed_batches,
        )
        started = time.perf_counter()
        model, history_chunk, optimizer_state = _run_cs_supervised_training_chunk(
            optimizer=native.optimizer,
            task=native.pair.task,
            model=model_in,
            optimizer_state=optimizer_in,
            hps=native.hps,
            where_train=native.where_train,
            key=key_chunk,
            start_batch=completed_batches,
            chunk_batches=chunk_batches,
            log_progress=not bool(args.disable_progress) and not bool(args.quiet_progress),
            log_every=max(1, int(args.log_step)),
            pre_step_fn=native.pre_step_fn,
        )
        duration_seconds = time.perf_counter() - started
        completed = completed_batches + chunk_batches
        pgd_diagnostics = {}
        if _training_diagnostics_enabled(args):
            pgd_diagnostics = _broad_epsilon_pgd_diagnostics_arrays(
                native.pair.task,
                model,
                native.hps,
                key=key_chunk,
                batch_index=completed - 1,
                chunk_batches=chunk_batches,
            )
        native.history = _append_history(native.history, history_chunk)
        native.current_model = model
        native.current_optimizer_state = optimizer_state
        native.current_completed_batches = completed
        state = TrainingState(
            model=model,
            optimizer_state=optimizer_state,
            completed_batches=completed,
            key=key_next,
            history=native.history,
        )
        native.records.append(
            CsSupervisedNativeChunkRecord(
                state=state,
                history_chunk=history_chunk,
                pgd_diagnostics=pgd_diagnostics,
                chunk_batches=chunk_batches,
                duration_seconds=duration_seconds,
            )
        )
    completed = completed_batches + chunk_batches
    train_loss = 0.0
    if history_chunk is not None:
        loss_scalars = _latest_loss_scalars(history_chunk, chunk_batches=chunk_batches)
        if "total" not in loss_scalars:
            raise KeyError("C&S supervised history chunk did not include total loss")
        train_loss = loss_scalars["total"]
    return {
        MODEL: _cs_model_slot(model, native.model_array_template),
        OPTIMIZER: _cs_optimizer_slot(optimizer_state),
        PRNG: key_next,
        COMPLETED_BATCHES: jnp.asarray(completed, dtype=jnp.int32),
        TRAIN_LOSS: float(train_loss),
        HISTORY_CHUNK_BYTES: _history_chunk_bytes(history_chunk),
    }


def _history_chunk_bytes(history_chunk: Any) -> bytes:
    if history_chunk is None:
        return b""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "history_chunk.eqx"
        _save_pytree(path, history_chunk)
        return path.read_bytes()


def _add_config_argument(
    parser: argparse.ArgumentParser,
    *flags: str,
    config_field: str,
    **kwargs: Any,
) -> argparse.Action:
    if "default" not in kwargs:
        kwargs["default"] = _config_default(config_field)
    if "choices" not in kwargs:
        choices = _config_choices(config_field)
        if choices is not None:
            kwargs["choices"] = choices
    return parser.add_argument(*flags, **kwargs)


def main(
    argv: list[str] | None = None,
    *,
    volume_commit: VolumeCommit | None = None,
) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    assert_jax_x64_disabled("C&S nominal GRU training/spec entry", allow_x64=args.allow_x64)
    if args.run_spec is not None:
        context = build_run_spec_execution_context(args, parser=parser)
        if context.args.dry_run:
            result = render_run_spec_execution_dry_run(context)
        elif context.args.full_train:
            result = _run_full_training_from_context(context, volume_commit=volume_commit)
        else:
            result = {
                "run_spec_path": str(context.run_spec_path),
                "run_spec": context.run_spec,
                "validated": True,
            }
    else:
        result = (
            run_full_training(args, volume_commit=volume_commit)
            if args.full_train and not args.dry_run
            else write_run_spec(args)
        )
    print(_json_dumps(result), end="")
    return 0


def render_run_spec_execution_dry_run(context: RunSpecExecutionContext) -> dict[str, Any]:
    """Render the execution plan for a validated spec without writing artifacts."""

    args = context.args
    return {
        "run_spec_path": str(context.run_spec_path),
        "run_spec": context.run_spec,
        "validated": True,
        "would_write": [],
        "would_execute": {
            "entrypoint": "rlrmp.train.cs_nominal_gru._run_full_training_from_context"
            if args.full_train
            else "validate_spec_only",
            "full_train": bool(args.full_train),
            "output_dir": str(args.output_dir),
            "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
            "resume": bool(args.resume),
            "stop_after_batches": args.stop_after_batches,
            "training_diagnostics": bool(args.training_diagnostics),
        },
    }


def _build_trainer(hps: TreeNamespace) -> optax.GradientTransformation:
    """Return the controller optimizer for legacy native-runtime call sites."""

    return _build_optimizer(hps)


def _append_history(history: Any | None, chunk: Any) -> Any:
    if history is None:
        return chunk
    return jt.map(_append_history_leaf, history, chunk, is_leaf=lambda x: x is None)


def _append_history_leaf(left: Any, right: Any) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    if eqx.is_array(left) and eqx.is_array(right):
        if left.ndim == 0 or right.ndim == 0:
            return right
        return jnp.concatenate([left, right], axis=0)
    return right


def _run_cs_supervised_training_chunk(
    *,
    optimizer: optax.GradientTransformation,
    task: Any,
    loss_func: Any | None = None,
    model: Any,
    optimizer_state: Any,
    hps: TreeNamespace,
    where_train: Callable[[Any], Any],
    key: Any,
    start_batch: int,
    chunk_batches: int,
    log_progress: bool,
    log_every: int,
    pre_step_fn: Callable[..., Any] | None,
) -> tuple[Any, Any, Any]:
    """Run one C&S supervised checkpoint-sized chunk through the native executor."""

    if chunk_batches < 1:
        raise ValueError("chunk_batches must be positive")
    n_replicates = int(hps.model.n_replicates)
    batch_size = int(hps.batch_size)
    active_loss = task.loss_func if loss_func is None else loss_func
    where_train_spec = filter_spec_leaves(model, where_train)
    flat_model, treedef_model = jtu.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jtu.tree_flatten(optimizer_state)

    def _ensemble_in_axis(leaf):
        if _is_replicate_axis_array(leaf, n_replicates):
            return 0
        return None

    flat_model_arr_spec = jt.map(_ensemble_in_axis, flat_model)
    train_step = eqx.filter_vmap(
        _supervised_train_step,
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
            0,
            None,
            None,
        ),
        out_axes=(
            eqx.if_array(0),
            0,
            flat_model_arr_spec,
            eqx.if_array(0),
            eqx.if_array(0),
        ),
    )
    history = init_training_history(
        active_loss,
        chunk_batches,
        n_replicates,
        ensembled=True,
        start_batch=0,
        task=task,
    )
    keys = jr.split(key, chunk_batches)
    chunk_started = time.perf_counter()
    key_eval = keys[-1]
    for local_batch in range(chunk_batches):
        global_batch = int(start_batch) + local_batch
        key_train, key_eval = jr.split(keys[local_batch], 2)
        batch_info = BatchInfo(
            size=batch_size,
            start=jnp.asarray(0),
            current=jnp.asarray(global_batch),
            total=jnp.asarray(hps.n_batches_condition),
        )
        key_train = jr.split(key_train, n_replicates)
        losses, _trial_specs, flat_model, flat_opt_state, _grads = train_step(
            task,
            active_loss,
            batch_info,
            flat_model,
            treedef_model,
            flat_opt_state,
            treedef_opt_state,
            where_train_spec,
            optimizer,
            key_train,
            None,
            pre_step_fn,
        )
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
        if (hyperparams := getattr(opt_state_for_history, "hyperparams", None)) is not None:
            history = eqx.tree_at(
                lambda history: history.learning_rate,
                history,
                history.learning_rate.at[local_batch].set(hyperparams["learning_rate"]),
            )
        if log_progress and should_log_batch(
            global_batch,
            int(hps.n_batches_condition),
            every=log_every,
        ):
            loss_mean = losses.map(jnp.mean)
            print(
                format_batch_line(
                    "cs_supervised",
                    global_batch,
                    int(hps.n_batches_condition),
                    loss=float(jax.device_get(loss_mean.total)),
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
    return model, history, optimizer_state


def _optimizer_diagnostics_arrays(
    optimizer_state: Any,
    *,
    start_batches: int = 0,
    completed_batches: int,
) -> dict[str, np.ndarray]:
    """Return scalar optimizer diagnostics for one completed batch range."""

    arrays: dict[str, np.ndarray] = {}
    gradient_state = _find_diagnostics_state(
        optimizer_state,
        GradientDiagnosticsState,
    )
    update_state = _find_diagnostics_state(
        optimizer_state,
        UpdateDiagnosticsState,
    )
    if gradient_state is not None:
        arrays.update(
            _gradient_diagnostics_arrays(
                gradient_state,
                completed_batches,
                start_batches=start_batches,
            )
        )
    if update_state is not None:
        arrays.update(
            _update_diagnostics_arrays(
                update_state,
                completed_batches,
                start_batches=start_batches,
            )
        )
    return arrays


def _diagnostic_series_range(array: Any, start_batches: int, completed_batches: int) -> np.ndarray:
    return _diagnostic_series(array, completed_batches)[start_batches:completed_batches]


def _make_policy_adversary_pre_step(policy: Any, config: Any) -> Callable:
    """Return a stable PyTree hook applying the current learned policy adversary."""

    return make_policy_adversary_pre_step(policy, config)


@eqx.filter_jit
def _supervised_train_step(
    task: Any,
    loss_func: Any,
    batch_info: BatchInfo,
    flat_model: Any,
    treedef_model: Any,
    flat_opt_state: Any,
    treedef_opt_state: Any,
    where_train_spec: Any,
    optimizer: optax.GradientTransformation,
    key: Any,
    loss_reduction_fn: Callable[[Any], Any] | None = None,
    pre_step_fn: Callable[..., Any] | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Run one supervised controller update without the retired Feedbax trainer."""

    key_trials, key_init, key_model = jr.split(key, 3)
    keys_trials = jr.split(key_trials, batch_info.size)
    keys_init = jr.split(key_init, batch_info.size)
    keys_model = jr.split(key_model, batch_info.size)
    trial_specs = eqx.filter_vmap(
        partial(
            task.get_train_trial_with_intervenor_params,
            batch_info=batch_info,
        )
    )(keys_trials)
    model = jtu.tree_unflatten(treedef_model, flat_model)
    if pre_step_fn is not None:
        trial_specs = pre_step_fn(task, model, trial_specs, loss_func, keys_model)
    init_states = eqx.filter_vmap(lambda _: init_state_from_component(model))(keys_init)
    init_states = eqx.filter_vmap(
        lambda state, trial_spec: _apply_trial_spec_initial_state(model, state, trial_spec)
    )(init_states, trial_specs)
    diff_model, static_model = eqx.partition(model, where_train_spec)
    opt_state = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)

    def train_loss(current_diff_model: Any) -> tuple[Any, tuple[Any, Any]]:
        current_model = eqx.combine(current_diff_model, static_model)
        states = _eval_trial_specs_for_training(
            current_model,
            trial_specs,
            init_states,
            keys_model,
        )
        losses = loss_func(states, trial_specs, current_model)
        if loss_reduction_fn is None:
            scalar_loss = losses.total
        else:
            scalar_loss = loss_reduction_fn(losses)
        return scalar_loss, (losses, states)

    (_, (losses, states)), grads = eqx.filter_value_and_grad(
        train_loss,
        has_aux=True,
    )(diff_model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    del states
    model = project_component_parameters(model)
    return losses, trial_specs, jtu.tree_leaves(model), jtu.tree_leaves(opt_state), grads


def _sample_adaptive_epsilon_training_batch(
    task: Any,
    *,
    batch_info: BatchInfo,
    keys_trials: Any,
) -> "TaskTrialSpec":
    """Sample the live stochastic controller-training batch, including interventions."""

    return eqx.filter_vmap(
        partial(
            task.get_train_trial_with_intervenor_params,
            batch_info=batch_info,
        )
    )(keys_trials)


def _weighted_loss_tree(clean_losses: Any, adv_losses: Any, outer_weight: Any) -> Any:
    def combine(clean_value: Any, adv_value: Any) -> Any:
        if eqx.is_array(clean_value) and eqx.is_array(adv_value):
            return (1.0 - outer_weight) * clean_value + outer_weight * adv_value
        return clean_value

    return jt.map(combine, clean_losses, adv_losses)


def _run_policy_adversary_training_chunk(
    *,
    trainer: optax.GradientTransformation,
    task: Any,
    model: Any,
    optimizer_state: Any,
    adversary_policy: Any,
    adversary_optimizer_state: Any,
    adversary_optimizer: optax.GradientTransformation,
    hps: TreeNamespace,
    where_train: Callable[[Any], Any],
    key: Any,
    start_batch: int,
    chunk_batches: int,
    log_progress: bool,
) -> tuple[Any, Any, Any, Any, Any, dict[str, np.ndarray]]:
    """Run a policy-adversary chunk without per-batch trainer re-entry."""

    if chunk_batches < 1:
        raise ValueError("chunk_batches must be positive")
    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
    batch_size = int(hps.batch_size)
    where_train_spec = filter_spec_leaves(model, where_train)
    flat_model, treedef_model = jtu.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jtu.tree_flatten(optimizer_state)

    def _ensemble_in_axis(leaf):
        if eqx.is_array(leaf) and leaf.ndim > 0 and leaf.shape[0] == n_replicates:
            return 0
        return None

    flat_model_arr_spec = jt.map(_ensemble_in_axis, flat_model)
    train_step = eqx.filter_vmap(
        _supervised_train_step,
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
            0,
            None,
            None,
        ),
        out_axes=(
            eqx.if_array(0),
            0,
            flat_model_arr_spec,
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
    progress_every = batch_log_every(int(hps.n_batches_condition))
    chunk_started = time.perf_counter()
    diagnostics: dict[str, np.ndarray] = {}

    for local_batch in range(chunk_batches):
        global_batch = start_batch + local_batch
        key_adversary, key_controller, key_eval = jr.split(keys[local_batch], 3)
        model_for_adversary = jtu.tree_unflatten(treedef_model, flat_model)
        (
            adversary_policy,
            adversary_optimizer_state,
            diagnostics,
        ) = _advance_policy_adversary(
            adversary_policy,
            adversary_optimizer_state,
            adversary_optimizer,
            task,
            model_for_adversary,
            hps,
            key=key_adversary,
            batch_index=global_batch,
        )
        pre_step_fn = _make_policy_adversary_pre_step(
            adversary_policy,
            hps.policy_adversary_training,
        )
        batch_info = BatchInfo(
            size=batch_size,
            start=jnp.asarray(0),
            current=jnp.asarray(local_batch),
            total=jnp.asarray(chunk_batches),
        )
        key_train = jr.split(key_controller, n_replicates)
        losses, _trial_specs, flat_model, flat_opt_state, _grads = train_step(
            task,
            task.loss_func,
            batch_info,
            flat_model,
            treedef_model,
            flat_opt_state,
            treedef_opt_state,
            where_train_spec,
            trainer,
            key_train,
            None,
            pre_step_fn,
        )
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
        if (hyperparams := getattr(opt_state_for_history, "hyperparams", None)) is not None:
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
            print(
                format_batch_line(
                    "policy_adversary",
                    global_batch,
                    int(hps.n_batches_condition),
                    loss=float(jax.device_get(loss_mean.total)),
                    adv=float(np.asarray(diagnostics.get("adversary_objective", np.nan))),
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
        adversary_policy,
        adversary_optimizer_state,
        diagnostics,
    )


def _advance_policy_adversary(
    policy: Any,
    optimizer_state: Any,
    optimizer: optax.GradientTransformation,
    task: Any,
    model: Any,
    hps: TreeNamespace,
    *,
    key: Any,
    batch_index: int,
) -> tuple[Any, Any, dict[str, np.ndarray]]:
    """Run the persistent policy-adversary ascent steps for one controller batch."""

    policy, optimizer_state, diagnostics = _advance_policy_adversary_compiled(
        policy,
        optimizer_state,
        optimizer,
        task,
        model,
        hps,
        key,
        jnp.asarray(batch_index),
    )
    arrays = {
        name: np.asarray(jax.device_get(value))
        for name, value in diagnostics.items()
        if eqx.is_array(value) or np.isscalar(value)
    }
    return policy, optimizer_state, arrays


def _advance_policy_adversary_compiled(
    policy: Any,
    optimizer_state: Any,
    optimizer: optax.GradientTransformation,
    task: Any,
    model: Any,
    hps: TreeNamespace,
    key: Any,
    batch_index: Any,
) -> tuple[Any, Any, dict[str, jnp.ndarray]]:
    from rlrmp.train.policy_adversary_native import (
        _advance_policy_adversary_compiled as native_step,
    )

    return native_step(
        policy,
        optimizer_state,
        optimizer,
        task,
        model,
        hps,
        key,
        batch_index,
    )


def _policy_adversary_batch_objective(
    policy: Any,
    task: Any,
    model: Any,
    hps: TreeNamespace,
    *,
    key: Any,
    batch_index: int,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
    batch_size = int(hps.batch_size)
    batch_info = BatchInfo(
        size=batch_size,
        start=jnp.asarray(0),
        current=jnp.asarray(batch_index),
        total=jnp.asarray(hps.n_batches_condition),
    )
    keys = jr.split(key, n_replicates)
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_axis_array(leaf, n_replicates),
    )
    objectives = []
    diagnostics_by_replicate = []
    for replicate_index, key_replicate in enumerate(keys):
        key_trials, _, key_model = jr.split(key_replicate, 3)
        keys_trials = jr.split(key_trials, batch_size)
        keys_model = jr.split(key_model, batch_size)
        trial_specs = eqx.filter_vmap(
            partial(
                task.get_train_trial_with_intervenor_params,
                batch_info=batch_info,
            )
        )(keys_trials)
        replicate_arrays = jt.map(
            lambda leaf: None if leaf is None else leaf[replicate_index],
            model_arrays,
            is_leaf=lambda leaf: leaf is None,
        )
        model_replicate = eqx.combine(replicate_arrays, model_other)
        model_replicate = _with_single_replicate_state_initializers(
            model_replicate,
            n_replicates=n_replicates,
            replicate_index=replicate_index,
        )
        objective, diagnostics = policy_adversary_objective(
            policy,
            task,
            model_replicate,
            trial_specs,
            task.loss_func,
            keys_model,
            hps.policy_adversary_training,
        )
        objectives.append(objective)
        diagnostics_by_replicate.append(diagnostics)
    objective = jnp.mean(jnp.stack(objectives))
    diagnostics = jt.map(lambda *values: jnp.mean(jnp.stack(values)), *diagnostics_by_replicate)
    return objective, diagnostics


def _policy_adversary_diagnostics_arrays(
    diagnostics: dict[str, np.ndarray],
    *,
    batch_index: int,
    chunk_batches: int,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "policy_adversary_diagnostic_sampled": np.zeros(chunk_batches, dtype=bool),
        "policy_adversary_diagnostic_global_batch": np.full(
            chunk_batches,
            np.nan,
            dtype=np.float32,
        ),
    }
    arrays["policy_adversary_diagnostic_sampled"][-1] = True
    arrays["policy_adversary_diagnostic_global_batch"][-1] = float(batch_index)
    for name, value in diagnostics.items():
        sampled = np.asarray(value)
        if sampled.ndim == 0:
            sampled = sampled.reshape((1,))
        chunk = np.full((chunk_batches, *sampled.shape), np.nan, dtype=sampled.dtype)
        chunk[-1] = sampled
        arrays[f"policy_adversary_{name}"] = chunk
    return arrays


def _adaptive_epsilon_zero_guard_from_state(
    adaptive_state: AdaptiveEpsilonState | None,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if adaptive_state is not None and isinstance(adaptive_state.zero_adversary_guard, dict):
        return _normalize_adaptive_epsilon_zero_guard(
            adaptive_state.zero_adversary_guard,
            enabled=enabled,
        )
    return _initial_adaptive_epsilon_zero_guard(enabled=enabled)


def _update_adaptive_epsilon_zero_guard(
    guard: dict[str, Any],
    adaptive_epsilon_diagnostics: dict[str, np.ndarray],
) -> dict[str, Any]:
    next_guard = dict(guard)
    next_guard["enabled"] = True
    next_guard["checkpoints_seen"] = int(next_guard.get("checkpoints_seen", 0)) + 1
    evidence = _adaptive_epsilon_zero_checkpoint_evidence(adaptive_epsilon_diagnostics)
    next_guard["last_checkpoint"] = evidence
    if evidence["active"] and evidence["zero_adversary"]:
        consecutive = int(next_guard.get("consecutive_active_zero_adversary_checkpoints", 0)) + 1
    else:
        consecutive = 0
    next_guard["consecutive_active_zero_adversary_checkpoints"] = consecutive
    next_guard["should_stop"] = consecutive >= 2
    return next_guard


def _adaptive_epsilon_zero_checkpoint_evidence(
    adaptive_epsilon_diagnostics: dict[str, np.ndarray],
) -> dict[str, Any]:
    gain = None
    gain_source = None
    gain_keys = (
        "adaptive_epsilon_adaptive_update_inner_selected_objective_gain_over_zero",
        "adaptive_epsilon_inner_selected_objective_gain_over_zero",
    )
    for key in gain_keys:
        gain = _latest_scalar(adaptive_epsilon_diagnostics.get(key))
        if gain is not None:
            gain_source = key
            break

    target_damage = _latest_scalar(
        adaptive_epsilon_diagnostics.get("adaptive_epsilon_target_damage")
    )
    outer_weight = _latest_scalar(adaptive_epsilon_diagnostics.get("adaptive_epsilon_outer_weight"))
    active = (
        target_damage is not None
        and target_damage > 0.0
        and outer_weight is not None
        and outer_weight > 0.0
    )
    zero_adversary = (
        active and gain is not None and gain <= ADAPTIVE_EPSILON_ZERO_ADVERSARY_GAIN_TOLERANCE
    )
    return {
        "active": bool(active),
        "zero_adversary": bool(zero_adversary),
        "selected_objective_gain_over_zero": gain,
        "gain_source": gain_source,
        "target_damage": target_damage,
        "outer_weight": outer_weight,
        "gain_tolerance": ADAPTIVE_EPSILON_ZERO_ADVERSARY_GAIN_TOLERANCE,
    }


def _broad_epsilon_pgd_diagnostics_arrays(
    task: Any,
    model: Any,
    hps: TreeNamespace,
    *,
    key: Any,
    batch_index: int,
    chunk_batches: int,
) -> dict[str, np.ndarray]:
    if not _broad_epsilon_pgd_training_enabled(hps):
        return {}

    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
    batch_size = int(hps.batch_size)
    batch_info = BatchInfo(
        size=batch_size,
        start=0,
        current=int(batch_index),
        total=int(hps.n_batches_condition),
    )

    def diagnostic_for_replicate(model_replicate: Any, key_replicate: Any):
        key_trials, _, key_model = jr.split(key_replicate, 3)
        keys_trials = jr.split(key_trials, batch_size)
        keys_model = jr.split(key_model, batch_size)
        trial_specs = eqx.filter_vmap(
            partial(
                task.get_train_trial_with_intervenor_params,
                batch_info=batch_info,
            )
        )(keys_trials)
        _, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
            task,
            model_replicate,
            trial_specs,
            task.loss_func,
            keys_model,
            hps.broad_epsilon_pgd_training,
            return_diagnostics=True,
        )
        return diagnostics

    keys = jr.split(key, n_replicates)
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_axis_array(leaf, n_replicates),
    )
    per_replicate_diagnostics = []
    for replicate_index, key_replicate in enumerate(keys):
        replicate_arrays = jt.map(
            lambda leaf: None if leaf is None else leaf[replicate_index],
            model_arrays,
            is_leaf=lambda leaf: leaf is None,
        )
        model_replicate = eqx.combine(replicate_arrays, model_other)
        model_replicate = _with_single_replicate_state_initializers(
            model_replicate,
            n_replicates=n_replicates,
            replicate_index=replicate_index,
        )
        per_replicate_diagnostics.append(diagnostic_for_replicate(model_replicate, key_replicate))
    diagnostics = jt.map(lambda *values: jnp.stack(values), *per_replicate_diagnostics)
    arrays: dict[str, np.ndarray] = {
        "pgd_broad_epsilon_diagnostic_sampled": np.zeros(chunk_batches, dtype=bool),
        "pgd_broad_epsilon_diagnostic_global_batch": np.full(
            chunk_batches,
            np.nan,
            dtype=np.float32,
        ),
    }
    arrays["pgd_broad_epsilon_diagnostic_sampled"][-1] = True
    arrays["pgd_broad_epsilon_diagnostic_global_batch"][-1] = float(batch_index)
    for name, value in diagnostics.items():
        sampled = np.asarray(jax.device_get(value))
        if sampled.ndim == 0:
            sampled = sampled.reshape((1,))
        chunk = np.full(
            (chunk_batches, *sampled.shape),
            np.nan,
            dtype=sampled.dtype,
        )
        chunk[-1] = sampled
        arrays[f"pgd_broad_epsilon_{name}"] = chunk
    return arrays


def _with_single_replicate_state_initializers(
    model: Any,
    *,
    n_replicates: int,
    replicate_index: int,
) -> Any:
    nodes = getattr(model, "nodes", {})
    for node_name, node in nodes.items():
        state_index = getattr(node, "state_index", None)
        if not isinstance(state_index, eqx.nn.StateIndex):
            continue
        changed = False

        def unbatch_init_leaf(leaf: Any) -> Any:
            nonlocal changed
            if _is_replicate_axis_array(leaf, n_replicates):
                changed = True
                return leaf[replicate_index]
            return leaf

        init = jt.map(unbatch_init_leaf, state_index.init)
        if changed:
            model = eqx.tree_at(
                lambda graph, name=node_name: graph.nodes[name].state_index.init,
                model,
                init,
            )
    return model


if __name__ == "__main__":
    raise SystemExit(main())
