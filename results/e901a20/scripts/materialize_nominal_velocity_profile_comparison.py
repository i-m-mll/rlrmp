"""Materialize nominal velocity profiles for the e901a20 policy-adversary run."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import feedbax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.models.networks import MaskedLinear
from jax_cookbook import load_with_hyperparameters

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    ReplicateCheckpointSelection,
    select_validation_checkpoints_for_run,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.io import update_marked_section
from rlrmp.paths import (
    REPO_ROOT,
    figure_artifact_dir,
    figure_spec_dir,
    mkdir_p,
    resolve_run_artifact_path,
    run_spec_path,
)
from rlrmp.train.cs_perturbation_training import (
    TargetRelativeMultiTargetTrainingConfig,
    apply_validation_bin,
    apply_validation_target_distribution,
)
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.viz import profile_comparison_grid

jax.config.update("jax_enable_x64", True)


EXPERIMENT = "e901a20"
TOPIC = "nominal_velocity_profile_comparison"
NOMINAL_MARKER = "nominal_velocity_profile_comparison"
NO_PGD_SPLIT_TOPIC = "no_pgd_heldout_split"
NO_PGD_SPLIT_MARKER = "no_pgd_heldout_split"
NO_PGD_CROSSED_TOPIC = "no_pgd_crossed_target_grid"
NO_PGD_CROSSED_MARKER = "no_pgd_crossed_target_grid"
OLD_COMPAT_FIRST_TARGET_TOPIC = "old_compatible_first_target_velocity"
OLD_COMPAT_FIRST_TARGET_MARKER = "old_compatible_first_target_velocity"
OLD_COMPAT_ALL_TARGET_TOPIC = "old_compatible_all_target_aligned_velocity"
OLD_COMPAT_ALL_TARGET_MARKER = "old_compatible_all_target_aligned_velocity"
OLD_COMPAT_N_ROLLOUT_REPEATS = 64
NUMERIC_SCALAR_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)


@dataclass(frozen=True)
class RunRef:
    """One run to include in the nominal velocity comparison."""

    experiment: str
    run_id: str
    label: str
    color: str


@dataclass(frozen=True)
class VelocityProfile:
    """Nominal target-radial velocity profile for one trained run."""

    run: RunRef
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_replicates: int
    n_trials: int
    target_distance_min_m: float
    target_distance_max_m: float
    peak_mean_length_normalized_forward_velocity_1_s: float
    time_of_peak_mean_forward_velocity_s: float


@dataclass(frozen=True)
class NominalEvaluation:
    """Per-replicate nominal target-radial velocity profiles."""

    run_spec: dict[str, Any]
    time_s: np.ndarray
    normalized_values: np.ndarray
    target_distance: np.ndarray
    targets_m: np.ndarray


@dataclass(frozen=True)
class CrossedPanel:
    """One crossed target-grid diagnostic panel."""

    name: str
    title: str
    target_config: TargetRelativeMultiTargetTrainingConfig
    primary_label: str
    secondary_label: str
    primary_color: str
    secondary_color: str
    primary_targets_m: np.ndarray
    secondary_targets_m: np.ndarray
    evaluation: NominalEvaluation
    primary_profile: VelocityProfile
    secondary_profile: VelocityProfile


@dataclass(frozen=True)
class CompanionProfile:
    """One validation-selected stochastic companion velocity profile."""

    run: RunRef
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_replicates: int
    n_target_conditions: int
    n_rollout_repeats: int
    target_distance_min_m: float
    target_distance_max_m: float
    peak_mean_velocity: float
    time_of_peak_mean_velocity_s: float
    selected_checkpoints: tuple[ReplicateCheckpointSelection, ...]


NO_PGD_REF = RunRef(
    "020a65b",
    "target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64",
    "020a65b no-PGD H0",
    "#64748b",
)


RUNS: tuple[RunRef, ...] = (
    NO_PGD_REF,
    RunRef(
        "020a65b",
        "target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64",
        "020a65b PGD H0",
        "#dc2626",
    ),
    RunRef(
        "e901a20",
        "h0_policy_adversary__plain",
        "Policy adversary plain",
        "#2563eb",
    ),
    RunRef(
        "e901a20",
        "h0_policy_adversary__energy",
        "Policy adversary energy",
        "#059669",
    ),
)


def repo_relative(path: Path) -> str:
    """Return a repo-relative path string."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.absolute().relative_to(REPO_ROOT.absolute()))


def load_run_spec(ref: RunRef) -> dict[str, Any]:
    """Load one tracked run spec."""

    path = run_spec_path(ref.experiment, ref.run_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing run spec for {ref.run_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_dir(ref: RunRef) -> Path:
    """Return one run artifact directory."""

    path = REPO_ROOT / "_artifacts" / ref.experiment / "runs" / ref.run_id
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact dir for {ref.run_id}: {path}")
    return path


def nominalize_trial_specs(trial_specs: Any) -> Any:
    """Return validation specs with explicit perturbation inputs zeroed."""

    specs = trial_specs
    if PLANT_INTERVENOR_LABEL in specs.intervene:
        scale = specs.intervene[PLANT_INTERVENOR_LABEL].scale
        specs = eqx.tree_at(
            lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
            specs,
            jnp.zeros_like(scale),
        )
    zero_input_keys = tuple(
        key
        for key, value in specs.inputs.items()
        if eqx.is_array(value)
        and (
            key == "epsilon"
            or key.startswith("perturbation_training.")
            or key.endswith("_perturbation")
        )
    )
    if zero_input_keys:
        inputs = dict(specs.inputs)
        for key in zero_input_keys:
            inputs[key] = jnp.zeros_like(inputs[key])
        specs = eqx.tree_at(lambda t: t.inputs, specs, inputs)
    return specs


def initial_effector_field(trial_specs: Any, field: str) -> jnp.ndarray:
    """Return an initial effector field from trial initial conditions."""

    for init_state in trial_specs.inits.values():
        if eqx.is_array(init_state):
            if field == "pos":
                return init_state[..., :2]
            if field == "vel":
                return init_state[..., 2:4]
        value = getattr(init_state, field, None)
        if value is not None:
            return value
        vector = getattr(init_state, "vector", None)
        if vector is not None:
            if field == "pos":
                return vector[..., :2]
            if field == "vel":
                return vector[..., 2:4]
    raise ValueError(f"Could not find initial effector {field!r} in trial specs")


def final_goal_position(trial_specs: Any) -> jnp.ndarray:
    """Return final target position for each nominal trial."""

    if not trial_specs.targets:
        raise ValueError("Trial specs do not declare targets")
    target = next(iter(trial_specs.targets.values())).value
    return target[:, -1, :]


def force_legacy_masked_readout(model: Any) -> Any:
    """Return a model template with the legacy dynamic readout-mask leaf present."""

    net = model.nodes["net"].net
    if getattr(net, "dtype", None) is not jnp.float64:
        # This static dtype is not a PyTree leaf, so ``eqx.tree_at`` cannot target it.
        # The model is a fresh deserialization template, so the local mutation is contained.
        object.__setattr__(net, "dtype", jnp.float64)
        net = model.nodes["net"].net
    if not isinstance(net.readout, eqx.nn.Linear):
        return model
    mask = jnp.ones_like(net.readout.weight, dtype=bool)
    masked_readout = MaskedLinear(
        net.readout.weight.shape[-1],
        net.readout.weight.shape[-2],
        mask,
        use_bias=net.readout.bias is not None,
        dtype=net.readout.weight.dtype,
        key=jr.PRNGKey(0),
    )
    masked_readout = eqx.tree_at(
        lambda layer: (layer.linear.weight, layer.linear.bias),
        masked_readout,
        (net.readout.weight, net.readout.bias),
    )
    return eqx.tree_at(lambda m: m.nodes["net"].net.readout, model, masked_readout)


def legacy_static_numeric_template(tree: Any, n_replicates: int) -> Any:
    """Return a template that consumes legacy serialized numeric scalar leaves."""

    def replace_numeric_scalar(leaf: Any) -> Any:
        if isinstance(leaf, NUMERIC_SCALAR_TYPES):
            return jnp.full((n_replicates,), leaf)
        if eqx.is_array(leaf) and np.issubdtype(leaf.dtype, np.floating):
            return leaf.astype(jnp.float64)
        return leaf

    return jt.map(replace_numeric_scalar, tree)


def legacy_static_numeric_filter(file_obj: Any, leaf: Any) -> Any:
    """Deserialize legacy numeric scalar placeholders while preserving template shape."""

    if eqx.is_array(leaf):
        out = jnp.load(file_obj)
        if leaf.ndim == 1 and out.shape == ():
            out = jnp.full(leaf.shape, out, dtype=leaf.dtype)
        if out.dtype != leaf.dtype:
            out = out.astype(leaf.dtype)
        return out
    return eqx.default_deserialise_filter_spec(file_obj, leaf)


def scalar_from_legacy_array(value: Any, scalar_type: type) -> Any:
    """Return the first scalar from a legacy replicated scalar array."""

    raw = np.asarray(value).reshape(-1)[0]
    if scalar_type is bool:
        return bool(raw)
    return scalar_type(raw)


def restore_legacy_static_numeric_metadata(model: Any) -> Any:
    """Restore known legacy scalar metadata fields after compatibility loading."""

    return eqx.tree_at(
        lambda m: (
            m.nodes["efferent"].delay,
            m.nodes["efferent"].noise_func.terms[0].noise_func.std,
            m.nodes["efferent"].noise_func.terms[0].noise_func.mean,
            m.nodes["efferent"].noise_func.terms[1].std,
            m.nodes["efferent"].noise_func.terms[1].mean,
            m.nodes["efferent"].add_noise,
            m.nodes["efferent"].init_value,
            m.nodes["mechanics"].dt,
            m.nodes["net"].net.input_size,
            m.nodes["net"].net.hidden_size,
            m.nodes["net"].net.out_size,
            m.nodes["net"].net.population_structure.n_input_only,
            m.nodes["net"].net.population_structure.n_readout_only,
            m.nodes["net"].net.population_structure.n_recurrent_only,
            m.nodes["net"].net.population_structure.n_input_readout,
            m.nodes["sensory"].delay,
            m.nodes["sensory"].noise_func.std,
            m.nodes["sensory"].noise_func.mean,
            m.nodes["sensory"].add_noise,
            m.nodes["sensory"].init_value,
        ),
        model,
        (
            scalar_from_legacy_array(model.nodes["efferent"].delay, int),
            scalar_from_legacy_array(
                model.nodes["efferent"].noise_func.terms[0].noise_func.std,
                float,
            ),
            scalar_from_legacy_array(
                model.nodes["efferent"].noise_func.terms[0].noise_func.mean,
                float,
            ),
            scalar_from_legacy_array(model.nodes["efferent"].noise_func.terms[1].std, float),
            scalar_from_legacy_array(model.nodes["efferent"].noise_func.terms[1].mean, float),
            scalar_from_legacy_array(model.nodes["efferent"].add_noise, bool),
            scalar_from_legacy_array(model.nodes["efferent"].init_value, float),
            scalar_from_legacy_array(model.nodes["mechanics"].dt, float),
            scalar_from_legacy_array(model.nodes["net"].net.input_size, int),
            scalar_from_legacy_array(model.nodes["net"].net.hidden_size, int),
            scalar_from_legacy_array(model.nodes["net"].net.out_size, int),
            scalar_from_legacy_array(
                model.nodes["net"].net.population_structure.n_input_only,
                int,
            ),
            scalar_from_legacy_array(
                model.nodes["net"].net.population_structure.n_readout_only,
                int,
            ),
            scalar_from_legacy_array(
                model.nodes["net"].net.population_structure.n_recurrent_only,
                int,
            ),
            scalar_from_legacy_array(
                model.nodes["net"].net.population_structure.n_input_readout,
                int,
            ),
            scalar_from_legacy_array(model.nodes["sensory"].delay, int),
            scalar_from_legacy_array(model.nodes["sensory"].noise_func.std, float),
            scalar_from_legacy_array(model.nodes["sensory"].noise_func.mean, float),
            scalar_from_legacy_array(model.nodes["sensory"].add_noise, bool),
            scalar_from_legacy_array(model.nodes["sensory"].init_value, float),
        ),
    )


def load_trained_model(ref: RunRef, hps: TreeNamespace, seed: int) -> Any:
    """Load a trained model with the matching template."""

    path = resolve_run_artifact_path(artifact_dir(ref), "trained_model.eqx")
    if not path.exists():
        raise FileNotFoundError(f"Missing trained model for {ref.run_id}: {path}")

    def base_template(key: Any) -> Any:
        return setup_task_model_pair(hps, key=key).model

    try:
        model, _hyperparameters = load_with_hyperparameters(
            path,
            setup_func=lambda key, **_kwargs: base_template(key),
        )
        return model
    except Exception:
        n_replicates = int(hps.model.n_replicates)

        def legacy_template(key: Any) -> Any:
            return legacy_static_numeric_template(
                force_legacy_masked_readout(base_template(key)),
                n_replicates,
            )

        try:
            model, _hyperparameters = load_with_hyperparameters(
                path,
                setup_func=lambda key, **_kwargs: legacy_template(key),
                filter_spec=legacy_static_numeric_filter,
            )
        except Exception as legacy_error:
            raise RuntimeError(
                f"Could not load trained model for {ref.run_id!r} with either normal "
                "or legacy static-numeric compatibility templates."
            ) from legacy_error
        try:
            return restore_legacy_static_numeric_metadata(model)
        except Exception as restore_error:
            raise RuntimeError(
                f"Loaded legacy model for {ref.run_id!r} but could not restore scalar metadata."
            ) from restore_error


def checkpoint_model_template(hps: TreeNamespace, seed: int) -> Any:
    """Return a checkpoint model template for normal or legacy checkpoint leaves."""

    return setup_task_model_pair(hps, key=jr.PRNGKey(seed)).model


def load_checkpoint_model_compatible(path: Path, hps: TreeNamespace, seed: int) -> Any:
    """Load one checkpoint model with the same legacy compatibility as final models."""

    template = checkpoint_model_template(hps, seed)
    try:
        return eqx.tree_deserialise_leaves(path, template)
    except Exception:
        n_replicates = int(hps.model.n_replicates)
        legacy_template = legacy_static_numeric_template(
            force_legacy_masked_readout(template),
            n_replicates,
        )
        model = eqx.tree_deserialise_leaves(
            path,
            legacy_template,
            filter_spec=legacy_static_numeric_filter,
        )
        return restore_legacy_static_numeric_metadata(model)


def load_validation_selected_model(ref: RunRef, run_spec: dict[str, Any], hps: TreeNamespace) -> tuple[
    tuple[Any, ...],
    tuple[ReplicateCheckpointSelection, ...],
]:
    """Load validation-selected checkpoint models for per-replicate evaluation."""

    n_replicates = int(hps.model.n_replicates)
    seed = int(run_spec.get("seed", 42))
    selections = tuple(
        select_validation_checkpoints_for_run(
            experiment=ref.experiment,
            run_id=ref.run_id,
            repo_root=REPO_ROOT,
        )
    )
    if len(selections) != n_replicates:
        raise ValueError(
            f"Selection count {len(selections)} does not match n_replicates={n_replicates}"
        )
    models = tuple(
        load_checkpoint_model_compatible(selection.checkpoint_path / "model.eqx", hps, seed)
        for selection in selections
    )

    return models, selections


def target_adapter_from_pair(pair: Any) -> Any:
    """Return the target-relative task adapter from the task stack."""

    task = pair.task
    for _depth in range(8):
        if task.__class__.__name__ == "TargetRelativeMultiTargetTrainingTaskAdapter":
            return task
        task = getattr(task, "task", None)
        if task is None:
            break
    raise ValueError("Could not find TargetRelativeMultiTargetTrainingTaskAdapter")


def nominal_trial_specs_for_target_config(pair: Any, target_config: Any) -> Any:
    """Build nominal validation trial specs for a custom target grid."""

    target_adapter = target_adapter_from_pair(pair)
    trial_specs = apply_validation_target_distribution(
        target_adapter.task.validation_trials,
        target_config,
    )
    top_task = pair.task
    if top_task.__class__.__name__ == "FixedTargetPerturbationTrainingTaskAdapter":
        trial_specs = apply_validation_bin(trial_specs, top_task.config, "nominal")
    return nominalize_trial_specs(trial_specs)


def target_array(targets: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Return target tuples as a stable float array."""

    return np.asarray(targets, dtype=np.float64)


def target_membership_mask(targets_m: np.ndarray, reference_targets_m: np.ndarray) -> np.ndarray:
    """Return a boolean mask selecting targets present in ``reference_targets_m``."""

    mask = np.zeros(targets_m.shape[0], dtype=np.bool_)
    for idx, target in enumerate(targets_m):
        mask[idx] = bool(
            np.any(np.all(np.isclose(reference_targets_m, target, atol=1e-6), axis=1))
        )
    if not np.any(mask):
        raise ValueError("Target membership mask selects no trials")
    return mask


def target_angle_deg(targets_m: np.ndarray) -> np.ndarray:
    """Return target angles in degrees on [0, 360)."""

    return np.mod(np.degrees(np.arctan2(targets_m[:, 1], targets_m[:, 0])), 360.0)


def trial_count_from_specs(trial_specs: Any) -> int:
    """Return the leading trial-axis length from a trial-spec tree."""

    for target in getattr(trial_specs, "targets", {}).values():
        value = getattr(target, "value", None)
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    for init in getattr(trial_specs, "inits", {}).values():
        shape = getattr(init, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    for value in getattr(trial_specs, "inputs", {}).values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    raise ValueError("Could not infer trial count from trial specs")


def repeat_trial_axis(trial_specs: Any, repeats: int, *, first_only: bool) -> Any:
    """Repeat trial-axis leaves for stochastic old-compatible evaluation."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    source_trials = trial_count_from_specs(trial_specs)

    def repeat_leaf(leaf: Any) -> Any:
        shape = getattr(leaf, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] == source_trials:
            source = leaf[:1] if first_only else leaf
            return jnp.repeat(source, repeats, axis=0)
        return leaf

    return jt.map(repeat_leaf, trial_specs)


def profile_from_values(
    ref: RunRef,
    time_s: np.ndarray,
    normalized_values: np.ndarray,
    target_distance: np.ndarray,
    target_mask: np.ndarray | None = None,
) -> VelocityProfile:
    """Summarize length-normalized forward velocities for a target subset."""

    if target_mask is None:
        selected = normalized_values
        selected_distance = target_distance
    else:
        if target_mask.dtype != np.bool_:
            raise TypeError("target_mask must be a boolean array")
        if target_mask.shape != target_distance.shape:
            raise ValueError(
                f"target_mask shape {target_mask.shape} does not match target distances "
                f"{target_distance.shape}"
            )
        if not np.any(target_mask):
            raise ValueError(f"Target mask for {ref.label!r} selects no trials")
        selected = normalized_values[:, target_mask, :]
        selected_distance = target_distance[target_mask]

    n_replicates = int(selected.shape[0])
    n_trials = int(selected.shape[1])
    pooled = selected.reshape(n_replicates * n_trials, selected.shape[-1])
    mean = np.mean(pooled, axis=0)
    std = np.std(pooled, axis=0, ddof=1)
    peak_idx = int(np.nanargmax(mean))
    return VelocityProfile(
        run=ref,
        time_s=time_s,
        mean=mean,
        std=std,
        n_replicates=n_replicates,
        n_trials=n_trials,
        target_distance_min_m=float(np.min(selected_distance)),
        target_distance_max_m=float(np.max(selected_distance)),
        peak_mean_length_normalized_forward_velocity_1_s=float(mean[peak_idx]),
        time_of_peak_mean_forward_velocity_s=float(time_s[peak_idx]),
    )


def evaluate_nominal_values_for_trial_specs(
    ref: RunRef,
    run_spec: dict[str, Any],
    hps: TreeNamespace,
    trial_specs: Any,
) -> NominalEvaluation:
    """Evaluate a trained run on explicit nominal trial specs."""

    n_replicates = int(hps.model.n_replicates)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model = load_trained_model(ref, hps, seed)
    n_trials = int(next(iter(trial_specs.targets.values())).value.shape[0])
    init_pos = initial_effector_field(trial_specs, "pos")
    init_vel = initial_effector_field(trial_specs, "vel")
    goal = final_goal_position(trial_specs)
    direction = goal - init_pos
    direction_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    direction_unit = direction / jnp.maximum(direction_norm, 1e-12)

    def is_replicate_array(leaf: Any) -> bool:
        return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates

    model_arrays, model_other = eqx.partition(model, is_replicate_array)

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> jnp.ndarray:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        states = pair.task.eval_trials(replicate_model, trial_specs, jr.split(key, n_trials))
        velocity = jnp.concatenate(
            [init_vel[:, None, :], states.mechanics.effector.vel],
            axis=1,
        )
        return jnp.sum(velocity * direction_unit[:, None, :], axis=-1)

    forward_velocity = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(0), n_replicates),
    )
    values = np.asarray(forward_velocity, dtype=np.float64)
    target_distance = np.asarray(direction_norm[..., 0], dtype=np.float64)
    normalized_values = values / np.maximum(target_distance[None, :, None], 1e-12)
    dt = float(run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
    time_s = np.arange(normalized_values.shape[-1], dtype=np.float64) * dt
    goal_np = np.asarray(goal, dtype=np.float64)
    if goal_np.shape[0] != n_trials:
        raise ValueError(
            f"Goal count {goal_np.shape[0]} does not match validation trial count {n_trials}"
        )
    return NominalEvaluation(
        run_spec=run_spec,
        time_s=time_s,
        normalized_values=normalized_values,
        target_distance=target_distance,
        targets_m=goal_np,
    )


def evaluate_nominal_values(ref: RunRef) -> NominalEvaluation:
    """Evaluate a trained run on nominal validation trials and return per-trial profiles."""

    run_spec = load_run_spec(ref)
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    trial_specs = nominalize_trial_specs(pair.task.validation_trials)
    return evaluate_nominal_values_for_trial_specs(ref, run_spec, hps, trial_specs)


def evaluate_profile(ref: RunRef) -> VelocityProfile:
    """Evaluate a trained run on nominal validation trials."""

    evaluation = evaluate_nominal_values(ref)
    return profile_from_values(
        ref,
        evaluation.time_s,
        evaluation.normalized_values,
        evaluation.target_distance,
    )


def companion_profile_from_values(
    ref: RunRef,
    *,
    time_s: np.ndarray,
    values: np.ndarray,
    target_distance: np.ndarray,
    n_target_conditions: int,
    n_rollout_repeats: int,
    selected_checkpoints: tuple[ReplicateCheckpointSelection, ...],
) -> CompanionProfile:
    """Summarize stochastic companion profiles over replicates and trials."""

    n_replicates = int(values.shape[0])
    pooled = values.reshape(n_replicates * values.shape[1], values.shape[-1])
    mean = np.mean(pooled, axis=0)
    std = np.std(pooled, axis=0, ddof=1)
    peak_idx = int(np.nanargmax(mean))
    return CompanionProfile(
        run=ref,
        time_s=time_s,
        mean=mean,
        std=std,
        n_replicates=n_replicates,
        n_target_conditions=n_target_conditions,
        n_rollout_repeats=n_rollout_repeats,
        target_distance_min_m=float(np.min(target_distance)),
        target_distance_max_m=float(np.max(target_distance)),
        peak_mean_velocity=float(mean[peak_idx]),
        time_of_peak_mean_velocity_s=float(time_s[peak_idx]),
        selected_checkpoints=selected_checkpoints,
    )


def evaluate_validation_selected_stochastic_values(
    ref: RunRef,
    *,
    first_target_only: bool,
    raw_x_velocity: bool,
    length_normalize: bool,
    n_rollout_repeats: int = OLD_COMPAT_N_ROLLOUT_REPEATS,
) -> CompanionProfile:
    """Evaluate one row with validation-selected checkpoints and repeated rollouts."""

    run_spec = load_run_spec(ref)
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    seed = int(run_spec.get("seed", 42))
    n_replicates = int(hps.model.n_replicates)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    base_trials = nominalize_trial_specs(pair.task.validation_trials)
    base_trial_count = trial_count_from_specs(base_trials)
    trial_specs = repeat_trial_axis(
        base_trials,
        n_rollout_repeats,
        first_only=first_target_only,
    )
    selected_models, selected_checkpoints = load_validation_selected_model(ref, run_spec, hps)
    n_trials = int(next(iter(trial_specs.targets.values())).value.shape[0])
    init_pos = initial_effector_field(trial_specs, "pos")
    init_vel = initial_effector_field(trial_specs, "vel")
    goal = final_goal_position(trial_specs)
    direction = goal - init_pos
    direction_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    direction_unit = direction / jnp.maximum(direction_norm, 1e-12)

    def is_replicate_array(leaf: Any) -> bool:
        return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates

    def replicate_model_from_checkpoint(model: Any, replicate: int) -> Any:
        model_arrays, model_other = eqx.partition(model, is_replicate_array)

        def select_replicate(leaf: Any) -> Any:
            if eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates:
                return leaf[replicate]
            return leaf

        return eqx.combine(jt.map(select_replicate, model_arrays), model_other)

    def eval_one_replicate(replicate_model: Any, key: Any) -> jnp.ndarray:
        states = pair.task.eval_trials(replicate_model, trial_specs, jr.split(key, n_trials))
        velocity = jnp.concatenate(
            [init_vel[:, None, :], states.mechanics.effector.vel],
            axis=1,
        )
        if raw_x_velocity:
            return velocity[..., 0]
        projected = jnp.sum(velocity * direction_unit[:, None, :], axis=-1)
        if length_normalize:
            projected = projected / jnp.maximum(direction_norm[:, 0, None], 1e-12)
        return projected

    values = jnp.stack(
        [
            eval_one_replicate(
                replicate_model_from_checkpoint(model, selection.replicate),
                key,
            )
            for model, selection, key in zip(
                selected_models,
                selected_checkpoints,
                jr.split(jr.PRNGKey(0), n_replicates),
                strict=True,
            )
        ],
        axis=0,
    )
    values_np = np.asarray(values, dtype=np.float64)
    target_distance = np.asarray(direction_norm[..., 0], dtype=np.float64)
    dt = float(run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
    time_s = np.arange(values_np.shape[-1], dtype=np.float64) * dt
    n_target_conditions = 1 if first_target_only else base_trial_count
    return companion_profile_from_values(
        ref,
        time_s=time_s,
        values=values_np,
        target_distance=target_distance,
        n_target_conditions=n_target_conditions,
        n_rollout_repeats=n_rollout_repeats,
        selected_checkpoints=selected_checkpoints,
    )


def evaluate_old_compatible_first_target_profiles() -> list[CompanionProfile]:
    """Evaluate first-target old-compatible profiles for all comparison rows."""

    return [
        evaluate_validation_selected_stochastic_values(
            ref,
            first_target_only=True,
            raw_x_velocity=True,
            length_normalize=False,
        )
        for ref in RUNS
    ]


def evaluate_old_compatible_all_target_profiles() -> list[CompanionProfile]:
    """Evaluate all-target aligned/scaled old-compatible profiles."""

    return [
        evaluate_validation_selected_stochastic_values(
            ref,
            first_target_only=False,
            raw_x_velocity=False,
            length_normalize=True,
        )
        for ref in RUNS
    ]


def held_out_target_mask(run_spec: dict[str, Any], trial_specs: Any) -> np.ndarray:
    """Return a boolean mask for held-out validation targets."""

    distribution = (
        run_spec.get("hps", {})
        .get("target_relative_multitarget", {})
        .get("target_distribution", {})
    )
    held_out_targets = np.asarray(distribution.get("held_out_targets_m"), dtype=np.float64)
    if held_out_targets.ndim != 2 or held_out_targets.shape[1] != 2:
        raise ValueError("Run spec does not contain a valid held_out_targets_m list")

    goals = np.asarray(final_goal_position(trial_specs), dtype=np.float64)
    target_mask = np.zeros(goals.shape[0], dtype=np.bool_)
    for idx, goal in enumerate(goals):
        target_mask[idx] = bool(np.any(np.all(np.isclose(held_out_targets, goal, atol=1e-6), axis=1)))
    if not np.any(target_mask):
        raise ValueError("No validation trials matched held_out_targets_m")
    if np.all(target_mask):
        raise ValueError("All validation trials matched held_out_targets_m")
    return target_mask


def evaluate_no_pgd_heldout_split() -> list[VelocityProfile]:
    """Evaluate no-PGD nominal profiles split by held-out target membership."""

    ref = NO_PGD_REF
    run_spec = load_run_spec(ref)
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run_spec.get("seed", 42))))
    trial_specs = nominalize_trial_specs(pair.task.validation_trials)
    held_out_mask = held_out_target_mask(run_spec, trial_specs)
    evaluation = evaluate_nominal_values(ref)
    return [
        profile_from_values(
            RunRef(
                ref.experiment,
                f"{ref.run_id}::non_held_out",
                "Non-held-out validation targets",
                "#64748b",
            ),
            evaluation.time_s,
            evaluation.normalized_values,
            evaluation.target_distance,
            ~held_out_mask,
        ),
        profile_from_values(
            RunRef(
                ref.experiment,
                f"{ref.run_id}::held_out",
                "Held-out diagonal targets",
                "#f97316",
            ),
            evaluation.time_s,
            evaluation.normalized_values,
            evaluation.target_distance,
            held_out_mask,
        ),
    ]


def evaluate_no_pgd_crossed_panels() -> list[CrossedPanel]:
    """Evaluate crossed direction/length grids for the no-PGD row."""

    ref = NO_PGD_REF
    run_spec = load_run_spec(ref)
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    base_cfg = target_adapter_from_pair(pair).config

    direction_cfg = TargetRelativeMultiTargetTrainingConfig(
        enabled=True,
        force_filter_feedback=bool(base_cfg.force_filter_feedback),
        seen_directions_deg=tuple(float(x) for x in base_cfg.seen_directions_deg),
        held_out_directions_deg=tuple(float(x) for x in base_cfg.held_out_directions_deg),
        seen_amplitudes_m=tuple(float(x) for x in base_cfg.seen_amplitudes_m),
        held_out_amplitudes_m=tuple(float(x) for x in base_cfg.seen_amplitudes_m),
        original_target_anchor_m=tuple(float(x) for x in base_cfg.original_target_anchor_m),
    )
    direction_trials = nominal_trial_specs_for_target_config(pair, direction_cfg)
    direction_eval = evaluate_nominal_values_for_trial_specs(
        ref,
        run_spec,
        hps,
        direction_trials,
    )
    direction_seen_targets = target_array(direction_cfg.seen_targets_m)
    direction_held_targets = target_array(direction_cfg.held_out_targets_m)
    direction_seen_mask = target_membership_mask(direction_eval.targets_m, direction_seen_targets)
    direction_held_mask = target_membership_mask(direction_eval.targets_m, direction_held_targets)

    length_cfg = TargetRelativeMultiTargetTrainingConfig(
        enabled=True,
        force_filter_feedback=bool(base_cfg.force_filter_feedback),
        seen_directions_deg=tuple(float(x) for x in base_cfg.seen_directions_deg),
        held_out_directions_deg=tuple(float(x) for x in base_cfg.seen_directions_deg),
        seen_amplitudes_m=tuple(float(x) for x in base_cfg.seen_amplitudes_m),
        held_out_amplitudes_m=tuple(float(x) for x in base_cfg.held_out_amplitudes_m),
        original_target_anchor_m=tuple(float(x) for x in base_cfg.original_target_anchor_m),
    )
    length_trials = nominal_trial_specs_for_target_config(pair, length_cfg)
    length_eval = evaluate_nominal_values_for_trial_specs(ref, run_spec, hps, length_trials)
    length_seen_targets = target_array(length_cfg.seen_targets_m)
    length_held_targets = target_array(length_cfg.held_out_targets_m)
    length_seen_mask = target_membership_mask(length_eval.targets_m, length_seen_targets)
    length_held_mask = target_membership_mask(length_eval.targets_m, length_held_targets)

    return [
        CrossedPanel(
            name="direction_effect_seen_lengths",
            title="Direction effect at seen lengths",
            target_config=direction_cfg,
            primary_label="Seen directions, seen lengths",
            secondary_label="Held-out directions, seen lengths",
            primary_color="#64748b",
            secondary_color="#f97316",
            primary_targets_m=direction_seen_targets,
            secondary_targets_m=direction_held_targets,
            evaluation=direction_eval,
            primary_profile=profile_from_values(
                RunRef(ref.experiment, f"{ref.run_id}::seen_dirs_seen_lengths", "", "#64748b"),
                direction_eval.time_s,
                direction_eval.normalized_values,
                direction_eval.target_distance,
                direction_seen_mask,
            ),
            secondary_profile=profile_from_values(
                RunRef(
                    ref.experiment,
                    f"{ref.run_id}::held_dirs_seen_lengths",
                    "",
                    "#f97316",
                ),
                direction_eval.time_s,
                direction_eval.normalized_values,
                direction_eval.target_distance,
                direction_held_mask,
            ),
        ),
        CrossedPanel(
            name="length_effect_seen_directions",
            title="Length effect at seen directions",
            target_config=length_cfg,
            primary_label="Seen lengths, seen directions",
            secondary_label="Held-out lengths, seen directions",
            primary_color="#64748b",
            secondary_color="#7c3aed",
            primary_targets_m=length_seen_targets,
            secondary_targets_m=length_held_targets,
            evaluation=length_eval,
            primary_profile=profile_from_values(
                RunRef(ref.experiment, f"{ref.run_id}::seen_lengths_seen_dirs", "", "#64748b"),
                length_eval.time_s,
                length_eval.normalized_values,
                length_eval.target_distance,
                length_seen_mask,
            ),
            secondary_profile=profile_from_values(
                RunRef(ref.experiment, f"{ref.run_id}::held_lengths_seen_dirs", "", "#7c3aed"),
                length_eval.time_s,
                length_eval.normalized_values,
                length_eval.target_distance,
                length_held_mask,
            ),
        ),
    ]


def add_band_trace(fig: go.Figure, profile: VelocityProfile) -> None:
    """Add a mean velocity trace with a one-standard-deviation band."""

    upper = profile.mean + profile.std
    lower = profile.mean - profile.std
    color = profile.run.color
    legend_group = f"{profile.run.experiment}::{profile.run.run_id}"
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.13),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            legendgroup=legend_group,
            name=profile.run.label,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile.time_s,
            y=profile.mean,
            mode="lines",
            line={"color": color, "width": 2.5},
            legendgroup=legend_group,
            name=profile.run.label,
        )
    )


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to a Plotly rgba color."""

    color = color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def write_outputs(profiles: list[VelocityProfile]) -> dict[str, Any]:
    """Write figure, data, manifest, and note outputs."""

    figure_dir = mkdir_p(figure_artifact_dir(EXPERIMENT, TOPIC))
    spec_dir = mkdir_p(figure_spec_dir(EXPERIMENT, TOPIC))
    notes_dir = mkdir_p(REPO_ROOT / "results" / EXPERIMENT / "notes")

    fig = go.Figure()
    for profile in profiles:
        add_band_trace(fig, profile)
    fig.update_layout(
        title="Nominal length-normalized target-radial velocity profiles",
        width=960,
        height=560,
        margin={"l": 72, "r": 24, "t": 72, "b": 68},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.22, "x": 0.0, "groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Time (s)", zeroline=False)
    fig.update_yaxes(title_text="Target-radial velocity / reach length (1/s)", zeroline=True)

    html_path = figure_dir / "nominal_forward_velocity_profiles.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    data_path = figure_dir / "nominal_forward_velocity_profiles.npz"
    np.savez_compressed(
        data_path,
        **{
            f"{profile.run.experiment}__{profile.run.run_id}__time_s": profile.time_s
            for profile in profiles
        },
        **{
            f"{profile.run.experiment}__{profile.run.run_id}__mean": profile.mean
            for profile in profiles
        },
        **{
            f"{profile.run.experiment}__{profile.run.run_id}__std": profile.std
            for profile in profiles
        },
    )

    rows = [
        {
            "experiment": profile.run.experiment,
            "run_id": profile.run.run_id,
            "label": profile.run.label,
            "n_replicates": profile.n_replicates,
            "n_trials": profile.n_trials,
            "n_pooled_profiles": profile.n_replicates * profile.n_trials,
            "target_distance_min_m": profile.target_distance_min_m,
            "target_distance_max_m": profile.target_distance_max_m,
            "peak_mean_length_normalized_forward_velocity_1_s": (
                profile.peak_mean_length_normalized_forward_velocity_1_s
            ),
            "time_of_peak_mean_forward_velocity_s": profile.time_of_peak_mean_forward_velocity_s,
        }
        for profile in profiles
    ]
    manifest = {
        "schema_version": "rlrmp.e901a20.nominal_velocity_profile_comparison.v1",
        "figure": repo_relative(html_path),
        "data": repo_relative(data_path),
        "evaluation_lens": "nominal_clean_validation_trials",
        "velocity_definition": (
            "effector velocity projected onto each trial's target direction, divided by "
            "that trial's reach length"
        ),
        "runs": rows,
    }
    manifest_path = figure_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_link = spec_dir / "figure.html"
    if figure_link.exists() or figure_link.is_symlink():
        figure_link.unlink()
    figure_link.symlink_to(os.path.relpath(html_path, start=figure_link.parent))
    spec = {
        "schema_version": "rlrmp.figure_spec.v1",
        "topic": TOPIC,
        "source_script": repo_relative(Path(__file__)),
        "manifest": repo_relative(manifest_path),
        "figure": repo_relative(html_path),
        "figure_link": repo_relative(figure_link),
        "data": repo_relative(data_path),
        "evaluation_lens": manifest["evaluation_lens"],
        "velocity_definition": manifest["velocity_definition"],
        "runs": rows,
        "inputs": [
            {
                "run_spec": repo_relative(run_spec_path(ref.experiment, ref.run_id)),
                "trained_model": repo_relative(
                    resolve_run_artifact_path(artifact_dir(ref), "trained_model.eqx")
                ),
            }
            for ref in RUNS
        ],
    }
    spec_path = spec_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table_lines = [
        "| Row | Peak mean normalized velocity (1/s) | Time of peak (s) | Reach lengths (m) | Pooled profiles |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| `{row['label']}` | "
            f"{row['peak_mean_length_normalized_forward_velocity_1_s']:.4f} | "
            f"{row['time_of_peak_mean_forward_velocity_s']:.3f} | "
            f"{row['target_distance_min_m']:.2f}-{row['target_distance_max_m']:.2f} | "
            f"{row['n_pooled_profiles']} |"
        )
    note = "\n".join(
        [
            "## Nominal velocity profile comparison",
            "",
            "Nominal-clean validation trials with perturbation inputs zeroed. Curves show "
            "target-radial velocity divided by trial reach length, then pooled over replicates "
            "and validation trials. Bands are one standard deviation over the pooled "
            "length-normalized profiles.",
            "",
            *table_lines,
            "",
            f"- Figure: `{repo_relative(html_path)}`",
            f"- Data: `{repo_relative(data_path)}`",
            f"- Manifest: `{repo_relative(manifest_path)}`",
            "",
        ]
    )
    note_path = notes_dir / f"{TOPIC}.md"
    update_marked_section(note_path, NOMINAL_MARKER, note)
    manifest["note"] = repo_relative(note_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_no_pgd_split_outputs(profiles: list[VelocityProfile]) -> dict[str, Any]:
    """Write no-PGD held-out split figure, data, manifest, and note outputs."""

    figure_dir = mkdir_p(figure_artifact_dir(EXPERIMENT, NO_PGD_SPLIT_TOPIC))
    spec_dir = mkdir_p(figure_spec_dir(EXPERIMENT, NO_PGD_SPLIT_TOPIC))
    notes_dir = mkdir_p(REPO_ROOT / "results" / EXPERIMENT / "notes")

    fig = go.Figure()
    for profile in profiles:
        add_band_trace(fig, profile)
    fig.update_layout(
        title="No-PGD nominal velocity: non-held-out vs held-out validation targets",
        width=960,
        height=560,
        margin={"l": 72, "r": 24, "t": 72, "b": 68},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.22, "x": 0.0, "groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Time (s)", zeroline=False)
    fig.update_yaxes(title_text="Target-radial velocity / reach length (1/s)", zeroline=True)

    html_path = figure_dir / "no_pgd_heldout_split.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    data_path = figure_dir / "no_pgd_heldout_split.npz"
    np.savez_compressed(
        data_path,
        **{f"{profile.run.label}__time_s": profile.time_s for profile in profiles},
        **{f"{profile.run.label}__mean": profile.mean for profile in profiles},
        **{f"{profile.run.label}__std": profile.std for profile in profiles},
    )

    rows = [
        {
            "label": profile.run.label,
            "n_replicates": profile.n_replicates,
            "n_trials": profile.n_trials,
            "n_pooled_profiles": profile.n_replicates * profile.n_trials,
            "target_distance_min_m": profile.target_distance_min_m,
            "target_distance_max_m": profile.target_distance_max_m,
            "peak_mean_length_normalized_forward_velocity_1_s": (
                profile.peak_mean_length_normalized_forward_velocity_1_s
            ),
            "time_of_peak_mean_forward_velocity_s": profile.time_of_peak_mean_forward_velocity_s,
        }
        for profile in profiles
    ]
    manifest = {
        "schema_version": "rlrmp.e901a20.no_pgd_heldout_split.v1",
        "figure": repo_relative(html_path),
        "data": repo_relative(data_path),
        "evaluation_lens": "nominal_clean_validation_trials",
        "source_run": {
            "experiment": NO_PGD_REF.experiment,
            "run_id": NO_PGD_REF.run_id,
            "label": NO_PGD_REF.label,
        },
        "split_definition": (
            "held_out_targets_m from the no-PGD run spec versus all remaining validation targets"
        ),
        "velocity_definition": (
            "effector velocity projected onto each trial's target direction, divided by "
            "that trial's reach length"
        ),
        "band_definition": (
            "one standard deviation over replicate x validation-target profiles within each split"
        ),
        "splits": rows,
    }
    manifest_path = figure_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_link = spec_dir / "figure.html"
    if figure_link.exists() or figure_link.is_symlink():
        figure_link.unlink()
    figure_link.symlink_to(os.path.relpath(html_path, start=figure_link.parent))
    spec = {
        "schema_version": "rlrmp.figure_spec.v1",
        "topic": NO_PGD_SPLIT_TOPIC,
        "source_script": repo_relative(Path(__file__)),
        "manifest": repo_relative(manifest_path),
        "figure": repo_relative(html_path),
        "figure_link": repo_relative(figure_link),
        "data": repo_relative(data_path),
        "evaluation_lens": manifest["evaluation_lens"],
        "split_definition": manifest["split_definition"],
        "velocity_definition": manifest["velocity_definition"],
        "band_definition": manifest["band_definition"],
        "source_run": manifest["source_run"],
        "splits": rows,
        "inputs": [
            {
                "run_spec": repo_relative(run_spec_path(NO_PGD_REF.experiment, NO_PGD_REF.run_id)),
                "trained_model": repo_relative(
                    resolve_run_artifact_path(artifact_dir(NO_PGD_REF), "trained_model.eqx")
                ),
            }
        ],
    }
    spec_path = spec_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table_lines = [
        "| Split | Peak mean normalized velocity (1/s) | Time of peak (s) | Reach lengths (m) | Pooled profiles |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| `{row['label']}` | "
            f"{row['peak_mean_length_normalized_forward_velocity_1_s']:.4f} | "
            f"{row['time_of_peak_mean_forward_velocity_s']:.3f} | "
            f"{row['target_distance_min_m']:.2f}-{row['target_distance_max_m']:.2f} | "
            f"{row['n_pooled_profiles']} |"
        )
    note = "\n".join(
        [
            "## No-PGD held-out split",
            "",
            "Nominal-clean validation trials for the 020a65b no-PGD H0 comparator only. "
            "Curves use the same target-radial velocity divided by reach length as the "
            "four-row comparison, but split validation targets into held-out diagonal "
            "targets and all remaining validation targets. Bands are one standard "
            "deviation over pooled replicate x validation-target profiles within each split.",
            "",
            *table_lines,
            "",
            f"- Figure: `{repo_relative(html_path)}`",
            f"- Data: `{repo_relative(data_path)}`",
            f"- Manifest: `{repo_relative(manifest_path)}`",
            "",
        ]
    )
    note_path = notes_dir / f"{NO_PGD_SPLIT_TOPIC}.md"
    update_marked_section(note_path, NO_PGD_SPLIT_MARKER, note)
    manifest["note"] = repo_relative(note_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def profile_summary_row(label: str, profile: VelocityProfile) -> dict[str, Any]:
    """Return manifest metadata for one profile."""

    return {
        "label": label,
        "n_replicates": profile.n_replicates,
        "n_trials": profile.n_trials,
        "n_pooled_profiles": profile.n_replicates * profile.n_trials,
        "target_distance_min_m": profile.target_distance_min_m,
        "target_distance_max_m": profile.target_distance_max_m,
        "peak_mean_length_normalized_forward_velocity_1_s": (
            profile.peak_mean_length_normalized_forward_velocity_1_s
        ),
        "time_of_peak_mean_forward_velocity_s": profile.time_of_peak_mean_forward_velocity_s,
    }


def target_summary_rows(panel: CrossedPanel) -> list[dict[str, Any]]:
    """Return per-target peak metadata for one crossed panel."""

    primary_mask = target_membership_mask(panel.evaluation.targets_m, panel.primary_targets_m)
    secondary_mask = target_membership_mask(panel.evaluation.targets_m, panel.secondary_targets_m)
    rows = []
    for idx, target in enumerate(panel.evaluation.targets_m):
        if primary_mask[idx]:
            group = panel.primary_label
        elif secondary_mask[idx]:
            group = panel.secondary_label
        else:
            group = "unmatched"
        mean_curve = np.mean(panel.evaluation.normalized_values[:, idx, :], axis=0)
        peak_idx = int(np.nanargmax(mean_curve))
        rows.append(
            {
                "panel": panel.name,
                "target_index": idx,
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "target_radius_m": float(panel.evaluation.target_distance[idx]),
                "target_angle_deg": float(target_angle_deg(panel.evaluation.targets_m[[idx]])[0]),
                "group": group,
                "peak_mean_length_normalized_forward_velocity_1_s": float(mean_curve[peak_idx]),
                "time_of_peak_mean_forward_velocity_s": float(panel.evaluation.time_s[peak_idx]),
            }
        )
    return rows


def add_panel_trace(
    fig: go.Figure,
    *,
    profile: VelocityProfile,
    label: str,
    color: str,
    legend_group: str,
    row: int,
    col: int = 1,
) -> None:
    """Add a profile trace and band to a subplot panel."""

    upper = profile.mean + profile.std
    lower = profile.mean - profile.std
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.13),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            legendgroup=legend_group,
            name=label,
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=profile.time_s,
            y=profile.mean,
            mode="lines",
            line={"color": color, "width": 2.5},
            legendgroup=legend_group,
            name=label,
        ),
        row=row,
        col=col,
    )


def write_no_pgd_crossed_outputs(panels: list[CrossedPanel]) -> dict[str, Any]:
    """Write crossed direction/length diagnostic outputs for the no-PGD row."""

    figure_dir = mkdir_p(figure_artifact_dir(EXPERIMENT, NO_PGD_CROSSED_TOPIC))
    spec_dir = mkdir_p(figure_spec_dir(EXPERIMENT, NO_PGD_CROSSED_TOPIC))
    notes_dir = mkdir_p(REPO_ROOT / "results" / EXPERIMENT / "notes")

    fig = profile_comparison_grid(
        len(panels),
        subplot_titles=[panel.title for panel in panels],
        vertical_spacing=0.12,
    )
    for row_idx, panel in enumerate(panels, start=1):
        add_panel_trace(
            fig,
            profile=panel.primary_profile,
            label=panel.primary_label,
            color=panel.primary_color,
            legend_group=f"{panel.name}::primary",
            row=row_idx,
        )
        add_panel_trace(
            fig,
            profile=panel.secondary_profile,
            label=panel.secondary_label,
            color=panel.secondary_color,
            legend_group=f"{panel.name}::secondary",
            row=row_idx,
        )
    fig.update_layout(
        title="No-PGD crossed target-grid velocity profiles",
        width=980,
        height=760,
        margin={"l": 72, "r": 24, "t": 82, "b": 76},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.14, "x": 0.0, "groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Time (s)", zeroline=False)
    fig.update_yaxes(title_text="Target-radial velocity / reach length (1/s)", zeroline=True)

    html_path = figure_dir / "no_pgd_crossed_target_grid.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    data_path = figure_dir / "no_pgd_crossed_target_grid.npz"
    data_payload: dict[str, Any] = {}
    for panel in panels:
        prefix = panel.name
        primary_mask = target_membership_mask(panel.evaluation.targets_m, panel.primary_targets_m)
        secondary_mask = target_membership_mask(panel.evaluation.targets_m, panel.secondary_targets_m)
        group = np.full(panel.evaluation.targets_m.shape[0], "", dtype="<U64")
        group[primary_mask] = panel.primary_label
        group[secondary_mask] = panel.secondary_label
        data_payload[f"{prefix}__time_s"] = panel.evaluation.time_s
        data_payload[f"{prefix}__values"] = panel.evaluation.normalized_values
        data_payload[f"{prefix}__targets_m"] = panel.evaluation.targets_m
        data_payload[f"{prefix}__target_radius_m"] = panel.evaluation.target_distance
        data_payload[f"{prefix}__target_angle_deg"] = target_angle_deg(panel.evaluation.targets_m)
        data_payload[f"{prefix}__target_group"] = group
        data_payload[f"{prefix}__primary_mean"] = panel.primary_profile.mean
        data_payload[f"{prefix}__primary_std"] = panel.primary_profile.std
        data_payload[f"{prefix}__secondary_mean"] = panel.secondary_profile.mean
        data_payload[f"{prefix}__secondary_std"] = panel.secondary_profile.std
    np.savez_compressed(data_path, **data_payload)

    panel_rows = []
    target_rows = []
    for panel in panels:
        panel_rows.append(
            {
                "name": panel.name,
                "title": panel.title,
                "primary": profile_summary_row(panel.primary_label, panel.primary_profile),
                "secondary": profile_summary_row(panel.secondary_label, panel.secondary_profile),
                "seen_directions_deg": list(panel.target_config.seen_directions_deg),
                "held_out_directions_deg": list(panel.target_config.held_out_directions_deg),
                "seen_amplitudes_m": list(panel.target_config.seen_amplitudes_m),
                "held_out_amplitudes_m": list(panel.target_config.held_out_amplitudes_m),
            }
        )
        target_rows.extend(target_summary_rows(panel))

    manifest = {
        "schema_version": "rlrmp.e901a20.no_pgd_crossed_target_grid.v1",
        "figure": repo_relative(html_path),
        "data": repo_relative(data_path),
        "evaluation_lens": "nominal_clean_custom_target_grid_final_checkpoint",
        "source_run": {
            "experiment": NO_PGD_REF.experiment,
            "run_id": NO_PGD_REF.run_id,
            "label": NO_PGD_REF.label,
        },
        "velocity_definition": (
            "effector velocity projected onto each trial's target direction, divided by "
            "that trial's reach length"
        ),
        "band_definition": (
            "one standard deviation over replicate x custom-target profiles within each curve"
        ),
        "checkpoint_policy": "final trained_model.eqx, matching the current split figure",
        "stochastic_runtime_policy": "unchanged from the current split figure",
        "panels": panel_rows,
        "targets": target_rows,
    }
    manifest_path = figure_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_link = spec_dir / "figure.html"
    if figure_link.exists() or figure_link.is_symlink():
        figure_link.unlink()
    figure_link.symlink_to(os.path.relpath(html_path, start=figure_link.parent))
    spec = {
        "schema_version": "rlrmp.figure_spec.v1",
        "topic": NO_PGD_CROSSED_TOPIC,
        "source_script": repo_relative(Path(__file__)),
        "manifest": repo_relative(manifest_path),
        "figure": repo_relative(html_path),
        "figure_link": repo_relative(figure_link),
        "data": repo_relative(data_path),
        "evaluation_lens": manifest["evaluation_lens"],
        "velocity_definition": manifest["velocity_definition"],
        "band_definition": manifest["band_definition"],
        "checkpoint_policy": manifest["checkpoint_policy"],
        "stochastic_runtime_policy": manifest["stochastic_runtime_policy"],
        "source_run": manifest["source_run"],
        "panels": panel_rows,
        "targets": target_rows,
        "inputs": [
            {
                "run_spec": repo_relative(run_spec_path(NO_PGD_REF.experiment, NO_PGD_REF.run_id)),
                "trained_model": repo_relative(
                    resolve_run_artifact_path(artifact_dir(NO_PGD_REF), "trained_model.eqx")
                ),
            }
        ],
    }
    spec_path = spec_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table_lines = [
        "| Panel | Curve | Peak mean normalized velocity (1/s) | Time of peak (s) | Reach lengths (m) | Pooled profiles |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for panel in panel_rows:
        for key in ("primary", "secondary"):
            row = panel[key]
            table_lines.append(
                f"| {panel['title']} | `{row['label']}` | "
                f"{row['peak_mean_length_normalized_forward_velocity_1_s']:.4f} | "
                f"{row['time_of_peak_mean_forward_velocity_s']:.3f} | "
                f"{row['target_distance_min_m']:.2f}-{row['target_distance_max_m']:.2f} | "
                f"{row['n_pooled_profiles']} |"
            )
    note = "\n".join(
        [
            "## No-PGD crossed target grid",
            "",
            "Nominal-clean custom target-grid trials for the 020a65b no-PGD H0 comparator. "
            "The first panel compares seen versus held-out directions while holding reach "
            "lengths to the seen training lengths (`0.10 m`, `0.15 m`). The second panel "
            "compares seen versus held-out lengths while holding directions to the seen "
            "training directions (`0, 60, 120, 180, 240, 300 deg`).",
            "",
            "This uses the final checkpoint and the same stochastic runtime behavior as the "
            "current split figure; it is intended to separate direction and length support "
            "before any validation-selected checkpoint follow-up.",
            "",
            *table_lines,
            "",
            f"- Figure: `{repo_relative(html_path)}`",
            f"- Data: `{repo_relative(data_path)}`",
            f"- Manifest: `{repo_relative(manifest_path)}`",
            "",
        ]
    )
    note_path = notes_dir / f"{NO_PGD_CROSSED_TOPIC}.md"
    update_marked_section(note_path, NO_PGD_CROSSED_MARKER, note)
    manifest["note"] = repo_relative(note_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def add_companion_trace(fig: go.Figure, profile: CompanionProfile) -> None:
    """Add a companion mean trace with a paired one-SD band."""

    upper = profile.mean + profile.std
    lower = profile.mean - profile.std
    color = profile.run.color
    legend_group = f"{profile.run.experiment}::{profile.run.run_id}"
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.13),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            legendgroup=legend_group,
            name=profile.run.label,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile.time_s,
            y=profile.mean,
            mode="lines",
            line={"color": color, "width": 2.5},
            legendgroup=legend_group,
            name=profile.run.label,
        )
    )


def companion_profile_row(profile: CompanionProfile) -> dict[str, Any]:
    """Return JSON-ready summary for one companion profile."""

    return {
        "experiment": profile.run.experiment,
        "run_id": profile.run.run_id,
        "label": profile.run.label,
        "n_replicates": profile.n_replicates,
        "n_target_conditions": profile.n_target_conditions,
        "n_rollout_repeats_per_target_per_replicate": profile.n_rollout_repeats,
        "n_pooled_profiles": profile.n_replicates
        * profile.n_target_conditions
        * profile.n_rollout_repeats,
        "target_distance_min_m": profile.target_distance_min_m,
        "target_distance_max_m": profile.target_distance_max_m,
        "peak_mean_velocity": profile.peak_mean_velocity,
        "time_of_peak_mean_velocity_s": profile.time_of_peak_mean_velocity_s,
        "selected_checkpoints": [
            selection.to_json(repo_root=REPO_ROOT) for selection in profile.selected_checkpoints
        ],
    }


def write_old_compatible_outputs(
    profiles: list[CompanionProfile],
    *,
    topic: str,
    marker: str,
    title: str,
    yaxis_title: str,
    velocity_definition: str,
    evaluation_lens: str,
    unit_label: str,
) -> dict[str, Any]:
    """Write one old-compatible companion figure, data, manifest, spec, and note."""

    figure_dir = mkdir_p(figure_artifact_dir(EXPERIMENT, topic))
    spec_dir = mkdir_p(figure_spec_dir(EXPERIMENT, topic))
    notes_dir = mkdir_p(REPO_ROOT / "results" / EXPERIMENT / "notes")

    fig = go.Figure()
    for profile in profiles:
        add_companion_trace(fig, profile)
    fig.update_layout(
        title=title,
        width=960,
        height=560,
        margin={"l": 72, "r": 24, "t": 72, "b": 68},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.22, "x": 0.0, "groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Time (s)", zeroline=False)
    fig.update_yaxes(title_text=yaxis_title, zeroline=True)

    html_path = figure_dir / f"{topic}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    data_path = figure_dir / f"{topic}.npz"
    np.savez_compressed(
        data_path,
        **{
            f"{profile.run.experiment}__{profile.run.run_id}__time_s": profile.time_s
            for profile in profiles
        },
        **{
            f"{profile.run.experiment}__{profile.run.run_id}__mean": profile.mean
            for profile in profiles
        },
        **{
            f"{profile.run.experiment}__{profile.run.run_id}__std": profile.std
            for profile in profiles
        },
    )

    rows = [companion_profile_row(profile) for profile in profiles]
    feedbax_commit = os.environ.get("RLRMP_COMPAT_FEEDBAX_COMMIT")
    runtime_provenance = {
        "feedbax_package": feedbax.__name__,
        "feedbax_runtime": (
            "feedbax git archive"
            if feedbax_commit
            else "current Python import path"
        ),
        "feedbax_commit": feedbax_commit,
        "runtime_note": (
            "Old-compatible velocity figures should be generated with pre-1e1c94f5 "
            "Feedbax network semantics for legacy MaskedLinear readout checkpoints."
        ),
    }
    manifest = {
        "schema_version": f"rlrmp.e901a20.{topic}.v1",
        "figure": repo_relative(html_path),
        "data": repo_relative(data_path),
        "evaluation_lens": evaluation_lens,
        "velocity_definition": velocity_definition,
        "band_definition": (
            "one standard deviation over pooled replicate x target-condition x stochastic "
            "rollout profiles"
        ),
        "checkpoint_policy": "validation_selected_per_replicate_sparse_history",
        "stochastic_runtime_policy": (
            f"{OLD_COMPAT_N_ROLLOUT_REPEATS} stochastic repeats per target condition per "
            "replicate, using jr.split(PRNGKey(0), n_replicates)"
        ),
        "runtime_provenance": runtime_provenance,
        "runs": rows,
    }
    manifest_path = figure_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_link = spec_dir / "figure.html"
    if figure_link.exists() or figure_link.is_symlink():
        figure_link.unlink()
    figure_link.symlink_to(os.path.relpath(html_path, start=figure_link.parent))
    spec = {
        "schema_version": "rlrmp.figure_spec.v1",
        "topic": topic,
        "source_script": repo_relative(Path(__file__)),
        "manifest": repo_relative(manifest_path),
        "figure": repo_relative(html_path),
        "figure_link": repo_relative(figure_link),
        "data": repo_relative(data_path),
        "evaluation_lens": evaluation_lens,
        "velocity_definition": velocity_definition,
        "band_definition": manifest["band_definition"],
        "checkpoint_policy": manifest["checkpoint_policy"],
        "stochastic_runtime_policy": manifest["stochastic_runtime_policy"],
        "runtime_provenance": runtime_provenance,
        "runs": rows,
        "inputs": [
            {
                "run_spec": repo_relative(run_spec_path(profile.run.experiment, profile.run.run_id)),
                "artifact_dir": repo_relative(artifact_dir(profile.run)),
            }
            for profile in profiles
        ],
    }
    spec_path = spec_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table_lines = [
        f"| Row | Peak mean velocity ({unit_label}) | Time of peak (s) | Targets | Repeats/target/rep | Pooled profiles |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| `{row['label']}` | "
            f"{row['peak_mean_velocity']:.4f} | "
            f"{row['time_of_peak_mean_velocity_s']:.3f} | "
            f"{row['n_target_conditions']} | "
            f"{row['n_rollout_repeats_per_target_per_replicate']} | "
            f"{row['n_pooled_profiles']} |"
        )
    note = "\n".join(
        [
            f"## {title}",
            "",
            velocity_definition,
            "",
            "All rows use validation-selected per-replicate checkpoints and 64 stochastic "
            "rollout repeats per target condition, matching the old pilot-figure checkpoint "
            "and repeat convention while keeping the requested target set for this panel.",
            "",
            "Runtime provenance: generated with "
            f"`{runtime_provenance['feedbax_runtime']}`"
            + (
                f" at compatibility commit `{runtime_provenance['feedbax_commit']}`."
                if runtime_provenance["feedbax_commit"]
                else "."
            ),
            "",
            *table_lines,
            "",
            f"- Figure: `{repo_relative(html_path)}`",
            f"- Data: `{repo_relative(data_path)}`",
            f"- Manifest: `{repo_relative(manifest_path)}`",
            "",
        ]
    )
    note_path = notes_dir / f"{topic}.md"
    update_marked_section(note_path, marker, note)
    manifest["note"] = repo_relative(note_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Evaluate all rows and materialize outputs."""

    profiles = [evaluate_profile(ref) for ref in RUNS]
    comparison_manifest = write_outputs(profiles)
    no_pgd_split_manifest = write_no_pgd_split_outputs(evaluate_no_pgd_heldout_split())
    no_pgd_crossed_manifest = write_no_pgd_crossed_outputs(evaluate_no_pgd_crossed_panels())
    first_target_manifest = write_old_compatible_outputs(
        evaluate_old_compatible_first_target_profiles(),
        topic=OLD_COMPAT_FIRST_TARGET_TOPIC,
        marker=OLD_COMPAT_FIRST_TARGET_MARKER,
        title="Old-compatible first-target velocity profiles",
        yaxis_title="Forward velocity (m/s)",
        velocity_definition=(
            "First validation target only, repeated 64 times per replicate; curves show raw "
            "x-axis effector velocity in m/s."
        ),
        evaluation_lens="validation_selected_first_target_stochastic_raw_x_velocity",
        unit_label="m/s",
    )
    all_target_manifest = write_old_compatible_outputs(
        evaluate_old_compatible_all_target_profiles(),
        topic=OLD_COMPAT_ALL_TARGET_TOPIC,
        marker=OLD_COMPAT_ALL_TARGET_MARKER,
        title="Old-compatible all-target aligned velocity profiles",
        yaxis_title="Target-radial velocity / reach length (1/s)",
        velocity_definition=(
            "All nominal validation targets, each repeated 64 times per replicate; curves "
            "show effector velocity projected onto each trial's target direction and divided "
            "by that trial's reach length."
        ),
        evaluation_lens="validation_selected_all_targets_stochastic_target_radial_length_normalized",
        unit_label="1/s",
    )
    print(
        json.dumps(
            {
                "all_target_aligned_old_compatible": all_target_manifest,
                "comparison": comparison_manifest,
                "first_target_old_compatible": first_target_manifest,
                "no_pgd_crossed_target_grid": no_pgd_crossed_manifest,
                "no_pgd_heldout_split": no_pgd_split_manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
