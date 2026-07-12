"""Paired broad-epsilon attribution diagnostics for C&S GRU runs."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax import TaskTrialSpec, WhereDict
from feedbax.runtime.batch import BatchInfo
from jax_cookbook.tree import filter_spec_leaves
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from jax_cookbook import load_with_hyperparameters

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.eval.checkpoint_selection import (
    ReplicateCheckpointSelection,
    load_validation_selected_checkpoint_model,
    build_validation_checkpoint_selection_manifest,
)
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT, mkdir_p, resolve_run_artifact_path
from rlrmp.paths import run_spec_path as tracked_run_spec_path
from rlrmp.runtime.run_spec_access import require_run_seed
from rlrmp.runtime.run_specs import resolve_run_record
from rlrmp.train.cs_nominal_gru import _where_train
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_TRAINING_MODE,
    BroadFullStateEpsilonTrainingTaskAdapter,
    FixedTargetPerturbationTrainingTaskAdapter,
    apply_training_perturbation_mixture,
)
from rlrmp.train.task_model import setup_task_model_pair

SCHEMA_VERSION = "rlrmp.gru_broad_epsilon_attribution.v1"
DEFAULT_EXPERIMENT = "b8aa38e"
DEFAULT_OUTPUT_TAG = "gru_broad_epsilon_attribution"


@dataclass(frozen=True)
class RunAttributionInputs:
    """Resolved input files and parsed spec for one GRU row."""

    run_id: str
    run_spec_path: Path
    artifact_dir: Path
    run_spec: dict[str, Any]


def materialize_broad_epsilon_attribution(
    *,
    experiment: str = DEFAULT_EXPERIMENT,
    run_ids: Sequence[str] | None = None,
    output_tag: str = DEFAULT_OUTPUT_TAG,
    n_rollout_trials: int = 8,
    max_runs: int | None = None,
    include_smoke: bool = False,
    use_validation_selected_checkpoints: bool = True,
    max_gradient_replicates: int = 1,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize paired active-vs-zero broad-epsilon diagnostics.

    The active condition is the run's own training sampler, including
    broad/full-state epsilon. The paired condition replays the same target and
    calibrated-perturbation sampler branches but removes only the broad-epsilon
    draw, so graph-channel perturbations and any base process-epsilon inputs are
    preserved.
    """

    if n_rollout_trials < 1:
        raise ValueError("n_rollout_trials must be at least 1")
    if max_gradient_replicates < 0:
        raise ValueError("max_gradient_replicates must be non-negative")

    selected_run_ids = tuple(
        run_ids
        or discover_broad_epsilon_run_ids(
            experiment,
            repo_root,
            include_smoke=include_smoke,
        )
    )
    if max_runs is not None:
        selected_run_ids = selected_run_ids[: int(max_runs)]
    runs = resolve_run_inputs(experiment, selected_run_ids, repo_root=repo_root)
    selection_manifest = (
        build_validation_checkpoint_selection_manifest(
            experiment=experiment,
            run_ids=tuple(run.run_id for run in runs),
            repo_root=repo_root,
        ).model_dump(mode="json", exclude_none=True)
        if use_validation_selected_checkpoints and runs
        else None
    )
    bulk_dir = repo_root / "_artifacts" / experiment / "broad_epsilon_attribution" / output_tag
    notes_dir = repo_root / "results" / experiment / "notes"
    mkdir_p(bulk_dir)
    mkdir_p(notes_dir)

    rows = []
    for run in runs:
        row = evaluate_run_broad_epsilon_attribution(
            run,
            experiment=experiment,
            n_rollout_trials=n_rollout_trials,
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
            max_gradient_replicates=max_gradient_replicates,
            bulk_dir=bulk_dir,
            repo_root=repo_root,
        )
        rows.append(row)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "output_tag": output_tag,
        "active_vs_zero_semantics": active_vs_zero_semantics(),
        "run_selection": {
            "requested_run_ids": list(selected_run_ids),
            "n_runs": len(rows),
            "selector": (
                "run hps.broad_epsilon_training.enabled == true"
                + (" including smoke rows" if include_smoke else " excluding smoke rows")
                if run_ids is None
                else "explicit run_ids"
            ),
        },
        "checkpoint_policy": (
            "validation_selected_per_replicate"
            if use_validation_selected_checkpoints
            else "final_checkpoint"
        ),
        "checkpoint_selection": selection_manifest,
        "gradient_policy": gradient_policy(max_gradient_replicates),
        "bulk_dir": repo_relative(bulk_dir, repo_root=repo_root),
        "summary": summarize_manifest_rows(rows),
        "rows": rows,
    }
    json_path = notes_dir / f"{output_tag}.json"
    md_path = notes_dir / f"{output_tag}.md"
    csv_path = notes_dir / f"{output_tag}.csv"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path.write_text(render_summary_csv(rows), encoding="utf-8")
    update_marked_section(
        md_path,
        "gru_broad_epsilon_attribution",
        render_markdown(manifest, csv_path=csv_path, repo_root=repo_root),
    )
    manifest["outputs"] = {
        "json": repo_relative(json_path, repo_root=repo_root),
        "markdown": repo_relative(md_path, repo_root=repo_root),
        "csv": repo_relative(csv_path, repo_root=repo_root),
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def evaluate_run_broad_epsilon_attribution(
    run: RunAttributionInputs,
    *,
    experiment: str,
    n_rollout_trials: int,
    use_validation_selected_checkpoints: bool,
    max_gradient_replicates: int,
    bulk_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Evaluate one row on paired sampled-epsilon and zero-epsilon trials."""

    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = require_run_seed(run.run_spec, source=run.run_spec_path)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model, checkpoint_selection = load_model_for_run(
        run,
        experiment=experiment,
        hps=hps,
        use_validation_selected_checkpoints=use_validation_selected_checkpoints,
        repo_root=repo_root,
    )
    try:
        active_trial_specs, zero_trial_specs = paired_broad_epsilon_training_specs(
            pair.task,
            key=jr.PRNGKey(seed + 31),
            n_trials=n_rollout_trials,
        )
    except ValueError as exc:
        return {
            "run_id": run.run_id,
            "status": "not_applicable",
            "reason": str(exc),
            "run_spec_path": repo_relative(run.run_spec_path, repo_root=repo_root),
            "artifact_dir": repo_relative(run.artifact_dir, repo_root=repo_root),
            "n_rollout_trials": int(n_rollout_trials),
            "n_replicates": n_replicates,
            "broad_epsilon_training": broad_epsilon_metadata(run.run_spec),
            "checkpoint_selection": [
                selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
            ],
        }
    active_states = evaluate_model_states(
        model=model,
        task=pair.task,
        trial_specs=active_trial_specs,
        n_replicates=n_replicates,
        seed=seed + 103,
    )
    zero_states = evaluate_model_states(
        model=model,
        task=pair.task,
        trial_specs=zero_trial_specs,
        n_replicates=n_replicates,
        seed=seed + 103,
    )
    active_loss = summarize_loss_tree(
        pair.task.loss_func(active_states, active_trial_specs, model)
    )
    zero_loss = summarize_loss_tree(
        pair.task.loss_func(zero_states, zero_trial_specs, model)
    )
    loss_delta = loss_delta_summary(active_loss, zero_loss)
    gradient = gradient_attribution_summary(
        model=model,
        task=pair.task,
        loss_func=pair.task.loss_func,
        active_trial_specs=active_trial_specs,
        zero_trial_specs=zero_trial_specs,
        n_replicates=n_replicates,
        max_gradient_replicates=max_gradient_replicates,
        seed=seed + 211,
    )
    bulk_path = write_bulk_arrays(
        bulk_dir / f"{run.run_id}.npz",
        base_trial_specs=zero_trial_specs,
        active_trial_specs=active_trial_specs,
        zero_trial_specs=zero_trial_specs,
        active_states=active_states,
        zero_states=zero_states,
    )
    base_epsilon = np.asarray(zero_trial_specs.inputs["epsilon"], dtype=np.float64)
    epsilon = np.asarray(active_trial_specs.inputs["epsilon"], dtype=np.float64)
    broad_delta = epsilon - base_epsilon
    return {
        "run_id": run.run_id,
        "status": "evaluated",
        "run_spec_path": repo_relative(run.run_spec_path, repo_root=repo_root),
        "artifact_dir": repo_relative(run.artifact_dir, repo_root=repo_root),
        "bulk_arrays": repo_relative(bulk_path, repo_root=repo_root),
        "n_rollout_trials": int(infer_batch_size(active_trial_specs)),
        "n_replicates": n_replicates,
        "broad_epsilon_training": broad_epsilon_metadata(run.run_spec),
        "epsilon": {
            "active_total": epsilon_summary(epsilon),
            "paired_without_broad": epsilon_summary(base_epsilon),
            "broad_delta": epsilon_summary(broad_delta),
        },
        "checkpoint_selection": [
            selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
        ],
        "loss": {
            "active": active_loss,
            "zero": zero_loss,
            "delta_active_minus_zero": loss_delta,
        },
        "gradient": gradient,
    }


def paired_broad_epsilon_training_specs(
    task: Any,
    *,
    key: Any,
    n_trials: int,
    batch_info: Any = None,
) -> tuple[TaskTrialSpec, TaskTrialSpec]:
    """Return same-training-batch trial specs with and without broad epsilon.

    The current training task wrappers compose broad full-state epsilon with
    calibrated graph-channel perturbation training. For attribution we need to
    compare exactly the same sampled target/perturbation batch while removing
    only the broad-epsilon contribution. This helper mirrors the wrapper PRNG
    splits rather than zeroing the entire epsilon input.
    """

    batch_info = batch_info or BatchInfo(size=int(n_trials), current=0, total=1)
    keys = jr.split(key, int(n_trials))

    def active_one(trial_key: Any) -> TaskTrialSpec:
        return task.get_train_trial_with_intervenor_params(trial_key, batch_info)

    def without_broad_one(trial_key: Any) -> TaskTrialSpec:
        return _training_specs_without_broad_epsilon(task, trial_key, batch_info)

    active = eqx.filter_vmap(active_one)(keys)
    paired_without_broad = eqx.filter_vmap(without_broad_one)(keys)
    return materialize_trial_batch(active), materialize_trial_batch(paired_without_broad)


def _training_specs_without_broad_epsilon(
    task: Any,
    key: Any,
    batch_info: Any = None,
) -> TaskTrialSpec:
    """Replay the task sampler with the broad-epsilon branch removed."""

    if isinstance(task, FixedTargetPerturbationTrainingTaskAdapter):
        key_trial, key_pert = jr.split(key)
        base_without_broad = _training_specs_without_broad_epsilon(
            task.task,
            key_trial,
            batch_info,
        )
        return apply_training_perturbation_mixture(
            base_without_broad,
            task.config,
            key_pert,
            batch_info,
        )
    if isinstance(task, BroadFullStateEpsilonTrainingTaskAdapter):
        key_base, _key_epsilon = jr.split(key)
        return task.task.get_train_trial_with_intervenor_params(key_base, batch_info)
    inner_task = getattr(task, "task", None)
    if inner_task is not None:
        return _training_specs_without_broad_epsilon(inner_task, key, batch_info)
    raise ValueError(
        "paired broad-epsilon attribution requires a "
        "BroadFullStateEpsilonTrainingTaskAdapter in the training task stack"
    )


def materialize_trial_batch(trial_specs: TaskTrialSpec) -> TaskTrialSpec:
    """Broadcast shared training-trial arrays to the explicit batch axis.

    Training samplers may return a batched set of initial states and inputs while
    leaving static target/timeline arrays unbatched. ``Task.eval_trials`` vmaps
    over the trial spec, so the attribution diagnostic materializes those shared
    leaves into a true batch before evaluation.
    """

    batch = infer_batch_size(trial_specs)

    def broadcast_leaf(leaf: Any) -> Any:
        shape = getattr(leaf, "shape", None)
        if shape is None:
            return leaf
        if len(shape) >= 1 and int(shape[0]) == batch:
            return leaf
        return jnp.broadcast_to(leaf, (batch, *shape))

    def broadcast_tree(value: Any) -> Any:
        return jt.map(broadcast_leaf, value)

    return TaskTrialSpec(
        inits=WhereDict(
            {key: broadcast_tree(value) for key, value in trial_specs.inits.items()}
        ),
        inputs={key: broadcast_tree(value) for key, value in trial_specs.inputs.items()},
        targets={key: broadcast_tree(value) for key, value in trial_specs.targets.items()},
        intervene=trial_specs.intervene,
        extra=trial_specs.extra,
        timeline=broadcast_tree(trial_specs.timeline),
    )


def discover_broad_epsilon_run_ids(
    experiment: str,
    repo_root: Path = REPO_ROOT,
    *,
    include_smoke: bool = False,
) -> tuple[str, ...]:
    """Return run IDs whose run spec declares broad-epsilon training enabled."""

    run_root = repo_root / "results" / experiment / "runs"
    if not run_root.exists():
        raise FileNotFoundError(f"Missing run-spec root: {run_root}")
    run_ids = []
    for path in sorted(run_root.glob("*.json")):
        run_id = path.stem
        run_spec = resolve_run_record(experiment, run_id, repo_root=repo_root)
        broad = broad_epsilon_metadata(run_spec)
        if broad.get("enabled") is True:
            if not include_smoke and run_id.startswith("smoke__"):
                continue
            run_ids.append(run_id)
    return tuple(run_ids)


def resolve_run_inputs(
    experiment: str,
    run_ids: Sequence[str],
    *,
    repo_root: Path = REPO_ROOT,
) -> list[RunAttributionInputs]:
    """Resolve run specs and bulk directories for the selected rows."""

    runs = []
    for run_id in run_ids:
        run_spec_path = tracked_run_spec_path(experiment, run_id, repo_root=repo_root)
        artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
        if not run_spec_path.exists():
            raise FileNotFoundError(f"Missing run spec: {run_spec_path}")
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Missing artifact directory: {artifact_dir}")
        runs.append(
            RunAttributionInputs(
                run_id=run_id,
                run_spec_path=run_spec_path,
                artifact_dir=artifact_dir,
                run_spec=resolve_run_record(experiment, run_id, repo_root=repo_root),
            )
        )
    return runs


def load_model_for_run(
    run: RunAttributionInputs,
    *,
    experiment: str,
    hps: TreeNamespace,
    use_validation_selected_checkpoints: bool,
    repo_root: Path,
) -> tuple[Any, list[ReplicateCheckpointSelection]]:
    """Load final or validation-selected checkpoint model for a run."""

    if use_validation_selected_checkpoints:
        return load_validation_selected_checkpoint_model(
            experiment=experiment,
            run_id=run.run_id,
            run_spec=run.run_spec,
            repo_root=repo_root,
        )
    model, _hyperparameters = load_with_hyperparameters(
        resolve_run_artifact_path(run.artifact_dir, "trained_model.eqx"),
        setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
    )
    return model, []


def zero_epsilon_trial_specs(trial_specs: TaskTrialSpec) -> TaskTrialSpec:
    """Return trial specs with the same batch and all inputs except zero epsilon."""

    if "epsilon" not in trial_specs.inputs:
        raise ValueError("paired broad-epsilon attribution requires an epsilon input")
    return replace_trial_input(
        trial_specs,
        "epsilon",
        jnp.zeros_like(jnp.asarray(trial_specs.inputs["epsilon"])),
    )


def truncate_trial_specs(trial_specs: TaskTrialSpec, n_trials: int) -> TaskTrialSpec:
    """Keep a bounded prefix of the sampled batch for diagnostic evaluation."""

    batch = infer_batch_size(trial_specs)
    n = min(int(n_trials), batch)
    if n == batch:
        return trial_specs

    def take_prefix(value: Any) -> Any:
        if _has_batch_prefix(value, batch):
            return value[:n]
        if not isinstance(value, (str, bytes, int, float, bool, type(None))):
            return jt.map(lambda leaf: leaf[:n] if _has_batch_prefix(leaf, batch) else leaf, value)
        return value

    return TaskTrialSpec(
        inits=WhereDict({key: take_prefix(value) for key, value in trial_specs.inits.items()}),
        inputs={key: take_prefix(value) for key, value in trial_specs.inputs.items()},
        targets={key: take_prefix(value) for key, value in trial_specs.targets.items()},
        intervene=trial_specs.intervene,
        extra=trial_specs.extra,
        timeline=take_prefix(trial_specs.timeline),
    )


def replace_trial_input(trial_specs: TaskTrialSpec, name: str, value: Any) -> TaskTrialSpec:
    """Return trial specs with one input replaced."""

    inputs = dict(trial_specs.inputs)
    inputs[name] = value
    return TaskTrialSpec(
        inits=WhereDict(trial_specs.inits),
        inputs=inputs,
        targets=trial_specs.targets,
        intervene=trial_specs.intervene,
        extra=trial_specs.extra,
        timeline=trial_specs.timeline,
    )


def evaluate_model_states(
    *,
    model: Any,
    task: Any,
    trial_specs: TaskTrialSpec,
    n_replicates: int,
    seed: int,
) -> Any:
    """Evaluate all replicate models on one shared trial spec bank."""

    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates,
    )
    n_trials = infer_batch_size(trial_specs)

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return task.eval_trials(replicate_model, trial_specs, jr.split(key, n_trials))

    return eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(seed), n_replicates),
    )


def summarize_loss_tree(loss_tree: Any) -> dict[str, Any]:
    """Return scalar total and top-level per-term weighted loss values."""

    terms = {}
    for name, child in loss_tree.items():
        terms[name] = float(np.asarray(child.total))
    return {
        "total": float(np.asarray(loss_tree.total)),
        "terms": terms,
    }


def loss_delta_summary(active: Mapping[str, Any], zero: Mapping[str, Any]) -> dict[str, Any]:
    """Return active-minus-zero loss deltas for total and all observed terms."""

    names = sorted(set(active.get("terms", {})) | set(zero.get("terms", {})))
    terms = {
        name: float(active.get("terms", {}).get(name, 0.0) - zero.get("terms", {}).get(name, 0.0))
        for name in names
    }
    return {
        "total": float(active["total"] - zero["total"]),
        "terms": terms,
    }


def gradient_attribution_summary(
    *,
    model: Any,
    task: Any,
    loss_func: Any,
    active_trial_specs: TaskTrialSpec,
    zero_trial_specs: TaskTrialSpec,
    n_replicates: int,
    max_gradient_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Compare bounded trainable-parameter gradients for active and zero epsilon."""

    if max_gradient_replicates == 0:
        return gradient_not_evaluated("max_gradient_replicates=0")

    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates,
    )
    replicate_count = min(int(max_gradient_replicates), int(n_replicates))
    rows = []
    for replicate in range(replicate_count):
        replicate_arrays = jt.map(lambda leaf: leaf[replicate], model_arrays)
        replicate_model = eqx.combine(replicate_arrays, model_other)
        active_grad = gradient_for_trial_specs(
            model=replicate_model,
            task=task,
            loss_func=loss_func,
            trial_specs=active_trial_specs,
            seed=seed + replicate,
        )
        zero_grad = gradient_for_trial_specs(
            model=replicate_model,
            task=task,
            loss_func=loss_func,
            trial_specs=zero_trial_specs,
            seed=seed + replicate,
        )
        rows.append(
            {
                "replicate": replicate,
                **gradient_pair_metrics(active_grad, zero_grad),
            }
        )
    return {
        "status": "evaluated",
        "gradient_kind": "raw_pre_optimizer_trainable_parameter_gradient",
        "update_direction_status": "not_materialized",
        "update_direction_reason": (
            "validation-selected models are assembled from per-replicate checkpoints; "
            "the paired diagnostic does not load a synchronized optimizer state, so "
            "the practical bounded attribution is raw gradient direction"
        ),
        "where_train": "rlrmp.train.cs_nominal_gru._where_train()[0]",
        "n_replicates_requested": int(max_gradient_replicates),
        "n_replicates_evaluated": replicate_count,
        "replicates": rows,
        "aggregate": aggregate_gradient_rows(rows),
    }


def gradient_for_trial_specs(
    *,
    model: Any,
    task: Any,
    loss_func: Any,
    trial_specs: TaskTrialSpec,
    seed: int,
) -> Any:
    """Return trainable-parameter gradient for one replicate and trial bank."""

    where_train = _where_train()[0]
    spec = filter_spec_leaves(model, where_train)
    trainable, static = eqx.partition(model, spec)
    n_trials = infer_batch_size(trial_specs)

    def objective(trainable_model: Any) -> Any:
        candidate = eqx.combine(trainable_model, static)
        states = task.eval_trials(candidate, trial_specs, jr.split(jr.PRNGKey(seed), n_trials))
        return loss_func(states, trial_specs, candidate).total

    return eqx.filter_grad(objective)(trainable)


def gradient_pair_metrics(active_grad: Any, zero_grad: Any) -> dict[str, float]:
    """Return norm and direction metrics for two gradient pytrees."""

    delta = jt.map(lambda active, zero: active - zero, active_grad, zero_grad)
    active_norm = pytree_l2_norm(active_grad)
    zero_norm = pytree_l2_norm(zero_grad)
    delta_norm = pytree_l2_norm(delta)
    dot = pytree_dot(active_grad, zero_grad)
    denom = max(active_norm * zero_norm, 1e-30)
    return {
        "active_gradient_norm": active_norm,
        "zero_gradient_norm": zero_norm,
        "active_minus_zero_gradient_norm": delta_norm,
        "active_zero_gradient_cosine": float(dot / denom),
        "relative_delta_norm_vs_active": float(delta_norm / max(active_norm, 1e-30)),
        "relative_delta_norm_vs_zero": float(delta_norm / max(zero_norm, 1e-30)),
    }


def aggregate_gradient_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Aggregate gradient metrics across evaluated replicate rows."""

    if not rows:
        return {}
    numeric_keys = [key for key in rows[0] if key != "replicate"]
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in numeric_keys
    }


def gradient_not_evaluated(reason: str) -> dict[str, Any]:
    """Return a manifest payload for intentionally skipped gradient work."""

    return {
        "status": "not_evaluated",
        "reason": reason,
        "gradient_kind": "raw_pre_optimizer_trainable_parameter_gradient",
        "update_direction_status": "not_materialized",
    }


def pytree_l2_norm(tree: Any) -> float:
    """Return the Euclidean norm over all array leaves in a PyTree."""

    total = 0.0
    for leaf in jt.leaves(tree):
        if eqx.is_array(leaf):
            array = np.asarray(leaf, dtype=np.float64)
            total += float(np.vdot(array, array).real)
    return float(np.sqrt(total))


def pytree_dot(left: Any, right: Any) -> float:
    """Return the array-leaf dot product between two like-structured PyTrees."""

    total = 0.0
    for left_leaf, right_leaf in zip(jt.leaves(left), jt.leaves(right), strict=True):
        if eqx.is_array(left_leaf) and eqx.is_array(right_leaf):
            total += float(
                np.vdot(
                    np.asarray(left_leaf, dtype=np.float64),
                    np.asarray(right_leaf, dtype=np.float64),
                ).real
            )
    return total


def write_bulk_arrays(
    path: Path,
    *,
    base_trial_specs: TaskTrialSpec,
    active_trial_specs: TaskTrialSpec,
    zero_trial_specs: TaskTrialSpec,
    active_states: Any,
    zero_states: Any,
) -> Path:
    """Write paired rollout arrays to ignored artifact storage."""

    mkdir_p(path.parent)
    np.savez_compressed(
        path,
        paired_without_broad_epsilon=np.asarray(
            base_trial_specs.inputs["epsilon"],
            dtype=np.float64,
        ),
        broad_epsilon_delta=(
            np.asarray(active_trial_specs.inputs["epsilon"], dtype=np.float64)
            - np.asarray(base_trial_specs.inputs["epsilon"], dtype=np.float64)
        ),
        active_epsilon=np.asarray(active_trial_specs.inputs["epsilon"], dtype=np.float64),
        zero_broad_epsilon=np.asarray(zero_trial_specs.inputs["epsilon"], dtype=np.float64),
        active_mechanics_vector=np.asarray(active_states.mechanics.vector, dtype=np.float64),
        zero_mechanics_vector=np.asarray(zero_states.mechanics.vector, dtype=np.float64),
        active_command=np.asarray(active_states.net.output, dtype=np.float64),
        zero_command=np.asarray(zero_states.net.output, dtype=np.float64),
        active_hidden=np.asarray(active_states.net.hidden, dtype=np.float64),
        zero_hidden=np.asarray(zero_states.net.hidden, dtype=np.float64),
    )
    return path


def epsilon_summary(epsilon: np.ndarray) -> dict[str, Any]:
    """Return JSON-compatible sampled epsilon diagnostics."""

    norms = np.sqrt(np.sum(np.square(epsilon), axis=(-2, -1)))
    return {
        "shape": list(epsilon.shape),
        "per_trial_l2": {
            "min": float(np.min(norms)),
            "mean": float(np.mean(norms)),
            "max": float(np.max(norms)),
        },
        "all_zero": bool(np.allclose(epsilon, 0.0)),
    }


def broad_epsilon_metadata(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the broad epsilon hps payload from a run spec."""

    broad = dict(run_spec.get("hps", {}).get("broad_epsilon_training", {}) or {})
    if broad.get("mode") == BROAD_EPSILON_TRAINING_MODE:
        broad["enabled"] = bool(broad.get("enabled", True))
    return broad


def summarize_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact aggregate summary over run rows."""

    evaluated = [row for row in rows if _attribution_row_evaluated(row)]
    deltas = [
        float(row["loss"]["delta_active_minus_zero"]["total"]) for row in evaluated
    ]
    status_counts: dict[str, int] = {}
    for row in rows:
        status = "evaluated" if _attribution_row_evaluated(row) else str(
            row.get("status", "unknown")
        )
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "n_rows": len(rows),
        "n_evaluated": len(evaluated),
        "status_counts": status_counts,
        "total_loss_delta_active_minus_zero": (
            {
                "min": float(np.min(deltas)),
                "mean": float(np.mean(deltas)),
                "max": float(np.max(deltas)),
            }
            if deltas
            else {}
        ),
    }


def render_summary_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render a CSV summary table."""

    fieldnames = [
        "run_id",
        "level",
        "n_rollout_trials",
        "active_epsilon_l2_mean",
        "paired_without_broad_epsilon_l2_mean",
        "broad_delta_l2_mean",
        "active_total_loss",
        "without_broad_total_loss",
        "delta_total_loss",
        "gradient_status",
        "gradient_relative_delta_norm_vs_active_mean",
    ]
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        if not _attribution_row_evaluated(row):
            writer.writerow(
                {
                    "run_id": row["run_id"],
                    "level": row.get("broad_epsilon_training", {}).get("level"),
                    "gradient_status": row.get("status"),
                }
            )
            continue
        grad_agg = row.get("gradient", {}).get("aggregate", {})
        writer.writerow(
            {
                "run_id": row["run_id"],
                "level": row.get("broad_epsilon_training", {}).get("level"),
                "n_rollout_trials": row["n_rollout_trials"],
                "active_epsilon_l2_mean": row["epsilon"]["active_total"]["per_trial_l2"][
                    "mean"
                ],
                "paired_without_broad_epsilon_l2_mean": row["epsilon"][
                    "paired_without_broad"
                ]["per_trial_l2"]["mean"],
                "broad_delta_l2_mean": row["epsilon"]["broad_delta"]["per_trial_l2"][
                    "mean"
                ],
                "active_total_loss": row["loss"]["active"]["total"],
                "without_broad_total_loss": row["loss"]["zero"]["total"],
                "delta_total_loss": row["loss"]["delta_active_minus_zero"]["total"],
                "gradient_status": row.get("gradient", {}).get("status"),
                "gradient_relative_delta_norm_vs_active_mean": grad_agg.get(
                    "relative_delta_norm_vs_active", {}
                ).get("mean"),
            }
        )
    return stream.getvalue()


def render_markdown(
    manifest: Mapping[str, Any],
    *,
    csv_path: Path,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Render a concise Markdown note for the attribution manifest."""

    lines = [
        "# Paired broad-epsilon attribution diagnostic",
        "",
        "Active uses the run's actual training sampler, including target sampling, "
        "calibrated graph-channel perturbations, and broad/full-state epsilon. The "
        "paired condition replays the same sampler branches and rollout PRNG seed but "
        "removes only the broad-epsilon draw. The manifest separates "
        "`paired_without_broad`, `broad_delta`, and `active_total` epsilon arrays.",
        "",
        f"- Schema: `{manifest['schema_version']}`",
        f"- Rows: {manifest['summary']['n_rows']}",
        f"- CSV summary: `{repo_relative(csv_path, repo_root=repo_root)}`",
        f"- Bulk arrays: `{manifest['bulk_dir']}`",
        "",
        "| run | level | n | broad L2 mean | active loss | without-broad loss | delta | grad |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in manifest["rows"]:
        if not _attribution_row_evaluated(row):
            lines.append(
                "| {run} | {level} |  |  |  |  |  | {status}: {reason} |".format(
                    run=row["run_id"],
                    level=row.get("broad_epsilon_training", {}).get("level"),
                    status=row.get("status", "not_evaluated"),
                    reason=row.get("reason", ""),
                )
            )
            continue
        lines.append(
            "| {run} | {level} | {n} | {eps:.6g} | {active:.6g} | {zero:.6g} | "
            "{delta:.6g} | {grad} |".format(
                run=row["run_id"],
                level=row.get("broad_epsilon_training", {}).get("level"),
                n=row["n_rollout_trials"],
                eps=row["epsilon"]["broad_delta"]["per_trial_l2"]["mean"],
                active=row["loss"]["active"]["total"],
                zero=row["loss"]["zero"]["total"],
                delta=row["loss"]["delta_active_minus_zero"]["total"],
                grad=row.get("gradient", {}).get("status"),
            )
        )
    lines.extend(
        [
            "",
            "Gradient attribution is raw pre-optimizer trainable-parameter gradient "
            "direction on the bounded replicate subset. Optimizer update direction is "
            "not materialized because validation-selected models are assembled from "
            "per-replicate checkpoints without a synchronized optimizer state.",
            "",
        ]
    )
    return "\n".join(lines)


def _attribution_row_evaluated(row: Mapping[str, Any]) -> bool:
    """Return whether a row contains evaluated attribution payloads."""

    return (
        row.get("status", "evaluated") == "evaluated"
        and "loss" in row
        and "epsilon" in row
    )


def active_vs_zero_semantics() -> dict[str, Any]:
    """Return schema text for the paired condition semantics."""

    return {
        "active": (
            "sampled training batch from the run's composed task wrappers, including "
            "target sampling, calibrated graph-channel perturbations when enabled, "
            "and the run-spec broad/full-state epsilon sampler"
        ),
        "zero": (
            "same model, sampler key, target branch, calibrated-perturbation branch, "
            "and rollout seed, but with only the broad-epsilon branch removed; existing "
            "non-broad epsilon or graph-channel perturbation inputs are preserved"
        ),
        "delta_sign": "active_minus_zero",
    }


def gradient_policy(max_gradient_replicates: int) -> dict[str, Any]:
    """Return manifest metadata for bounded gradient work."""

    return {
        "max_gradient_replicates": int(max_gradient_replicates),
        "bounded_subset_reason": (
            "GRU rollout gradients are expensive across all rows and all replicates; "
            "the CLI exposes max_gradient_replicates for an evaluable subset"
        ),
        "gradient_kind": "raw_pre_optimizer_trainable_parameter_gradient",
    }


def infer_batch_size(trial_specs: TaskTrialSpec) -> int:
    """Infer the leading batch dimension from trial specs."""

    epsilon = trial_specs.inputs.get("epsilon")
    epsilon_shape = getattr(epsilon, "shape", None)
    if epsilon_shape is not None and len(epsilon_shape) >= 3:
        return int(epsilon_shape[0])
    for target in trial_specs.targets.values():
        value = getattr(target, "value", None)
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 3:
            return int(shape[0])
    target = trial_specs.inputs.get("effector_target")
    if target is not None and hasattr(target, "pos"):
        shape = getattr(target.pos, "shape", None)
        if shape is not None and len(shape) >= 3:
            return int(shape[0])
    for value in trial_specs.inits.values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[0])
    if epsilon_shape is not None and len(epsilon_shape) >= 1:
        return int(epsilon_shape[0])
    if target is not None and hasattr(target, "pos"):
        shape = getattr(target.pos, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    for value in trial_specs.inits.values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    raise ValueError("could not infer trial batch size")


def _has_batch_prefix(value: Any, batch: int) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) >= 1 and int(shape[0]) == int(batch)


def repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    """Return a repo-relative path string when possible."""

    path = Path(path)
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)
