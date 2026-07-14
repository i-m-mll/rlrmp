"""Broad-epsilon paired rollout and differentiable-gradient evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax import TaskTrialSpec, WhereDict
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.runtime.batch import BatchInfo
from jax_cookbook.tree import filter_spec_leaves

from rlrmp.analysis.gru_standard_certificate import normalize_gru_hps
from rlrmp.eval.checkpoint_selection import load_validation_selected_checkpoint_model
from rlrmp.eval.trial_inputs import resolve_evaluation_run_inputs
from rlrmp.paths import REPO_ROOT
from rlrmp.runtime.run_spec_access import require_run_seed
from rlrmp.train.cs_nominal_gru import _where_train
from rlrmp.train.cs_perturbation_training import (
    BroadFullStateEpsilonTrainingTaskAdapter,
    FixedTargetPerturbationTrainingTaskAdapter,
    apply_training_perturbation_mixture,
)
from rlrmp.train.science_vocabulary import ScienceMode
from rlrmp.train.task_model import setup_task_model_pair


def evaluate_broad_epsilon_runs(
    params: Mapping[str, Any],
    *,
    repo_root=REPO_ROOT,
) -> dict[str, Any]:
    """Execute registered paired rollouts and gradients for declared run IDs."""

    experiment = str(params["source_experiment"])
    run_ids = tuple(str(run_id) for run_id in params["run_ids"])
    labels_value = params.get("labels")
    labels = None if labels_value is None else tuple(str(label) for label in labels_value)
    n_trials = int(params["n_rollout_trials"])
    max_gradient_replicates = int(params["max_gradient_replicates"])
    runs = resolve_evaluation_run_inputs(
        experiment=experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )
    return {
        "source_experiment": experiment,
        "checkpoint_policy": "validation_selected_per_replicate",
        "rows": [
            _evaluate_run(
                run,
                experiment=experiment,
                n_trials=n_trials,
                max_gradient_replicates=max_gradient_replicates,
                repo_root=repo_root,
            )
            for run in runs
        ],
    }


def _evaluate_run(
    run: Any,
    *,
    experiment: str,
    n_trials: int,
    max_gradient_replicates: int,
    repo_root: Any,
) -> dict[str, Any]:
    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = require_run_seed(run.run_spec, source=run.run_spec_path)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=experiment,
        run_id=run.run_id,
        run_spec=run.run_spec,
        repo_root=repo_root,
    )
    try:
        active_specs, zero_specs = paired_broad_epsilon_training_specs(
            pair.task,
            key=jr.PRNGKey(seed + 31),
            n_trials=n_trials,
        )
    except ValueError as exc:
        return {
            "run_id": run.run_id,
            "label": run.label,
            "status": "not_applicable",
            "reason": str(exc),
        }
    active_states = evaluate_model_states(
        model=model,
        task=pair.task,
        trial_specs=active_specs,
        n_replicates=n_replicates,
        seed=seed + 103,
    )
    zero_states = evaluate_model_states(
        model=model,
        task=pair.task,
        trial_specs=zero_specs,
        n_replicates=n_replicates,
        seed=seed + 103,
    )
    active_loss = summarize_loss_tree(pair.task.loss_func(active_states, active_specs, model))
    zero_loss = summarize_loss_tree(pair.task.loss_func(zero_states, zero_specs, model))
    gradient = gradient_attribution_summary(
        model=model,
        task=pair.task,
        loss_func=pair.task.loss_func,
        active_trial_specs=active_specs,
        zero_trial_specs=zero_specs,
        n_replicates=n_replicates,
        max_gradient_replicates=max_gradient_replicates,
        seed=seed + 211,
    )
    active_epsilon = np.asarray(active_specs.inputs["epsilon"], dtype=np.float64)
    zero_epsilon = np.asarray(zero_specs.inputs["epsilon"], dtype=np.float64)
    return {
        "run_id": run.run_id,
        "label": run.label,
        "status": "evaluated",
        "n_rollout_trials": infer_batch_size(active_specs),
        "checkpoint_selection": [
            selected.to_json(repo_root=repo_root) for selected in checkpoint_selection
        ],
        "epsilon": {
            "active_total": epsilon_summary(active_epsilon),
            "paired_without_broad": epsilon_summary(zero_epsilon),
            "broad_delta": epsilon_summary(active_epsilon - zero_epsilon),
        },
        "loss": {
            "active": active_loss,
            "zero": zero_loss,
            "delta_active_minus_zero": loss_delta_summary(active_loss, zero_loss),
        },
        "gradient": gradient,
        "active_states": active_states,
        "zero_states": zero_states,
    }


def paired_broad_epsilon_training_specs(
    task: Any,
    *,
    key: Any,
    n_trials: int,
    batch_info: Any = None,
) -> tuple[TaskTrialSpec, TaskTrialSpec]:
    """Sample one training batch with and without only the broad-epsilon branch."""

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    batch_info = batch_info or BatchInfo(size=int(n_trials), current=0, total=1)
    keys = jr.split(key, int(n_trials))
    active = eqx.filter_vmap(
        lambda trial_key: task.get_train_trial_with_intervenor_params(trial_key, batch_info)
    )(keys)
    paired_without_broad = eqx.filter_vmap(
        lambda trial_key: _training_specs_without_broad_epsilon(task, trial_key, batch_info)
    )(keys)
    return materialize_trial_batch(active), materialize_trial_batch(paired_without_broad)


def _training_specs_without_broad_epsilon(
    task: Any,
    key: Any,
    batch_info: Any = None,
) -> TaskTrialSpec:
    if isinstance(task, FixedTargetPerturbationTrainingTaskAdapter):
        key_trial, key_pert = jr.split(key)
        base = _training_specs_without_broad_epsilon(task.task, key_trial, batch_info)
        return apply_training_perturbation_mixture(base, task.config, key_pert, batch_info)
    if isinstance(task, BroadFullStateEpsilonTrainingTaskAdapter):
        key_base, _ = jr.split(key)
        return task.task.get_train_trial_with_intervenor_params(key_base, batch_info)
    inner_task = getattr(task, "task", None)
    if inner_task is not None:
        return _training_specs_without_broad_epsilon(inner_task, key, batch_info)
    raise ValueError(
        "paired broad-epsilon evaluation requires a "
        "BroadFullStateEpsilonTrainingTaskAdapter in the training task stack"
    )


def materialize_trial_batch(trial_specs: TaskTrialSpec) -> TaskTrialSpec:
    """Broadcast shared sampler leaves onto the explicit trial axis."""

    batch = infer_batch_size(trial_specs)

    def broadcast_leaf(leaf: Any) -> Any:
        shape = getattr(leaf, "shape", None)
        if shape is None or (len(shape) >= 1 and int(shape[0]) == batch):
            return leaf
        return jnp.broadcast_to(leaf, (batch, *shape))

    def broadcast_tree(value: Any) -> Any:
        return jt.map(broadcast_leaf, value)

    return TaskTrialSpec(
        inits=WhereDict({key: broadcast_tree(value) for key, value in trial_specs.inits.items()}),
        inputs={key: broadcast_tree(value) for key, value in trial_specs.inputs.items()},
        targets={key: broadcast_tree(value) for key, value in trial_specs.targets.items()},
        intervene=trial_specs.intervene,
        extra=trial_specs.extra,
        timeline=broadcast_tree(trial_specs.timeline),
    )


def zero_epsilon_trial_specs(trial_specs: TaskTrialSpec) -> TaskTrialSpec:
    """Return the same trial bank with its epsilon input zeroed."""

    if "epsilon" not in trial_specs.inputs:
        raise ValueError("paired broad-epsilon evaluation requires an epsilon input")
    return replace_trial_input(
        trial_specs,
        "epsilon",
        jnp.zeros_like(jnp.asarray(trial_specs.inputs["epsilon"])),
    )


def truncate_trial_specs(trial_specs: TaskTrialSpec, n_trials: int) -> TaskTrialSpec:
    """Keep a bounded prefix of a sampled trial bank."""

    batch = infer_batch_size(trial_specs)
    n = min(int(n_trials), batch)
    if n < 1:
        raise ValueError("n_trials must be at least 1")
    if n == batch:
        return trial_specs

    def take_prefix(value: Any) -> Any:
        if _has_batch_prefix(value, batch):
            return value[:n]
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return value
        return jt.map(lambda leaf: leaf[:n] if _has_batch_prefix(leaf, batch) else leaf, value)

    return TaskTrialSpec(
        inits=WhereDict({key: take_prefix(value) for key, value in trial_specs.inits.items()}),
        inputs={key: take_prefix(value) for key, value in trial_specs.inputs.items()},
        targets={key: take_prefix(value) for key, value in trial_specs.targets.items()},
        intervene=trial_specs.intervene,
        extra=trial_specs.extra,
        timeline=take_prefix(trial_specs.timeline),
    )


def replace_trial_input(trial_specs: TaskTrialSpec, name: str, value: Any) -> TaskTrialSpec:
    """Return trial specs with one named input replaced."""

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
    """Evaluate all replicate models on one shared trial bank."""

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


def gradient_for_trial_specs(
    *,
    model: Any,
    task: Any,
    loss_func: Any,
    trial_specs: TaskTrialSpec,
    seed: int,
) -> Any:
    """Return the trainable-parameter gradient for one cached trial bank."""

    spec = filter_spec_leaves(model, _where_train()[0])
    trainable, static = eqx.partition(model, spec)
    n_trials = infer_batch_size(trial_specs)

    def objective(trainable_model: Any) -> Any:
        candidate = eqx.combine(trainable_model, static)
        states = task.eval_trials(candidate, trial_specs, jr.split(jr.PRNGKey(seed), n_trials))
        return loss_func(states, trial_specs, candidate).total

    return eqx.filter_grad(objective)(trainable)


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
    """Evaluate a bounded replicate subset of active/zero gradient pairs."""

    if max_gradient_replicates == 0:
        return {"status": "not_evaluated", "reason": "max_gradient_replicates=0"}
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates,
    )
    rows = []
    for replicate in range(min(max_gradient_replicates, n_replicates)):
        replicate_model = eqx.combine(
            jt.map(lambda leaf: leaf[replicate], model_arrays), model_other
        )
        active = gradient_for_trial_specs(
            model=replicate_model,
            task=task,
            loss_func=loss_func,
            trial_specs=active_trial_specs,
            seed=seed + replicate,
        )
        zero = gradient_for_trial_specs(
            model=replicate_model,
            task=task,
            loss_func=loss_func,
            trial_specs=zero_trial_specs,
            seed=seed + replicate,
        )
        rows.append({"replicate": replicate, **gradient_pair_metrics(active, zero)})
    metric_names = tuple(key for key in rows[0] if key != "replicate") if rows else ()
    return {
        "status": "evaluated",
        "gradient_kind": "raw_pre_optimizer_trainable_parameter_gradient",
        "replicates": rows,
        "aggregate": {
            name: {
                "mean": float(np.mean([row[name] for row in rows])),
                "max": float(np.max([row[name] for row in rows])),
            }
            for name in metric_names
        },
    }


def gradient_pair_metrics(active_grad: Any, zero_grad: Any) -> dict[str, float]:
    """Return fixed-fixture norm and direction parity metrics."""

    delta = jt.map(lambda active, zero: active - zero, active_grad, zero_grad)
    active_norm = pytree_l2_norm(active_grad)
    zero_norm = pytree_l2_norm(zero_grad)
    delta_norm = pytree_l2_norm(delta)
    denom = max(active_norm * zero_norm, 1e-30)
    return {
        "active_gradient_norm": active_norm,
        "zero_gradient_norm": zero_norm,
        "active_minus_zero_gradient_norm": delta_norm,
        "active_zero_gradient_cosine": float(pytree_dot(active_grad, zero_grad) / denom),
        "relative_delta_norm_vs_active": float(delta_norm / max(active_norm, 1e-30)),
        "relative_delta_norm_vs_zero": float(delta_norm / max(zero_norm, 1e-30)),
    }


def summarize_loss_tree(loss_tree: Any) -> dict[str, Any]:
    """Return scalar total and top-level weighted loss terms."""

    return {
        "total": float(np.asarray(loss_tree.total)),
        "terms": {name: float(np.asarray(child.total)) for name, child in loss_tree.items()},
    }


def loss_delta_summary(active: Mapping[str, Any], zero: Mapping[str, Any]) -> dict[str, Any]:
    """Return active-minus-zero loss deltas."""

    names = sorted(set(active.get("terms", {})) | set(zero.get("terms", {})))
    return {
        "total": float(active["total"] - zero["total"]),
        "terms": {
            name: float(
                active.get("terms", {}).get(name, 0.0) - zero.get("terms", {}).get(name, 0.0)
            )
            for name in names
        },
    }


def epsilon_summary(epsilon: np.ndarray) -> dict[str, Any]:
    """Return fixed-fixture epsilon endpoint metrics."""

    values = np.asarray(epsilon, dtype=np.float64)
    norms = np.sqrt(np.sum(np.square(values), axis=(-2, -1)))
    return {
        "shape": list(values.shape),
        "per_trial_l2": {
            "min": float(np.min(norms)),
            "mean": float(np.mean(norms)),
            "max": float(np.max(norms)),
        },
        "all_zero": bool(np.allclose(values, 0.0)),
    }


def broad_epsilon_metadata(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return broad-epsilon settings without embedding experiment identities."""

    broad = dict(run_spec.get("hps", {}).get("broad_epsilon_training", {}) or {})
    if broad.get("mode") == ScienceMode.BROAD_EPSILON and "enabled" not in broad:
        broad["enabled"] = True
    return broad


def active_vs_zero_semantics() -> dict[str, Any]:
    """Describe the paired evaluation contract."""

    return {
        "active": "sampled composed training batch including broad/full-state epsilon",
        "zero": "same sampler key with only the broad-epsilon branch removed",
        "delta_sign": "active_minus_zero",
    }


def pytree_l2_norm(tree: Any) -> float:
    total = sum(
        float(np.vdot(np.asarray(leaf, dtype=np.float64), np.asarray(leaf, dtype=np.float64)).real)
        for leaf in jt.leaves(tree)
        if eqx.is_array(leaf)
    )
    return float(np.sqrt(total))


def pytree_dot(left: Any, right: Any) -> float:
    return sum(
        float(np.vdot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)).real)
        for a, b in zip(jt.leaves(left), jt.leaves(right), strict=True)
        if eqx.is_array(a) and eqx.is_array(b)
    )


def infer_batch_size(trial_specs: TaskTrialSpec) -> int:
    """Infer the explicit leading trial dimension."""

    epsilon = trial_specs.inputs.get("epsilon")
    epsilon_shape = getattr(epsilon, "shape", None)
    if epsilon_shape is not None and len(epsilon_shape) >= 3:
        return int(epsilon_shape[0])
    for value in (
        *trial_specs.targets.values(),
        *trial_specs.inputs.values(),
        *trial_specs.inits.values(),
    ):
        candidate = getattr(value, "value", getattr(value, "pos", value))
        shape = getattr(candidate, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    raise ValueError("could not infer trial batch size")


def _has_batch_prefix(value: Any, batch: int) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) >= 1 and int(shape[0]) == int(batch)


__all__ = [
    "active_vs_zero_semantics",
    "broad_epsilon_metadata",
    "epsilon_summary",
    "evaluate_model_states",
    "evaluate_broad_epsilon_runs",
    "gradient_for_trial_specs",
    "gradient_pair_metrics",
    "loss_delta_summary",
    "paired_broad_epsilon_training_specs",
    "summarize_loss_tree",
    "truncate_trial_specs",
    "zero_epsilon_trial_specs",
]
