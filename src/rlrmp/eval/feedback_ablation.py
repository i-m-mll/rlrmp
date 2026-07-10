"""Evaluation-layer execution for feedback-ablation rollout states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt

from rlrmp.analysis.pipelines._selected_eval_rollouts import SelectedEvalRolloutProduct


@dataclass(frozen=True)
class DetailedRolloutEvaluation:
    """Rollout product plus arrays used by feedback-ablation reducers."""

    rollout: SelectedEvalRolloutProduct
    feedback: Any
    mechanics_vector: Any


def evaluate_model_on_trial_specs(
    *, model: Any, task: Any, trial_specs: Any, n_replicates: int, seed: int
) -> DetailedRolloutEvaluation:
    """Execute one feedback-ablation evaluation at the eval-layer boundary."""

    model_arrays, model_other = eqx.partition(
        model, lambda leaf: _is_replicate_array(leaf, n_replicates)
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, _infer_batch_size(trial_specs)),
        )

    if _has_replicate_specific_trial_inputs(trial_specs, n_replicates):
        keys = jr.split(jr.PRNGKey(seed), n_replicates)
        states_by_replicate = []
        for replicate in range(n_replicates):
            replicate_model = eqx.combine(
                _select_replicate_tree(model_arrays, replicate, n_replicates), model_other
            )
            replicate_trial_specs = _select_replicate_trial_inputs(
                trial_specs, replicate, n_replicates
            )
            states_by_replicate.append(
                task.eval_trials(
                    replicate_model,
                    replicate_trial_specs,
                    jr.split(keys[replicate], _infer_batch_size(replicate_trial_specs)),
                )
            )
        states = jt.map(lambda *xs: jnp.stack(xs, axis=0), *states_by_replicate)
    else:
        states = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
            model_arrays, jr.split(jr.PRNGKey(seed), n_replicates)
        )
    rollout = SelectedEvalRolloutProduct.from_states(
        states,
        trial_specs,
        dt=0.01,
        include_mechanics_vector=True,
        include_feedback=True,
    )
    return DetailedRolloutEvaluation(
        rollout=rollout,
        feedback=rollout.feedback,
        mechanics_vector=rollout.mechanics_vector,
    )


def _has_replicate_specific_trial_inputs(trial_specs: Any, n_replicates: int) -> bool:
    return any(
        key.startswith("feedback_ablation:")
        and getattr(value, "shape", ())[:1] == (n_replicates,)
        for key, value in trial_specs.inputs.items()
    )


def _select_replicate_trial_inputs(trial_specs: Any, replicate: int, n_replicates: int) -> Any:
    inputs = {}
    for key, value in trial_specs.inputs.items():
        if key.startswith("feedback_ablation:") and getattr(value, "shape", ())[:1] == (
            n_replicates,
        ):
            inputs[key] = value[replicate]
        else:
            inputs[key] = value
    return eqx.tree_at(lambda ts: ts.inputs, trial_specs, inputs)


def _select_replicate_tree(tree: Any, replicate: int, n_replicates: int) -> Any:
    return jt.map(
        lambda leaf: leaf[replicate] if _is_replicate_array(leaf, n_replicates) else leaf,
        tree,
    )


def _infer_batch_size(trial_specs: Any) -> int:
    for values in (trial_specs.inputs.values(), trial_specs.inits.values()):
        for value in values:
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) >= 1:
                return int(shape[0])
            pos = getattr(value, "pos", None)
            if pos is not None:
                return int(pos.shape[0])
    raise ValueError("could not infer trial batch size")


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates


__all__ = ["DetailedRolloutEvaluation", "evaluate_model_on_trial_specs"]
