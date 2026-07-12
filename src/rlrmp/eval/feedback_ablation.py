"""Evaluation-layer execution for feedback-ablation rollout states."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt

from rlrmp.eval.rollout_states import CachedEvaluationStates


@dataclass(frozen=True)
class DetailedRolloutEvaluation:
    """Rollout product plus arrays used by feedback-ablation reducers."""

    rollout: CachedEvaluationStates
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
    rollout = CachedEvaluationStates.from_states(
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


def evaluate_feedback_ablation_runs(
    params: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Execute all model/checkpoint feedback-ablation runs declared by a spec."""

    # These analysis-layer imports expose the established scoring reducers while
    # rollout ownership remains here, at the registered evaluation boundary.
    # They are local to avoid the module-registration cycle during package import.
    from rlrmp.analysis.pipelines import gru_feedback_ablation as analysis
    from rlrmp.eval.perturbation_bank import (
        default_cs_perturbation_bank,
    )
    from rlrmp.eval.trial_inputs import resolve_evaluation_run_inputs

    source_experiment = str(params["source_experiment"])
    run_ids = tuple(str(run_id) for run_id in params["run_ids"])
    labels_value = params.get("labels")
    labels = (
        None
        if labels_value is None
        else tuple(str(label) for label in labels_value)
    )
    bank_value = params.get("bank")
    bank = (
        dict(bank_value)
        if isinstance(bank_value, Mapping)
        else default_cs_perturbation_bank(
            mode=str(params["bank_mode"]),
            calibration_level=params.get("calibration_level"),
            calibration_reach=params.get("calibration_reach"),
            feedback_scale_manifest_path=_optional_repo_path(
                params.get("feedback_scale_manifest_path"),
                repo_root=repo_root,
            ),
        )
    )
    evaluation_bins_value = params.get("evaluation_bins")
    evaluation_bins = (
        {str(key): value for key, value in evaluation_bins_value.items()}
        if isinstance(evaluation_bins_value, Mapping)
        else analysis.selected_feedback_ablation_bins_for_bank(
            bank,
            preferred_level=str(params["feedback_selection_level"]),
        )
    )
    run_inputs = resolve_evaluation_run_inputs(
        experiment=source_experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )
    runs = {
        run.run_id: analysis._execute_feedback_ablation_run(
            run,
            source_experiment=source_experiment,
            n_rollout_trials=int(params["n_rollout_trials"]),
            include_checkpoint_rescore=bool(params["include_checkpoint_rescore"]),
            bank=bank,
            evaluation_bins=evaluation_bins,
            preferred_checkpoint_manifest_path=_optional_repo_path(
                params.get("preferred_checkpoint_manifest_path"),
                repo_root=repo_root,
            ),
            repo_root=repo_root,
        )
        for run in run_inputs
    }
    return {
        "source_experiment": source_experiment,
        "run_ids": list(run_ids),
        "labels": None if labels is None else list(labels),
        "scope": str(params["scope"]),
        "bank_mode": str(params["bank_mode"]),
        "bank": {
            "bank_id": bank.get("bank_id"),
            "calibration_metadata_hooks": bank.get("calibration_metadata_hooks"),
            "n_perturbations": len(bank.get("perturbations", ())),
        },
        "evaluation_bins": evaluation_bins,
        "ablation_modes": list(analysis.default_ablation_modes()),
        "runs": runs,
    }


def _optional_repo_path(value: Any, *, repo_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else repo_root / path


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


__all__ = [
    "DetailedRolloutEvaluation",
    "evaluate_feedback_ablation_runs",
    "evaluate_model_on_trial_specs",
]
