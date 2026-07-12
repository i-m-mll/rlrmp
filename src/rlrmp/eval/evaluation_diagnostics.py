"""Registered-evaluation execution for cached GRU diagnostic states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.random as jr
from feedbax.config.namespace import TreeNamespace, dict_to_namespace

from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.eval.checkpoint_selection import (
    load_materialized_fixed_bank_manifest,
    load_validation_selected_checkpoint_model,
)
from rlrmp.eval.gru_diagnostics import (
    DEFAULT_JACOBIAN_TIMEPOINTS,
    RolloutEvaluation,
    diagnostic_definitions,
    summarize_controller_feedback_scales,
    summarize_gru_gates,
    summarize_gru_jacobians,
    summarize_rollout_behavior,
)
from rlrmp.eval.rollout_states import CachedEvaluationStates
from rlrmp.eval.trial_inputs import (
    EvaluationRunInputs,
    repeat_single_validation_trial,
    resolve_evaluation_run_inputs,
)
from rlrmp.paths import REPO_ROOT
from rlrmp.runtime.run_spec_access import require_run_dt, require_run_seed
from rlrmp.train.task_model import setup_task_model_pair

DEFAULT_N_ROLLOUT_TRIALS = int(
    load_analysis_parameter_preset("gru_pilot_figures").parameters["n_rollout_trials"]
)


def evaluate_gru_diagnostics_runs(
    params: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Execute selected-checkpoint rollouts and cache states plus summaries."""

    experiment = str(params["source_experiment"])
    run_ids = tuple(str(run_id) for run_id in params["run_ids"])
    labels_value = params.get("labels")
    labels = None if labels_value is None else tuple(str(label) for label in labels_value)
    n_rollout_trials = int(params.get("n_rollout_trials", DEFAULT_N_ROLLOUT_TRIALS))
    jacobian_timepoints = tuple(
        str(value) for value in params.get("jacobian_timepoints", DEFAULT_JACOBIAN_TIMEPOINTS)
    )
    manifest_value = params.get("preferred_checkpoint_manifest_path")
    preferred_manifest_path = (
        None
        if manifest_value is None
        else _resolve_repo_path(manifest_value, repo_root=repo_root)
    )
    runs = resolve_evaluation_run_inputs(
        experiment=experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )
    checkpoint_policy = _checkpoint_policy(preferred_manifest_path)
    run_payloads = {
        run.run_id: _evaluate_run(
            run,
            experiment=experiment,
            n_rollout_trials=n_rollout_trials,
            jacobian_timepoints=jacobian_timepoints,
            checkpoint_policy=checkpoint_policy,
            preferred_manifest_path=preferred_manifest_path,
            repo_root=repo_root,
        )
        for run in runs
    }
    return {
        "source_experiment": experiment,
        "checkpoint_policy": checkpoint_policy,
        "scope": "post_hoc_evaluation_non_certificate_diagnostics",
        "standard_certificate_metrics": {
            "status": "excluded",
            "note": (
                "This cached evaluation records rollout behavior and recurrent-controller "
                "diagnostics only; certificate metrics remain separate."
            ),
        },
        "runs": run_payloads,
    }


def _evaluate_run(
    run: EvaluationRunInputs,
    *,
    experiment: str,
    n_rollout_trials: int,
    jacobian_timepoints: Sequence[str],
    checkpoint_policy: str,
    preferred_manifest_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    if n_rollout_trials < 1:
        raise ValueError("n_rollout_trials must be at least 1")
    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = require_run_seed(run.run_spec, source=run.run_spec_path)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=experiment,
        run_id=run.run_id,
        run_spec=run.run_spec,
        preferred_manifest_path=preferred_manifest_path,
        checkpoint_selection_mode=(
            "fixed_bank_manifest" if preferred_manifest_path is not None else "sparse_history"
        ),
        repo_root=repo_root,
    )
    trial_specs = repeat_single_validation_trial(pair.task.validation_trials, n_rollout_trials)
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_array(leaf, n_replicates),
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return pair.task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, n_rollout_trials),
        )

    states = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(0), n_replicates),
    )
    cached = CachedEvaluationStates.from_states(
        states,
        trial_specs,
        dt=require_run_dt(run.run_spec, hps, source=run.run_spec_path),
        checkpoint_selection=tuple(checkpoint_selection),
    )
    rollout = cached.to_rollout_evaluation(RolloutEvaluation)
    return {
        "label": run.label,
        "checkpoint_selection": [
            selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
        ],
        "n_replicates": int(cached.command.shape[0]),
        "n_rollout_trials_per_replicate": int(cached.command.shape[1]),
        "n_time_steps": int(cached.command.shape[2]),
        "dt_s": cached.dt,
        "definitions": diagnostic_definitions(),
        "behavior": summarize_rollout_behavior(rollout),
        "controller_feedback_scales": summarize_controller_feedback_scales(
            rollout,
            run_id=run.run_id,
            checkpoint_policy=checkpoint_policy,
        ),
        "gru_gates": summarize_gru_gates(model.nodes["net"].hidden, rollout),
        "local_recurrent_jacobians": summarize_gru_jacobians(
            model.nodes["net"].hidden,
            rollout,
            timepoint_policy=jacobian_timepoints,
        ),
        "cached_states": cached,
    }


def _checkpoint_policy(manifest_path: Path | None) -> str:
    manifest = load_materialized_fixed_bank_manifest(manifest_path=manifest_path)
    if manifest is None:
        return "validation_selected_per_replicate"
    return str(
        manifest.metadata.get("checkpoint_policy") or "fixed_bank_rescored_per_replicate"
    )


def _resolve_repo_path(value: Any, *, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else repo_root / path


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates


__all__ = ["DEFAULT_N_ROLLOUT_TRIALS", "evaluate_gru_diagnostics_runs"]
