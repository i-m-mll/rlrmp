"""Worst-case full-state epsilon audit for frozen C&S GRU rollouts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax.types import TreeNamespace, dict_to_namespace
from jaxtyping import Array, Float

from rlrmp.analysis.math.cs_game_card import TARGET_POS, build_canonical_game
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    CheckpointSelectionMode,
    load_validation_selected_checkpoint_model,
)
from rlrmp.analysis.pipelines.gru_perturbation_bank import summarize_perturbation_response
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    RunFigureInputs,
    repeat_single_validation_trial,
    resolve_run_inputs,
)
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_LEVELS,
    BROAD_EPSILON_REFERENCE_REACH_M,
)


SCHEMA_VERSION = "rlrmp.gru_worst_case_epsilon_audit.v1"
DEFAULT_SOURCE_EXPERIMENT = "b8aa38e"
DEFAULT_RESULT_EXPERIMENT = "b8aa38e"
DEFAULT_RUN_IDS = ("smoke__broad_strong_cal_small",)
DEFAULT_OUTPUT_FILENAME = "gru_worst_case_epsilon_audit_manifest.json"
DEFAULT_NOTE_FILENAME = "gru_worst_case_epsilon_audit.md"
DEFAULT_BULK_SUBDIR = "worst_case_epsilon_audit"

ObjectiveFn = Callable[[Float[Array, "T m_w"]], Float[Array, ""]]


@dataclass(frozen=True)
class EpsilonOptimizationResult:
    """Result of projected ascent over one open-loop epsilon sequence."""

    epsilon: np.ndarray
    objective: float
    initial_objective: float
    final_objective: float
    energy: float
    l2_norm: float
    restart_index: int
    history: tuple[dict[str, float | int], ...]
    restart_summaries: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class JaxRolloutEvaluation:
    """JAX rollout arrays needed by the differentiable audit objective."""

    mechanics_vector: Any
    command: Any
    position: Any
    velocity: Any
    hidden: Any
    gru_input: Any


def project_l2_ball(
    epsilon: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    """Project ``epsilon`` onto the closed flattened L2 ball."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    eps = jnp.asarray(epsilon)
    norm = jnp.linalg.norm(eps)
    scale = jnp.minimum(1.0, jnp.asarray(radius, dtype=eps.dtype) / (norm + 1e-30))
    return eps * scale


def epsilon_energy(epsilon: Any) -> float:
    """Return ``sum_t ||epsilon_t||_2^2`` as a Python float."""

    eps = np.asarray(epsilon, dtype=np.float64)
    return float(np.sum(np.square(eps)))


def declared_epsilon_l2_radius(
    run_spec: Mapping[str, Any],
    *,
    reach_length_m: float | None = None,
    budget_level_override: str | None = None,
    budget_scale_override: float | None = None,
) -> float:
    """Return the declared rollout L2 radius from a b8aa38e run spec."""

    if budget_level_override is not None:
        if budget_level_override not in BROAD_EPSILON_LEVELS:
            levels = ", ".join(sorted(BROAD_EPSILON_LEVELS))
            raise ValueError(
                f"Unknown budget_level_override {budget_level_override!r}; "
                f"expected one of {levels}."
            )
        contract = {
            **BROAD_EPSILON_LEVELS[budget_level_override],
            "reference_reach_m": BROAD_EPSILON_REFERENCE_REACH_M,
        }
        raw_radius = contract["closed_loop_epsilon_l2_15cm"]
        radius = float(raw_radius) * float(
            1.0 if budget_scale_override is None else budget_scale_override
        )
        if reach_length_m is not None:
            radius *= float(reach_length_m) / float(BROAD_EPSILON_REFERENCE_REACH_M)
        return radius

    config = run_spec.get("hps", {}).get("broad_epsilon_training", {})
    contract = config.get("budget_contract", {})
    raw_radius = contract.get("effective_l2_radius_15cm")
    if raw_radius is None:
        raw_radius = contract.get("closed_loop_epsilon_l2_15cm")
    if raw_radius is None:
        raise ValueError("run spec lacks broad_epsilon_training budget_contract L2 radius")
    radius = float(raw_radius) * float(config.get("budget_scale", 1.0) or 1.0)
    if bool(config.get("reach_length_scaling", False)) and reach_length_m is not None:
        reference = float(contract.get("reference_reach_m", 0.15) or 0.15)
        if reference <= 0:
            raise ValueError("reference_reach_m must be positive for reach-scaled budgets")
        radius *= float(reach_length_m) / reference
    return radius


def optimize_epsilon_sequence(
    objective: ObjectiveFn,
    *,
    shape: tuple[int, int],
    radius: float,
    n_steps: int,
    n_restarts: int,
    step_size: float,
    seed: int = 0,
    initial_candidates: Sequence[Any] = (),
) -> EpsilonOptimizationResult:
    """Run projected gradient ascent with best-incumbent retention."""

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if n_restarts < 0:
        raise ValueError("n_restarts must be non-negative")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    starts = [
        project_l2_ball(jnp.asarray(candidate, dtype=jnp.float64), radius)
        for candidate in initial_candidates
    ]
    n_random = max(0, n_restarts - len(starts))
    if n_random:
        keys = jr.split(jr.PRNGKey(seed), n_random)
        starts.extend(
            project_l2_ball(jr.normal(key, shape, dtype=jnp.float64), radius) for key in keys
        )
    if not starts:
        starts = [jnp.zeros(shape, dtype=jnp.float64)]

    value_and_grad = jax.value_and_grad(objective)
    best_global: dict[str, Any] | None = None
    restart_summaries: list[dict[str, float | int]] = []

    for restart_index, start in enumerate(starts):
        eps = start
        initial_value = float(objective(eps))
        best_eps = eps
        best_value = initial_value
        history = [
            {"step": 0, "objective": initial_value, "epsilon_l2": float(jnp.linalg.norm(eps))}
        ]
        for step in range(1, n_steps + 1):
            _value, grad = value_and_grad(eps)
            grad_norm = jnp.linalg.norm(grad)
            direction = grad / (grad_norm + 1e-30)
            eps = project_l2_ball(eps + float(step_size) * direction, radius)
            current_value = float(objective(eps))
            if current_value > best_value:
                best_eps = eps
                best_value = current_value
            history.append(
                {
                    "step": int(step),
                    "objective": current_value,
                    "best_objective": float(best_value),
                    "epsilon_l2": float(jnp.linalg.norm(eps)),
                    "gradient_l2": float(grad_norm),
                }
            )
        final_value = float(objective(eps))
        restart_summaries.append(
            {
                "restart_index": int(restart_index),
                "initial_objective": initial_value,
                "final_objective": final_value,
                "best_objective": float(best_value),
                "best_epsilon_l2": float(jnp.linalg.norm(best_eps)),
            }
        )
        if best_global is None or best_value > best_global["objective"]:
            best_global = {
                "epsilon": best_eps,
                "objective": float(best_value),
                "initial_objective": initial_value,
                "final_objective": final_value,
                "restart_index": restart_index,
                "history": tuple(history),
            }

    assert best_global is not None
    best_epsilon = np.asarray(best_global["epsilon"], dtype=np.float64)
    return EpsilonOptimizationResult(
        epsilon=best_epsilon,
        objective=float(best_global["objective"]),
        initial_objective=float(best_global["initial_objective"]),
        final_objective=float(best_global["final_objective"]),
        energy=epsilon_energy(best_epsilon),
        l2_norm=float(np.linalg.norm(best_epsilon)),
        restart_index=int(best_global["restart_index"]),
        history=best_global["history"],
        restart_summaries=tuple(restart_summaries),
    )


def materialize_gru_worst_case_epsilon_audit(
    *,
    source_experiment: str = DEFAULT_SOURCE_EXPERIMENT,
    result_experiment: str = DEFAULT_RESULT_EXPERIMENT,
    run_ids: Sequence[str] = DEFAULT_RUN_IDS,
    labels: Sequence[str] | None = None,
    n_rollout_trials: int = 1,
    n_steps: int = 12,
    n_restarts: int = 3,
    step_size: float | None = None,
    n_random_baselines: int = 3,
    seed: int = 0,
    budget_level_override: str | None = None,
    budget_scale_override: float | None = None,
    output_path: Path | None = None,
    note_path: Path | None = None,
    bulk_dir: Path | None = None,
    preferred_checkpoint_manifest_path: Path | None = None,
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize the same-channel worst-case epsilon audit for GRU rows."""

    if n_rollout_trials < 1:
        raise ValueError("n_rollout_trials must be at least 1")
    if n_random_baselines < 0:
        raise ValueError("n_random_baselines must be non-negative")

    output_path = output_path or (
        repo_root / "results" / result_experiment / "notes" / DEFAULT_OUTPUT_FILENAME
    )
    note_path = note_path or (
        repo_root / "results" / result_experiment / "notes" / DEFAULT_NOTE_FILENAME
    )
    bulk_dir = bulk_dir or repo_root / "_artifacts" / result_experiment / DEFAULT_BULK_SUBDIR
    mkdir_p(output_path.parent)
    mkdir_p(bulk_dir)

    runs = resolve_run_inputs(
        experiment=source_experiment,
        run_ids=tuple(run_ids),
        labels=labels,
        repo_root=repo_root,
    )
    run_summaries = {
        run.run_id: audit_run_worst_case_epsilon(
            run,
            source_experiment=source_experiment,
            n_rollout_trials=n_rollout_trials,
            n_steps=n_steps,
            n_restarts=n_restarts,
            step_size=step_size,
            n_random_baselines=n_random_baselines,
            seed=seed,
            budget_level_override=budget_level_override,
            budget_scale_override=budget_scale_override,
            bulk_dir=bulk_dir / run.run_id,
            preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
            checkpoint_selection_mode=checkpoint_selection_mode,
            repo_root=repo_root,
        )
        for run in runs
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": "020a65b",
        "source_experiment": source_experiment,
        "result_experiment": result_experiment,
        "scope": "same_channel_worst_case_full_state_epsilon_audit",
        "disturbance_channel": {
            "shape": ["time", 8],
            "injection": "mechanics.epsilon through C&S LSS B_w[:8, :] = I_8",
            "lag_history_direct_write": False,
            "budget": "flattened rollout L2 ball from run_spec hps.broad_epsilon_training",
            "budget_level_override": budget_level_override,
            "budget_scale_override": budget_scale_override,
        },
        "optimizer": {
            "method": "projected_gradient_ascent",
            "n_steps": int(n_steps),
            "n_restarts": int(n_restarts),
            "step_size": "radius * 0.25" if step_size is None else float(step_size),
            "best_incumbent_retention": True,
            "objective": "mean realized full Q/R/Q_f rollout cost over selected replicates/trials",
        },
        "comparators": {
            "zero": "all-zero epsilon sequence",
            "random_iid_projected": (
                "iid standard-normal T x 8 sequences projected to the same L2 ball"
            ),
        },
        "limits": {
            "default_run_ids": list(DEFAULT_RUN_IDS),
            "full_b8aa38e_matrix_cost": (
                "CLI defaults intentionally smoke one row; pass multiple --run-id values "
                "for the full row set because each row loads checkpoints and runs PGD "
                "through frozen GRU rollouts."
            ),
        },
        "runs": run_summaries,
    }
    public_manifest = _json_ready(manifest)
    output_path.write_text(json.dumps(public_manifest, indent=2, sort_keys=True) + "\n")
    note_path.write_text(render_worst_case_epsilon_markdown(public_manifest), encoding="utf-8")
    return public_manifest


def audit_run_worst_case_epsilon(
    run: RunFigureInputs,
    *,
    source_experiment: str,
    n_rollout_trials: int,
    n_steps: int,
    n_restarts: int,
    step_size: float | None,
    n_random_baselines: int,
    seed: int,
    budget_level_override: str | None,
    budget_scale_override: float | None,
    bulk_dir: Path,
    preferred_checkpoint_manifest_path: Path | None = None,
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Audit one frozen validation-selected GRU run."""

    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run.run_spec.get("seed", 42))))
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=source_experiment,
        run_id=run.run_id,
        run_spec=run.run_spec,
        preferred_manifest_path=preferred_checkpoint_manifest_path,
        checkpoint_selection_mode=checkpoint_selection_mode,
        repo_root=repo_root,
    )
    trial_specs = repeat_single_validation_trial(pair.task.validation_trials, n_rollout_trials)
    if "epsilon" not in getattr(trial_specs, "inputs", {}):
        return {
            "label": run.label,
            "status": "blocked",
            "reason": "trial_specs.inputs lacks epsilon; C&S LSS epsilon channel is unavailable",
        }

    base_epsilon_input = jnp.asarray(trial_specs.inputs["epsilon"], dtype=jnp.float64)
    horizon = int(base_epsilon_input.shape[-2])
    epsilon_dim = int(base_epsilon_input.shape[-1])
    if epsilon_dim != 8:
        return {
            "label": run.label,
            "status": "blocked",
            "reason": f"expected epsilon dim 8, got {epsilon_dim}",
        }

    reach_length = _validation_reach_length_m(trial_specs)
    radius = declared_epsilon_l2_radius(
        run.run_spec,
        reach_length_m=reach_length,
        budget_level_override=budget_level_override,
        budget_scale_override=budget_scale_override,
    )
    effective_step = float(step_size) if step_size is not None else max(radius * 0.25, 1e-12)
    objective = _build_objective(
        model=model,
        task=pair.task,
        trial_specs=trial_specs,
        n_replicates=n_replicates,
        seed=seed,
    )
    zero_epsilon = np.zeros((horizon, epsilon_dim), dtype=np.float64)
    random_epsilons = _projected_random_epsilons(
        shape=(horizon, epsilon_dim),
        radius=radius,
        n_random=n_random_baselines,
        seed=seed + 17,
    )
    optimization = optimize_epsilon_sequence(
        objective,
        shape=(horizon, epsilon_dim),
        radius=radius,
        n_steps=n_steps,
        n_restarts=n_restarts,
        step_size=effective_step,
        seed=seed,
        initial_candidates=(zero_epsilon, *random_epsilons),
    )

    zero = _evaluate_candidate(
        candidate_id="zero",
        epsilon=zero_epsilon,
        model=model,
        task=pair.task,
        trial_specs=trial_specs,
        n_replicates=n_replicates,
        seed=seed,
        budget_radius=radius,
    )
    optimized = _evaluate_candidate(
        candidate_id="optimized_pgd",
        epsilon=optimization.epsilon,
        model=model,
        task=pair.task,
        trial_specs=trial_specs,
        n_replicates=n_replicates,
        seed=seed,
        base_candidate=zero,
        budget_radius=radius,
    )
    random_candidates = [
        _evaluate_candidate(
            candidate_id=f"random_iid_projected_{idx}",
            epsilon=epsilon,
            model=model,
            task=pair.task,
            trial_specs=trial_specs,
            n_replicates=n_replicates,
            seed=seed,
            base_candidate=zero,
            budget_radius=radius,
        )
        for idx, epsilon in enumerate(random_epsilons)
    ]
    best_random = max(
        random_candidates,
        key=lambda row: float(row["objective_cost"]["total"]["mean"]),
        default=None,
    )
    mkdir_p(bulk_dir)
    arrays_path = bulk_dir / "epsilon_candidates.npz"
    np.savez_compressed(
        arrays_path,
        zero=zero_epsilon,
        optimized=optimization.epsilon,
        **{f"random_{idx}": epsilon for idx, epsilon in enumerate(random_epsilons)},
    )

    return {
        "label": run.label,
        "status": "evaluated",
        "run_spec_path": _repo_relative(run.run_spec_path, repo_root=repo_root),
        "artifact_dir": _repo_relative(run.artifact_dir, repo_root=repo_root),
        "checkpoint_selection": [
            selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
        ],
        "n_replicates": n_replicates,
        "n_rollout_trials_per_replicate": int(n_rollout_trials),
        "horizon": horizon,
        "epsilon_dim": epsilon_dim,
        "budget": {
            "l2_radius": radius,
            "energy": radius * radius,
            "reach_length_m": reach_length,
            "source": (
                f"BROAD_EPSILON_LEVELS[{budget_level_override!r}] override"
                if budget_level_override is not None
                else "run_spec.hps.broad_epsilon_training.budget_contract"
            ),
            "budget_level_override": budget_level_override,
            "budget_scale_override": budget_scale_override,
        },
        "optimizer": {
            "n_steps": int(n_steps),
            "n_restarts": int(n_restarts),
            "step_size": effective_step,
            "best_restart_index": optimization.restart_index,
            "initial_objective": optimization.initial_objective,
            "final_objective": optimization.final_objective,
            "best_objective": optimization.objective,
            "history": list(optimization.history),
            "restart_summaries": list(optimization.restart_summaries),
        },
        "candidates": {
            "zero": zero,
            "optimized_pgd": optimized,
            "random_iid_projected": random_candidates,
            "best_random_iid_projected": best_random,
        },
        "summary": {
            "optimized_delta_cost_total_mean": optimized["delta_vs_zero"]["delta_cost"]["total"][
                "mean"
            ],
            "optimized_cost_total_mean": optimized["objective_cost"]["total"]["mean"],
            "zero_cost_total_mean": zero["objective_cost"]["total"]["mean"],
            "best_random_cost_total_mean": None
            if best_random is None
            else best_random["objective_cost"]["total"]["mean"],
            "optimized_beats_zero": (
                optimized["objective_cost"]["total"]["mean"]
                >= zero["objective_cost"]["total"]["mean"]
            ),
            "optimized_beats_best_random": None
            if best_random is None
            else (
                optimized["objective_cost"]["total"]["mean"]
                >= best_random["objective_cost"]["total"]["mean"]
            ),
        },
        "bulk_arrays": {
            "path": _repo_relative(arrays_path, repo_root=repo_root),
            "format": "np.savez_compressed",
            "arrays": [
                "zero",
                "optimized",
                *[f"random_{idx}" for idx in range(len(random_epsilons))],
            ],
        },
    }


def _build_objective(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
) -> ObjectiveFn:
    def objective(epsilon: Float[Array, "T m_w"]) -> Float[Array, ""]:
        evaluation = _evaluate_jax_rollout(
            model=model,
            task=task,
            trial_specs=_with_epsilon_sequence(trial_specs, epsilon),
            n_replicates=n_replicates,
            seed=seed,
        )
        costs = _jax_full_qrf_rollout_cost(
            states=evaluation.mechanics_vector,
            commands=evaluation.command,
            initial_states=trial_specs.inits["mechanics.vector"],
        )
        return jnp.mean(costs["total"])

    return objective


def _evaluate_jax_rollout(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
) -> JaxRolloutEvaluation:
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_array(leaf, n_replicates),
    )
    batch_size = _infer_batch_size(trial_specs)

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return task.eval_trials(replicate_model, trial_specs, jr.split(key, batch_size))

    states = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(seed), n_replicates),
    )
    return JaxRolloutEvaluation(
        mechanics_vector=states.mechanics.vector,
        command=states.net.output,
        position=states.mechanics.effector.pos,
        velocity=states.mechanics.effector.vel,
        hidden=states.net.hidden,
        gru_input=states.net.input,
    )


def _evaluate_candidate(
    *,
    candidate_id: str,
    epsilon: Any,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
    base_candidate: Mapping[str, Any] | None = None,
    budget_radius: float | None = None,
) -> dict[str, Any]:
    candidate_specs = _with_epsilon_sequence(trial_specs, jnp.asarray(epsilon, dtype=jnp.float64))
    jax_eval = _evaluate_jax_rollout(
        model=model,
        task=task,
        trial_specs=candidate_specs,
        n_replicates=n_replicates,
        seed=seed,
    )
    cost_arrays = _jax_full_qrf_rollout_cost(
        states=jax_eval.mechanics_vector,
        commands=jax_eval.command,
        initial_states=trial_specs.inits["mechanics.vector"],
    )
    evaluation = _numpy_rollout_evaluation(jax_eval, candidate_specs)
    cost_summary = _cost_arrays_to_summary(cost_arrays)
    row = {
        "candidate_id": candidate_id,
        "epsilon": _epsilon_summary(epsilon, budget_radius=budget_radius),
        "objective_cost": _public_cost_summary(cost_summary),
        "endpoint_terminal": _endpoint_terminal_summary(evaluation),
    }
    if base_candidate is not None:
        base_cost = base_candidate["_private_cost_summary"]
        row["delta_vs_zero"] = {
            "delta_cost": _delta_cost_summary(base_cost, cost_summary),
            "response": summarize_perturbation_response(
                base_candidate["_private_evaluation"],
                evaluation,
                base_full_qrf_cost=base_cost,
                perturbed_full_qrf_cost=cost_summary,
            ),
        }
    row["_private_cost_summary"] = cost_summary
    row["_private_evaluation"] = evaluation
    return row


def _with_epsilon_sequence(trial_specs: Any, epsilon: Float[Array, "T m_w"]) -> Any:
    base = jnp.asarray(trial_specs.inputs["epsilon"], dtype=epsilon.dtype)
    payload = jnp.broadcast_to(epsilon, base.shape)
    inputs = dict(trial_specs.inputs)
    inputs["epsilon"] = payload
    return eqx.tree_at(lambda ts: ts.inputs, trial_specs, inputs)


def _jax_full_qrf_rollout_cost(
    *,
    states: Any,
    commands: Any,
    initial_states: Any,
    target_pos: Any = TARGET_POS,
) -> dict[str, Any]:
    _plant, schedule = build_canonical_game()
    state_array = jnp.asarray(states, dtype=jnp.float64)
    command_array = jnp.asarray(commands, dtype=jnp.float64)
    initial_array = jnp.asarray(initial_states, dtype=jnp.float64)
    initial_array = jnp.broadcast_to(
        initial_array,
        (*state_array.shape[:-2], state_array.shape[-1]),
    )
    x_pre = jnp.concatenate([initial_array[..., None, :], state_array[..., :-1, :]], axis=-2)
    x_pre = _goal_centered_vectors_jax(x_pre, target_pos=target_pos)
    x_terminal = _goal_centered_vectors_jax(state_array[..., -1, :], target_pos=target_pos)
    q = jnp.asarray(schedule.Q, dtype=jnp.float64)
    r = jnp.asarray(schedule.R, dtype=jnp.float64)
    q_f = jnp.asarray(schedule.Q_f, dtype=jnp.float64)
    state_terms = jnp.einsum("...ti,tij,...tj->...t", x_pre, q, x_pre)
    control_terms = jnp.einsum("...ti,tij,...tj->...t", command_array, r, command_array)
    terminal_terms = jnp.einsum("...i,ij,...j->...", x_terminal, q_f, x_terminal)
    stage_state = jnp.sum(state_terms, axis=-1)
    control = jnp.sum(control_terms, axis=-1)
    return {
        "total": stage_state + control + terminal_terms,
        "stage_state": stage_state,
        "control": control,
        "terminal": terminal_terms,
        "timewise_stage_state": state_terms,
        "timewise_control": control_terms,
    }


def _goal_centered_vectors_jax(values: Any, *, target_pos: Any) -> Any:
    arr = jnp.asarray(values, dtype=jnp.float64)
    target = jnp.asarray(target_pos, dtype=arr.dtype)
    if arr.shape[-1] % 8 != 0:
        raise ValueError(f"expected state dimension divisible by 8, got {arr.shape[-1]}")
    reshaped = arr.reshape((*arr.shape[:-1], arr.shape[-1] // 8, 8))
    centered = reshaped.at[..., 0:2].add(-target)
    return centered.reshape(arr.shape)


def _numpy_rollout_evaluation(jax_eval: JaxRolloutEvaluation, trial_specs: Any) -> Any:
    from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import RolloutEvaluation
    from rlrmp.analysis.pipelines.gru_pilot_figures import initial_effector_velocity

    evaluation = RolloutEvaluation(
        position=np.asarray(jax_eval.position, dtype=np.float64),
        velocity=np.asarray(jax_eval.velocity, dtype=np.float64),
        command=np.asarray(jax_eval.command, dtype=np.float64),
        hidden=np.asarray(jax_eval.hidden, dtype=np.float64),
        gru_input=np.asarray(jax_eval.gru_input, dtype=np.float64),
        initial_position=np.asarray(_initial_effector_position(trial_specs), dtype=np.float64),
        initial_velocity=np.asarray(initial_effector_velocity(trial_specs), dtype=np.float64),
        target_position=np.asarray(trial_specs.inputs["effector_target"].pos, dtype=np.float64),
        dt=0.01,
    )
    object.__setattr__(
        evaluation,
        "mechanics_vector",
        np.asarray(jax_eval.mechanics_vector, dtype=np.float64),
    )
    return evaluation


def _epsilon_summary(epsilon: Any, *, budget_radius: float | None = None) -> dict[str, Any]:
    eps = np.asarray(epsilon, dtype=np.float64)
    component_energy = np.sum(np.square(eps), axis=0)
    time_energy = np.sum(np.square(eps), axis=-1)
    l2_norm = float(np.linalg.norm(eps))
    energy = float(np.sum(np.square(eps)))
    budget = (
        {
            "l2_radius": float(budget_radius),
            "energy": float(budget_radius) * float(budget_radius),
            "within_l2_budget": l2_norm <= float(budget_radius) + 1e-12,
            "l2_slack": float(budget_radius) - l2_norm,
            "energy_slack": float(budget_radius) * float(budget_radius) - energy,
        }
        if budget_radius is not None
        else None
    )
    return {
        "shape": list(eps.shape),
        "energy": energy,
        "l2_norm": l2_norm,
        "budget_compliance": budget,
        "max_abs": float(np.max(np.abs(eps))) if eps.size else 0.0,
        "component_energy": component_energy.tolist(),
        "peak_time_index": int(np.argmax(time_energy)) if time_energy.size else None,
        "peak_time_energy": float(np.max(time_energy)) if time_energy.size else 0.0,
    }


def _endpoint_terminal_summary(evaluation: Any) -> dict[str, Any]:
    endpoint = np.linalg.norm(
        evaluation.position[:, :, -1, :] - evaluation.target_position[None, :, -1, :],
        axis=-1,
    )
    terminal_speed = np.linalg.norm(evaluation.velocity[:, :, -1, :], axis=-1)
    return {
        "endpoint_error_m": _summary_stats(endpoint),
        "terminal_speed_m_s": _summary_stats(terminal_speed),
    }


def _cost_arrays_to_summary(costs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "available",
        "lens": "realized_deterministic_rollout_full_qrf",
        **{
            key: {"values": np.asarray(costs[key], dtype=np.float64).tolist()}
            | _summary_stats(np.asarray(costs[key], dtype=np.float64))
            for key in ("total", "stage_state", "control", "terminal")
        },
    }


def _public_cost_summary(cost_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {metric: value for metric, value in cost_summary[key].items() if metric != "values"}
        for key in ("total", "stage_state", "control", "terminal")
    }


def _delta_cost_summary(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: _summary_stats(
            np.asarray(candidate[key]["values"], dtype=np.float64)
            - np.asarray(base[key]["values"], dtype=np.float64)
        )
        for key in ("total", "stage_state", "control", "terminal")
    }


def _projected_random_epsilons(
    *,
    shape: tuple[int, int],
    radius: float,
    n_random: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    keys = jr.split(jr.PRNGKey(seed), n_random)
    return tuple(
        np.asarray(
            project_l2_ball(jr.normal(key, shape, dtype=jnp.float64), radius),
            dtype=np.float64,
        )
        for key in keys
    )


def _validation_reach_length_m(trial_specs: Any) -> float:
    target = np.asarray(trial_specs.inputs["effector_target"].pos, dtype=np.float64)
    target_final = target[0, -1, :]
    initial = np.asarray(_initial_effector_position(trial_specs), dtype=np.float64)[0]
    return float(np.linalg.norm(target_final - initial))


def _initial_effector_position(trial_specs: Any) -> Any:
    for init_state in trial_specs.inits.values():
        position = getattr(init_state, "pos", None)
        if position is not None:
            return position
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 2:
            return jnp.asarray(init_state)[..., 0:2]
    raise ValueError("Trial spec does not include an effector position initial state")


def _infer_batch_size(trial_specs: Any) -> int:
    for value in getattr(trial_specs, "inputs", {}).values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
        pos = getattr(value, "pos", None)
        shape = getattr(pos, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    for init in getattr(trial_specs, "inits", {}).values():
        shape = getattr(init, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    return 1


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    return bool(
        eqx.is_array(leaf)
        and getattr(leaf, "ndim", 0) >= 1
        and int(getattr(leaf, "shape", (0,))[0]) == int(n_replicates)
    )


def _summary_stats(values: Any) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def render_worst_case_epsilon_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact tracked note for the audit manifest."""

    lines = [
        "# GRU Worst-Case Epsilon Audit",
        "",
        "Same-channel full-state epsilon audit for frozen b8aa38e GRU rows.",
        "",
        "| run | status | budget L2 | zero cost | optimized cost | delta cost | best random cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_id, row in manifest["runs"].items():
        if row.get("status") != "evaluated":
            lines.append(f"| `{run_id}` | {row.get('status')} | n/a | n/a | n/a | n/a | n/a |")
            continue
        best_random = row["summary"]["best_random_cost_total_mean"]
        lines.append(
            "| "
            f"`{run_id}` | evaluated | {row['budget']['l2_radius']:.8g} | "
            f"{row['summary']['zero_cost_total_mean']:.8g} | "
            f"{row['summary']['optimized_cost_total_mean']:.8g} | "
            f"{row['summary']['optimized_delta_cost_total_mean']:.8g} | "
            f"{'n/a' if best_random is None else f'{best_random:.8g}'} |"
        )
    lines.extend(
        [
            "",
            "Limits: this is open-loop projected ascent over one declared `T x 8` "
            "epsilon sequence, not a closed-loop Riccati adversary. Defaults smoke "
            "one b8aa38e row; pass explicit `--run-id` values for a broader matrix.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(val)
            for key, val in value.items()
            if not str(key).startswith("_private_")
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_RUN_IDS",
    "SCHEMA_VERSION",
    "declared_epsilon_l2_radius",
    "epsilon_energy",
    "materialize_gru_worst_case_epsilon_audit",
    "optimize_epsilon_sequence",
    "project_l2_ball",
]
