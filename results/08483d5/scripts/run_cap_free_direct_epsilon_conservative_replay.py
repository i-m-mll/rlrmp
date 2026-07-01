from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace

from compute_gru_pgd_damage_sanity import (
    full_qrf_cost,
    full_qrf_cost_context,
    json_ready,
    stats,
    trial_target_position,
    with_epsilon_delta,
)
from rlrmp.analysis.frozen_policy_gate import validate_direct_hvp_lambda_source
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_broad_epsilon_attribution import infer_batch_size
from rlrmp.io import update_marked_section
from rlrmp.train.cs_nominal_gru import _is_replicate_axis_array
from rlrmp.train.task_model import setup_task_model_pair


REPO_ROOT = Path(__file__).resolve().parents[3]
ISSUE = "08483d5"
RUN_ID = "h0_6d_no_pgd_const_band16_cpu"
RUN_SPEC_PATH = REPO_ROOT / "results" / ISSUE / "runs" / f"{RUN_ID}.json"
CHECKPOINT_PATH = (
    REPO_ROOT
    / "_artifacts"
    / ISSUE
    / "runs"
    / RUN_ID
    / "checkpoints"
    / "checkpoint_latest"
    / "model.eqx"
)
LAMBDA_SOURCE_PATH = REPO_ROOT / "results" / "06a4dc8" / "canonical_soft_lambda_hvp.json"
BETA105_DAMAGE_SOURCE_PATH = (
    REPO_ROOT / "results" / ISSUE / "notes" / "output_feedback_damage_estimate_beta1p05.json"
)
BETA14_DAMAGE_SOURCE_PATH = (
    REPO_ROOT / "results" / ISSUE / "notes" / "output_feedback_damage_estimate.json"
)
OUTPUT_JSON = (
    REPO_ROOT / "results" / ISSUE / "notes" / "cap_free_direct_epsilon_conservative_replay.json"
)
OUTPUT_MD = (
    REPO_ROOT / "results" / ISSUE / "notes" / "cap_free_direct_epsilon_conservative_replay.md"
)

DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42
DEFAULT_OPTIMIZER_STEPS = 12
DEFAULT_ADAPTIVE_ITERATIONS = 5
DEFAULT_ADAM_LR = 2e-5
DEFAULT_GRAD_CLIP_L2 = 1e6
DEFAULT_ETA = 0.1
DEFAULT_MAX_LOG_STEP = 0.1
DEFAULT_DEADBAND_FRACTION = 0.1


@dataclass(frozen=True)
class OptimizerConfig:
    """Cap-free direct-epsilon optimizer controls."""

    n_steps: int
    learning_rate: float
    grad_clip_l2: float
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    nonzero_energy_threshold: float = 1e-20
    nonzero_norm_threshold: float = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--optimizer-steps", type=int, default=DEFAULT_OPTIMIZER_STEPS)
    parser.add_argument("--adam-lr", type=float, default=DEFAULT_ADAM_LR)
    parser.add_argument("--grad-clip-l2", type=float, default=DEFAULT_GRAD_CLIP_L2)
    parser.add_argument("--adaptive-iterations", type=int, default=DEFAULT_ADAPTIVE_ITERATIONS)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--max-log-step", type=float, default=DEFAULT_MAX_LOG_STEP)
    parser.add_argument("--deadband-fraction", type=float, default=DEFAULT_DEADBAND_FRACTION)
    parser.add_argument(
        "--include-beta14",
        action="store_true",
        help="Also run the older beta 1.4 deterministic/noisy targets.",
    )
    args = parser.parse_args()
    start_s = time.perf_counter()

    opt = OptimizerConfig(
        n_steps=int(args.optimizer_steps),
        learning_rate=float(args.adam_lr),
        grad_clip_l2=float(args.grad_clip_l2),
    )

    run_spec = _read_json(RUN_SPEC_PATH)
    lambda_source = _read_json(LAMBDA_SOURCE_PATH)
    lambda_curv = float(lambda_source["pooled_summary"]["lambda_star_p90"])
    lambda0 = float(validate_direct_hvp_lambda_source(lambda_source, beta=1.05)["candidate_lambda"])

    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run_spec.get("seed", args.seed))))
    model_template = pair.model
    model = eqx.tree_deserialise_leaves(CHECKPOINT_PATH, model_template)

    trial_specs, batch_record = fixed_validation_batch(
        pair.task.validation_trials,
        batch_size=int(args.batch_size),
    )
    base_epsilon = jnp.asarray(trial_specs.inputs["epsilon"], dtype=jnp.float64)
    zero_delta = jnp.zeros_like(base_epsilon)
    replicate_keys = jr.split(jr.PRNGKey(int(args.seed)), n_replicates)
    target_position = trial_target_position(trial_specs)
    cost_context = full_qrf_cost_context(
        initial_states=jnp.asarray(trial_specs.inits["mechanics.vector"], dtype=jnp.float64),
        target_pos=jnp.asarray(target_position, dtype=jnp.float64),
    )

    def costs_for_delta(delta: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return rollout_costs_all_replicates(
            model=model,
            task=pair.task,
            trial_specs=with_epsilon_delta(trial_specs, delta),
            n_replicates=n_replicates,
            keys=replicate_keys,
            context=cost_context,
        )

    def objective_terms(delta: jnp.ndarray, lambda_value: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        costs = costs_for_delta(delta)
        task_cost = jnp.mean(jnp.asarray(costs["total"], dtype=jnp.float64))
        energy_per_trial = jnp.sum(jnp.square(delta), axis=tuple(range(1, delta.ndim)))
        energy = jnp.mean(energy_per_trial)
        objective = task_cost - lambda_value * energy
        return objective, task_cost, energy

    def objective_only(delta: jnp.ndarray, lambda_value: jnp.ndarray) -> jnp.ndarray:
        return objective_terms(delta, lambda_value)[0]

    objective_and_grad = eqx.filter_jit(jax.value_and_grad(objective_only))
    terms_jit = eqx.filter_jit(objective_terms)
    clean_cost_arrays = costs_for_delta(zero_delta)
    clean_summary = summarize_costs(clean_cost_arrays)

    def evaluate(lambda_value: float, label: str) -> dict[str, Any]:
        return optimize_direct_epsilon(
            lambda_value=float(lambda_value),
            label=label,
            zero_delta=zero_delta,
            objective_and_grad=objective_and_grad,
            terms_fn=terms_jit,
            costs_for_delta=costs_for_delta,
            clean_cost_arrays=clean_cost_arrays,
            opt=opt,
        )

    targets = beta105_targets()
    if bool(args.include_beta14):
        targets.extend(beta14_targets())
    target_rows = [
        adaptive_sequence(
            target_label=target["target_label"],
            damage_ref=float(target["damage_ref"]),
            damage_source_path=target["source_path"],
            lambda0=lambda0,
            evaluate=evaluate,
            iterations=int(args.adaptive_iterations),
            eta=float(args.eta),
            max_log_step=float(args.max_log_step),
            deadband_fraction=float(args.deadband_fraction),
        )
        for target in targets
    ]

    runtime_s = time.perf_counter() - start_s
    payload = {
        "schema_version": "rlrmp.08483d5_cap_free_direct_epsilon_conservative_replay.v1",
        "issue": ISSUE,
        "created_by": repo_rel(Path(__file__)),
        "command": command_record(),
        "runtime_seconds": runtime_s,
        "baseline": {
            "run_id": RUN_ID,
            "run_spec_path": repo_rel(RUN_SPEC_PATH),
            "checkpoint_path": repo_rel(CHECKPOINT_PATH),
            "checkpoint_kind": "clean 6D no-PGD H0 const_band16 baseline",
            "model_contract": {
                "n_replicates": n_replicates,
                "replicate_handling": "all checkpoint replicates aggregated",
                "no_integrator_state": bool(hps.model.no_integrator_state),
                "physical_state_dim": int(hps.model.physical_state_dim),
                "state_dim": int(hps.model.state_dim),
                "epsilon_dim": int(base_epsilon.shape[-1]),
                "horizon_steps": int(base_epsilon.shape[-2]),
            },
        },
        "batch": {
            **batch_record,
            "seed": int(args.seed),
            "paired_clean_and_adversarial_rollout_keys": True,
            "target_position_m": np.asarray(target_position).tolist(),
            "initial_position_m": np.asarray(
                trial_specs.inits["mechanics.vector"][..., :2], dtype=np.float64
            ).tolist(),
        },
        "lambda_source": {
            "path": repo_rel(LAMBDA_SOURCE_PATH),
            "lambda_curv_p90": lambda_curv,
            "lambda0_beta_1p05": lambda0,
            "cap_independent_source_validated": True,
            "source_objective_energy_convention": lambda_source["objective_contract"]["energy"],
        },
        "damage_references": [
            {
                "target_label": target["target_label"],
                "damage_ref": float(target["damage_ref"]),
                "source_path": target["source_path"],
            }
            for target in targets
        ],
        "optimizer": optimizer_json(opt),
        "adaptive_rule": {
            "aggregate_damage_estimate": "mean adversarial total cost minus mean clean total cost over all replicate/trial cells",
            "ema_initialization": "first aggregate damage for each target sequence",
            "ema_alpha": float(args.eta),
            "lambda_update": (
                "outside the deadband, log(lambda_next)=log(lambda)+"
                "clip(eta*log(max(ema_damage, eps)/target), +/-max_log_step)"
            ),
            "eta": float(args.eta),
            "max_log_step": float(args.max_log_step),
            "deadband_fraction": float(args.deadband_fraction),
            "deadband_action": "lambda unchanged when EMA damage is within target +/- deadband_fraction",
        },
        "cap_free_contract": {
            "uses_run_broad_epsilon_pgd_inner_maximizer": False,
            "uses_projection": False,
            "uses_radius_or_trust_region_bound": False,
            "uses_safety_cap": False,
            "epsilon_scale_role": "recorded output only, not a guard criterion",
            "stabilization_controls": [
                "Adam learning rate",
                "finite optimizer step count",
                "gradient L2 clipping",
                "nonfinite proposal rejection",
            ],
        },
        "clean_cost": clean_summary,
        "adaptive_targets": target_rows,
        "overall_assessment": {
            "all_targets_pass": all(bool(target["assessment"]["pass"]) for target in target_rows),
            "jumpiness_summary": jumpiness_summary(target_rows),
        },
    }

    OUTPUT_JSON.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    update_marked_section(
        OUTPUT_MD,
        "cap_free_direct_epsilon_conservative_replay",
        render_markdown(payload),
    )
    print(json.dumps(summary_for_stdout(payload), indent=2, sort_keys=True))


def fixed_validation_batch(trial_specs: Any, *, batch_size: int) -> tuple[Any, dict[str, Any]]:
    available = infer_batch_size(trial_specs)
    if available < batch_size:
        raise ValueError(
            f"normal validation trials expose only {available} trials; requested {batch_size}"
        )
    sliced = slice_trial_batch(trial_specs, batch_size=batch_size, available=available)
    target = np.asarray(trial_target_position(sliced), dtype=np.float64)
    initial = np.asarray(sliced.inits["mechanics.vector"][..., :2], dtype=np.float64)
    reach_lengths = np.linalg.norm(target - initial, axis=-1)
    repeated_singleton = bool(np.max(np.linalg.norm(target - target[0], axis=-1)) < 1e-12)
    return sliced, {
        "construction": (
            f"first {batch_size} trials from pair.task.validation_trials; the task exposed "
            f"{available} normal validation trials, so no singleton repetition helper was used"
        ),
        "batch_size": int(batch_size),
        "available_validation_trials": int(available),
        "source": "pair.task.validation_trials",
        "repeated_singleton": repeated_singleton,
        "reach_length_m": stats(reach_lengths),
        "limitation": None,
    }


def slice_trial_batch(trial_specs: Any, *, batch_size: int, available: int) -> Any:
    def slice_leaf(leaf: Any) -> Any:
        if eqx.is_array(leaf) and getattr(leaf, "ndim", 0) >= 1 and int(leaf.shape[0]) == available:
            return leaf[:batch_size]
        return leaf

    return jt.map(slice_leaf, trial_specs)


def rollout_costs_all_replicates(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    keys: Any,
    context: Mapping[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: bool(_is_replicate_axis_array(leaf, n_replicates)),
    )
    batch_size = int(trial_specs.inputs["epsilon"].shape[0])

    def eval_one(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return task.eval_trials(replicate_model, trial_specs, jr.split(key, batch_size))

    states = eqx.filter_vmap(eval_one, in_axes=(0, 0))(model_arrays, keys)
    return full_qrf_cost(
        states=jnp.asarray(states.mechanics.vector, dtype=jnp.float64),
        commands=jnp.asarray(states.net.output, dtype=jnp.float64),
        context=context,
    )


def optimize_direct_epsilon(
    *,
    lambda_value: float,
    label: str,
    zero_delta: jnp.ndarray,
    objective_and_grad: Any,
    terms_fn: Any,
    costs_for_delta: Callable[[jnp.ndarray], Mapping[str, jnp.ndarray]],
    clean_cost_arrays: Mapping[str, jnp.ndarray],
    opt: OptimizerConfig,
) -> dict[str, Any]:
    lambda_array = jnp.asarray(lambda_value, dtype=zero_delta.dtype)
    delta = zero_delta
    m = jnp.zeros_like(delta)
    v = jnp.zeros_like(delta)
    zero_objective, zero_task_cost, zero_energy = terms_fn(zero_delta, lambda_array)
    best_delta = zero_delta
    best_objective = zero_objective
    nonfinite_seen = False
    history = []

    for step in range(1, int(opt.n_steps) + 1):
        objective, grad = objective_and_grad(delta, lambda_array)
        grad, grad_norm_raw, grad_clip_scale = clip_by_global_l2(grad, opt.grad_clip_l2)
        m = opt.adam_b1 * m + (1.0 - opt.adam_b1) * grad
        v = opt.adam_b2 * v + (1.0 - opt.adam_b2) * jnp.square(grad)
        m_hat = m / (1.0 - opt.adam_b1**step)
        v_hat = v / (1.0 - opt.adam_b2**step)
        proposal = delta + opt.learning_rate * m_hat / (jnp.sqrt(v_hat) + opt.adam_eps)
        proposal_objective, proposal_task_cost, proposal_energy = terms_fn(proposal, lambda_array)
        proposal_finite = bool(np.asarray(jnp.isfinite(proposal_objective)))
        if not proposal_finite:
            nonfinite_seen = True
            history.append(
                {
                    "step": step,
                    "objective": float(np.asarray(objective)),
                    "proposal_finite": False,
                    "gradient_l2_raw": float(np.asarray(grad_norm_raw)),
                    "gradient_clip_scale": float(np.asarray(grad_clip_scale)),
                }
            )
            break
        improved = bool(np.asarray(proposal_objective > best_objective))
        if improved:
            best_delta = proposal
            best_objective = proposal_objective
        delta = proposal
        if step in {1, int(opt.n_steps)} or step % max(1, int(opt.n_steps) // 4) == 0:
            history.append(
                {
                    "step": step,
                    "objective": float(np.asarray(proposal_objective)),
                    "task_cost": float(np.asarray(proposal_task_cost)),
                    "energy_mean_per_trial": float(np.asarray(proposal_energy)),
                    "best_objective": float(np.asarray(best_objective)),
                    "gradient_l2_raw": float(np.asarray(grad_norm_raw)),
                    "gradient_clip_scale": float(np.asarray(grad_clip_scale)),
                    "proposal_finite": True,
                    "improved_best": improved,
                }
            )

    selected_cost_arrays = costs_for_delta(best_delta)
    selected_costs = summarize_costs(selected_cost_arrays)
    clean_costs = summarize_costs(clean_cost_arrays)
    damage_arrays = {
        key: jnp.asarray(selected_cost_arrays[key]) - jnp.asarray(clean_cost_arrays[key])
        for key in selected_cost_arrays
    }
    damage_summary = summarize_costs(damage_arrays)
    epsilon = summarize_epsilon_cap_free(best_delta)
    aggregate_damage = float(damage_summary["total"]["mean"])
    objective_gain = float(np.asarray(best_objective - zero_objective))
    nonzero = bool(
        epsilon["energy_mean_per_trial"] > opt.nonzero_energy_threshold
        and epsilon["l2_norm_mean_per_trial"] > opt.nonzero_norm_threshold
    )
    finite = bool(
        not nonfinite_seen
        and np.isfinite(float(np.asarray(best_objective)))
        and np.isfinite(float(selected_costs["total"]["mean"]))
    )
    return {
        "label": label,
        "lambda": float(lambda_value),
        "clean_cost": clean_costs["total"]["mean"],
        "selected_cost": selected_costs["total"]["mean"],
        "selected_cost_components": selected_costs,
        "paired_damage": aggregate_damage,
        "damage_components": damage_summary,
        "per_replicate_damage": per_replicate_summary(damage_arrays["total"]),
        "epsilon": epsilon,
        "finite": finite,
        "nonzero": nonzero,
        "objective_at_zero": float(np.asarray(zero_objective)),
        "task_cost_at_zero": float(np.asarray(zero_task_cost)),
        "energy_at_zero": float(np.asarray(zero_energy)),
        "selected_objective": float(np.asarray(best_objective)),
        "objective_gain": objective_gain,
        "nonfinite_seen": bool(nonfinite_seen),
        "optimizer_history": history,
    }


def clip_by_global_l2(grad: jnp.ndarray, max_norm: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    norm = jnp.linalg.norm(jnp.ravel(grad))
    scale = jnp.minimum(1.0, jnp.asarray(max_norm, dtype=grad.dtype) / (norm + 1e-30))
    return grad * scale, norm, scale


def adaptive_sequence(
    *,
    target_label: str,
    damage_ref: float,
    damage_source_path: str,
    lambda0: float,
    evaluate: Callable[[float, str], dict[str, Any]],
    iterations: int,
    eta: float,
    max_log_step: float,
    deadband_fraction: float,
) -> dict[str, Any]:
    rows = []
    lambda_value = float(lambda0)
    ema_damage: float | None = None
    for iteration in range(iterations):
        row = evaluate(lambda_value, f"{target_label}_conservative_{iteration:02d}")
        raw_damage = float(row["paired_damage"])
        if ema_damage is None:
            ema_damage = raw_damage
            ema_update = "initialized_from_first_aggregate_damage"
        else:
            ema_damage = (1.0 - float(eta)) * ema_damage + float(eta) * raw_damage
            ema_update = "ema=(1-eta)*previous_ema+eta*raw_damage"
        ema_ratio = ema_damage / float(damage_ref)
        lower = 1.0 - float(deadband_fraction)
        upper = 1.0 + float(deadband_fraction)
        in_deadband = bool(lower <= ema_ratio <= upper)
        if in_deadband:
            unclipped = 0.0
            applied = 0.0
            next_lambda = float(lambda_value)
            decision = "unchanged_deadband"
        else:
            safe_damage = max(float(ema_damage), float(damage_ref) * 1e-12)
            unclipped = float(eta) * math.log(safe_damage / float(damage_ref))
            applied = float(np.clip(unclipped, -float(max_log_step), float(max_log_step)))
            next_lambda = float(lambda_value * math.exp(applied))
            decision = "increase_lambda" if applied > 0 else "decrease_lambda"
        row.update(
            {
                "iteration": iteration,
                "damage_ref": float(damage_ref),
                "damage_over_ref": raw_damage / float(damage_ref),
                "ema_damage": float(ema_damage),
                "ema_damage_over_ref": float(ema_ratio),
                "deadband": {
                    "fraction": float(deadband_fraction),
                    "lower_ratio": lower,
                    "upper_ratio": upper,
                    "in_deadband": in_deadband,
                },
                "next_lambda": next_lambda,
                "update": {
                    "ema_update": ema_update,
                    "decision": decision,
                    "eta": float(eta),
                    "max_log_step": float(max_log_step),
                    "unclipped_log_step": unclipped,
                    "applied_log_step": applied,
                },
            }
        )
        rows.append(row)
        lambda_value = next_lambda
    return {
        "target_label": target_label,
        "damage_ref": float(damage_ref),
        "damage_source_path": damage_source_path,
        "rows": rows,
        "assessment": assess_rows(rows, damage_ref),
    }


def assess_rows(rows: list[Mapping[str, Any]], damage_ref: float) -> dict[str, Any]:
    raw = np.asarray([float(row["paired_damage"]) for row in rows], dtype=np.float64)
    ema = np.asarray([float(row["ema_damage"]) for row in rows], dtype=np.float64)
    lambdas = np.asarray([float(row["lambda"]) for row in rows], dtype=np.float64)
    finite_all = all(bool(row["finite"]) for row in rows)
    nonzero_all = all(bool(row["nonzero"]) for row in rows)
    last_in_deadband = bool(rows[-1]["deadband"]["in_deadband"])
    first_abs_error = abs(float(raw[0]) - damage_ref)
    last_abs_error = abs(float(raw[-1]) - damage_ref)
    moved_toward = last_abs_error < first_abs_error
    target_reached = last_in_deadband
    return {
        "pass": bool(finite_all and nonzero_all and (moved_toward or target_reached)),
        "criterion": "finite_nonzero_and_moved_toward_or_in_deadband",
        "target_reached": bool(target_reached),
        "finite_all": bool(finite_all),
        "nonzero_all": bool(nonzero_all),
        "first_damage": float(raw[0]),
        "last_damage": float(raw[-1]),
        "first_ema_damage": float(ema[0]),
        "last_ema_damage": float(ema[-1]),
        "damage_ref": float(damage_ref),
        "first_abs_error": float(first_abs_error),
        "last_abs_error": float(last_abs_error),
        "moved_toward_target": bool(moved_toward),
        "last_in_deadband": last_in_deadband,
        "raw_damage_step_abs_diffs": np.abs(np.diff(raw)).tolist(),
        "ema_damage_step_abs_diffs": np.abs(np.diff(ema)).tolist(),
        "lambda_step_ratios": (lambdas[1:] / lambdas[:-1]).tolist() if len(lambdas) > 1 else [],
        "max_raw_damage_step_abs_diff": float(np.max(np.abs(np.diff(raw)))) if len(raw) > 1 else 0.0,
        "max_ema_damage_step_abs_diff": float(np.max(np.abs(np.diff(ema)))) if len(ema) > 1 else 0.0,
    }


def summarize_costs(costs: Mapping[str, jnp.ndarray]) -> dict[str, dict[str, Any]]:
    return {key: stats(np.asarray(value, dtype=np.float64)) for key, value in costs.items()}


def per_replicate_summary(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        return {"shape": list(arr.shape), "summary": stats(arr)}
    rows = []
    means = []
    for index in range(arr.shape[0]):
        row_stats = stats(arr[index])
        row_stats["median"] = float(np.median(arr[index]))
        rows.append({"replicate_index": index, **row_stats})
        means.append(row_stats["mean"])
    means_arr = np.asarray(means, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "replicate_rows": rows,
        "replicate_mean_summary": {
            "mean": float(np.mean(means_arr)),
            "median": float(np.median(means_arr)),
            "std": float(np.std(means_arr)),
            "min": float(np.min(means_arr)),
            "max": float(np.max(means_arr)),
        },
    }


def summarize_epsilon_cap_free(epsilon: Any) -> dict[str, Any]:
    eps = np.asarray(epsilon, dtype=np.float64)
    per_trial_energy = np.sum(np.square(eps), axis=tuple(range(1, eps.ndim)))
    norms = np.sqrt(per_trial_energy)
    return {
        "shape": list(eps.shape),
        "energy_total": float(np.sum(per_trial_energy)),
        "energy_mean_per_trial": float(np.mean(per_trial_energy)),
        "energy_max_per_trial": float(np.max(per_trial_energy)),
        "l2_norm_mean_per_trial": float(np.mean(norms)),
        "l2_norm_max_per_trial": float(np.max(norms)),
        "max_abs": float(np.max(np.abs(eps))) if eps.size else 0.0,
    }


def optimizer_json(opt: OptimizerConfig) -> dict[str, Any]:
    return {
        "method": "cap_free_adam_ascent",
        "objective": "mean(task_cost) - lambda * mean(sum_t,d epsilon[t,d]^2)",
        "initialization": "zero_epsilon",
        "n_steps": int(opt.n_steps),
        "learning_rate": float(opt.learning_rate),
        "gradient_clip_l2": float(opt.grad_clip_l2),
        "adam_b1": float(opt.adam_b1),
        "adam_b2": float(opt.adam_b2),
        "adam_eps": float(opt.adam_eps),
        "selected_by": "best finite soft objective encountered",
        "scientific_bound": None,
    }


def beta105_targets() -> list[dict[str, Any]]:
    source = _read_json(BETA105_DAMAGE_SOURCE_PATH)
    return [
        {
            "target_label": "beta1p05_deterministic_output_feedback",
            "damage_ref": source["rollouts"]["deterministic_noise_off"]["paired_damage"],
            "source_path": repo_rel(BETA105_DAMAGE_SOURCE_PATH),
        },
        {
            "target_label": "beta1p05_paired_nominal_noise_output_feedback",
            "damage_ref": source["rollouts"]["nominal_noise_paired"]["paired_damage"],
            "source_path": repo_rel(BETA105_DAMAGE_SOURCE_PATH),
        },
    ]


def beta14_targets() -> list[dict[str, Any]]:
    source = _read_json(BETA14_DAMAGE_SOURCE_PATH)
    return [
        {
            "target_label": "beta1p4_deterministic_output_feedback",
            "damage_ref": source["rollouts"]["deterministic_noise_off"]["paired_damage"],
            "source_path": repo_rel(BETA14_DAMAGE_SOURCE_PATH),
        },
        {
            "target_label": "beta1p4_paired_nominal_noise_output_feedback",
            "damage_ref": source["rollouts"]["nominal_noise_paired"]["paired_damage"],
            "source_path": repo_rel(BETA14_DAMAGE_SOURCE_PATH),
        },
    ]


def jumpiness_summary(target_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for target in target_rows:
        assessment = target["assessment"]
        out[target["target_label"]] = {
            "max_raw_damage_step_abs_diff": assessment["max_raw_damage_step_abs_diff"],
            "max_ema_damage_step_abs_diff": assessment["max_ema_damage_step_abs_diff"],
            "lambda_step_ratios": assessment["lambda_step_ratios"],
            "last_in_deadband": assessment["last_in_deadband"],
        }
    return out


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Cap-Free Direct-Epsilon Conservative Replay",
        "",
        "## Headline",
        "",
        (
            "This no-launch replay loads the clean 6D no-PGD H0 `const_band16` baseline, "
            "uses normal 15 cm validation trials with varying reach directions, and "
            "aggregates each damage estimate over all five GRU checkpoint replicates."
        ),
        "",
        (
            "The inner objective is `mean(task_cost) - lambda * "
            "mean(sum_t,d epsilon[t,d]^2)`. No projection, safety cap, inherited radius, "
            "or trust-region value is used."
        ),
        "",
        "## Inputs",
        "",
        f"- Command: `{payload['command']['full_command']}`.",
        f"- Runtime: `{payload['runtime_seconds']:.1f}` seconds.",
        f"- Run spec: `{payload['baseline']['run_spec_path']}`.",
        f"- Checkpoint: `{payload['baseline']['checkpoint_path']}`.",
        f"- Lambda source: `{payload['lambda_source']['path']}`.",
        f"- Batch: {payload['batch']['batch_size']} trials from "
        f"{payload['batch']['available_validation_trials']} validation trials; "
        f"repeated singleton: `{payload['batch']['repeated_singleton']}`.",
        f"- Reach length mean: `{payload['batch']['reach_length_m']['mean']:.8g}` m; "
        f"std: `{payload['batch']['reach_length_m']['std']:.3g}`.",
        f"- Replicates: all {payload['baseline']['model_contract']['n_replicates']}.",
        "",
        "## Optimizer And Adaptive Rule",
        "",
        f"- Optimizer steps: `{payload['optimizer']['n_steps']}`; Adam learning rate: "
        f"`{payload['optimizer']['learning_rate']}`; gradient clip L2: "
        f"`{payload['optimizer']['gradient_clip_l2']}`.",
        f"- `lambda0` beta 1.05: `{payload['lambda_source']['lambda0_beta_1p05']:.9g}`.",
        f"- EMA initializes from the first aggregate damage. EMA alpha, lambda eta: "
        f"`{payload['adaptive_rule']['eta']}`.",
        f"- Max log lambda step: `{payload['adaptive_rule']['max_log_step']}`; deadband: "
        f"`+/-{100 * payload['adaptive_rule']['deadband_fraction']:.0f}%`.",
        "",
    ]
    for target in payload["adaptive_targets"]:
        assessment = target["assessment"]
        lines.extend(
            [
                f"## Target: {target['target_label']}",
                "",
                f"- Reference damage: `{target['damage_ref']:.10g}` from "
                f"`{target['damage_source_path']}`.",
                f"- First raw damage: `{assessment['first_damage']:.8g}`; last raw damage: "
                f"`{assessment['last_damage']:.8g}`.",
                f"- First EMA damage: `{assessment['first_ema_damage']:.8g}`; last EMA damage: "
                f"`{assessment['last_ema_damage']:.8g}`.",
                f"- Max raw step change: `{assessment['max_raw_damage_step_abs_diff']:.8g}`; "
                f"max EMA step change: `{assessment['max_ema_damage_step_abs_diff']:.8g}`.",
                f"- Last decision in deadband: `{assessment['last_in_deadband']}`; "
                f"target reached: `{assessment['target_reached']}`; "
                f"directional/smoothness criterion: `{assessment['pass']}`.",
                "",
                "| iter | lambda | raw damage | EMA damage | damage/ref | EMA/ref | decision | "
                "log step | next lambda | energy/trial | finite | nonzero |",
                "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|:---:|",
            ]
        )
        for row in target["rows"]:
            lines.append(format_adaptive_row(row))
        lines.extend(
            [
                "",
                "Per-replicate damage on the final row:",
                "",
                "| replicate | mean | median | std | min | max |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rep in target["rows"][-1]["per_replicate_damage"]["replicate_rows"]:
            lines.append(
                f"| {rep['replicate_index']} | {rep['mean']:.8g} | "
                f"{rep.get('median', rep['mean']):.8g} | {rep['std']:.8g} | "
                f"{rep['min']:.8g} | {rep['max']:.8g} |"
            )
        summary = target["rows"][-1]["per_replicate_damage"]["replicate_mean_summary"]
        lines.extend(
            [
                "",
                (
                    "Final-row replicate-mean summary: "
                    f"mean `{summary['mean']:.8g}`, median `{summary['median']:.8g}`, "
                    f"std `{summary['std']:.8g}`."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "The conservative rule is less jumpy if the EMA step changes are smaller than "
                "the raw damage changes and lambda changes stay bounded by the 0.1 log-step cap. "
                "It still fails as a target-tracking rule if the raw or EMA sequence moves away "
                "from the requested damage target or never enters the deadband."
            ),
            "",
            "## Residual Uncertainty",
            "",
            "- This is a local frozen replay, not controller training.",
            "- The first 64 of 72 validation trials are used to satisfy the requested batch size; "
            "reach length is fixed at 15 cm but directions vary.",
            "- Optimizer iterations are kept practical for CPU; batch size and replicate aggregation are not reduced.",
            "- The unconstrained inner optimizer can still pick very large epsilon if the soft objective rewards it.",
            "",
            "## Checks Run",
            "",
            f"- `{payload['command']['full_command']}`",
            "",
        ]
    )
    return "\n".join(lines)


def format_adaptive_row(row: Mapping[str, Any]) -> str:
    return (
        f"| {row['iteration']} | {row['lambda']:.6g} | {row['paired_damage']:.6g} | "
        f"{row['ema_damage']:.6g} | {row['damage_over_ref']:.5g} | "
        f"{row['ema_damage_over_ref']:.5g} | {row['update']['decision']} | "
        f"{row['update']['applied_log_step']:.5g} | {row['next_lambda']:.6g} | "
        f"{row['epsilon']['energy_mean_per_trial']:.6g} | {row['finite']} | "
        f"{row['nonzero']} |"
    )


def summary_for_stdout(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "json": repo_rel(OUTPUT_JSON),
        "markdown": repo_rel(OUTPUT_MD),
        "runtime_seconds": payload["runtime_seconds"],
        "all_targets_pass": payload["overall_assessment"]["all_targets_pass"],
        "targets": {
            target["target_label"]: {
                "pass": target["assessment"]["pass"],
                "first_damage": target["assessment"]["first_damage"],
                "last_damage": target["assessment"]["last_damage"],
                "last_ema_damage": target["assessment"]["last_ema_damage"],
                "damage_ref": target["damage_ref"],
                "last_in_deadband": target["assessment"]["last_in_deadband"],
            }
            for target in payload["adaptive_targets"]
        },
    }


def command_record() -> dict[str, Any]:
    argv = " ".join(shell_quote(part) for part in sys.argv)
    return {
        "argv": argv,
        "full_command": f"PYTHONPATH=src uv run --no-sync python {argv}",
        "pythonpath_required": "PYTHONPATH=src",
        "notes": "PYTHONPATH=src avoids stale editable rlrmp imports from sibling worktrees.",
    }


def shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "-_./:=+" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_rel(path: Path) -> str:
    try:
        return str(Path(path).absolute().relative_to(REPO_ROOT.absolute()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
