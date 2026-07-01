from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
from collections.abc import Mapping
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
    repeat_single_validation_trial,
    stats,
    trial_target_position,
    with_epsilon_delta,
)
from rlrmp.analysis.frozen_policy_gate import validate_direct_hvp_lambda_source
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.io import update_marked_section
from rlrmp.train.cs_nominal_gru import (
    _is_replicate_axis_array,
    _with_single_replicate_state_initializers,
)
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
DAMAGE_SOURCE_PATH = REPO_ROOT / "results" / ISSUE / "notes" / "output_feedback_damage_estimate.json"
OUTPUT_JSON = REPO_ROOT / "results" / ISSUE / "notes" / "cap_free_direct_epsilon_adaptive_replay.json"
OUTPUT_MD = REPO_ROOT / "results" / ISSUE / "notes" / "cap_free_direct_epsilon_adaptive_replay.md"

DETERMINISTIC_DAMAGE_REF = 3704.96326882
PAIRED_NOISE_DAMAGE_REF = 6131.6906765
DEFAULT_BATCH_SIZE = 2
DEFAULT_REPLICATE_INDEX = 0
DEFAULT_SEED = 42


@dataclass(frozen=True)
class OptimizerConfig:
    """Cap-free direct-epsilon optimizer controls."""

    n_steps: int
    learning_rate: float
    grad_clip_l2: float
    adam_b1: float
    adam_b2: float
    adam_eps: float
    nonzero_energy_threshold: float
    nonzero_norm_threshold: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--replicate-index", type=int, default=DEFAULT_REPLICATE_INDEX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--optimizer-steps", type=int, default=35)
    parser.add_argument("--adam-lr", type=float, default=2e-5)
    parser.add_argument("--grad-clip-l2", type=float, default=1e6)
    parser.add_argument("--adaptive-iterations", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--max-log-step", type=float, default=0.75)
    args = parser.parse_args()

    opt = OptimizerConfig(
        n_steps=int(args.optimizer_steps),
        learning_rate=float(args.adam_lr),
        grad_clip_l2=float(args.grad_clip_l2),
        adam_b1=0.9,
        adam_b2=0.999,
        adam_eps=1e-8,
        nonzero_energy_threshold=1e-20,
        nonzero_norm_threshold=1e-12,
    )

    run_spec = _read_json(RUN_SPEC_PATH)
    damage_source = _read_json(DAMAGE_SOURCE_PATH)
    lambda_source = _read_json(LAMBDA_SOURCE_PATH)
    lambda_curv = float(lambda_source["pooled_summary"]["lambda_star_p90"])
    lambda0 = float(validate_direct_hvp_lambda_source(lambda_source, beta=1.05)["candidate_lambda"])

    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run_spec.get("seed", args.seed))))
    model_template = pair.model
    model_ensemble = eqx.tree_deserialise_leaves(CHECKPOINT_PATH, model_template)
    model = _replicate_model(model_ensemble, hps, int(args.replicate_index))

    trial_specs = repeat_single_validation_trial(pair.task.validation_trials, int(args.batch_size))
    base_epsilon = jnp.asarray(trial_specs.inputs["epsilon"], dtype=jnp.float64)
    zero_delta = jnp.zeros_like(base_epsilon)
    key = jr.fold_in(jr.PRNGKey(int(args.seed)), int(args.replicate_index))
    rollout_keys = jr.split(key, int(args.batch_size))
    target_position = trial_target_position(trial_specs)
    cost_context = full_qrf_cost_context(
        initial_states=jnp.asarray(trial_specs.inits["mechanics.vector"], dtype=jnp.float64),
        target_pos=jnp.asarray(target_position[0], dtype=jnp.float64),
    )

    def costs_for_delta(delta: jnp.ndarray) -> dict[str, jnp.ndarray]:
        candidate = with_epsilon_delta(trial_specs, delta)
        states = pair.task.eval_trials(model, candidate, rollout_keys)
        return full_qrf_cost(
            states=jnp.asarray(states.mechanics.vector, dtype=jnp.float64),
            commands=jnp.asarray(states.net.output, dtype=jnp.float64),
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

    clean_costs = costs_for_delta(zero_delta)
    clean_summary = summarize_costs(clean_costs)

    def evaluate(lambda_value: float, label: str) -> dict[str, Any]:
        return optimize_direct_epsilon(
            lambda_value=float(lambda_value),
            label=label,
            zero_delta=zero_delta,
            objective_and_grad=objective_and_grad,
            terms_fn=terms_jit,
            costs_for_delta=costs_for_delta,
            clean_task_cost=float(clean_summary["total"]["mean"]),
            opt=opt,
        )

    bracket_rows = [
        evaluate(lambda_curv * multiplier, f"bracket_x{multiplier:g}")
        for multiplier in (100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01)
    ]
    bracket = estimate_activity_bracket(bracket_rows)

    target_rows = [
        adaptive_sequence(
            target_label="deterministic_output_feedback",
            damage_ref=DETERMINISTIC_DAMAGE_REF,
            lambda0=lambda0,
            evaluate=evaluate,
            iterations=int(args.adaptive_iterations),
            eta=float(args.eta),
            max_log_step=float(args.max_log_step),
        ),
        adaptive_sequence(
            target_label="paired_nominal_noise_output_feedback",
            damage_ref=PAIRED_NOISE_DAMAGE_REF,
            lambda0=lambda0,
            evaluate=evaluate,
            iterations=int(args.adaptive_iterations),
            eta=float(args.eta),
            max_log_step=float(args.max_log_step),
        ),
    ]

    payload = {
        "schema_version": "rlrmp.08483d5_cap_free_direct_epsilon_adaptive_replay.v1",
        "issue": ISSUE,
        "created_by": repo_rel(Path(__file__)),
        "baseline": {
            "run_id": RUN_ID,
            "run_spec_path": repo_rel(RUN_SPEC_PATH),
            "checkpoint_path": repo_rel(CHECKPOINT_PATH),
            "checkpoint_kind": "clean 6D no-PGD H0 const_band16 baseline",
            "model_contract": {
                "n_replicates": n_replicates,
                "replicate_index_used": int(args.replicate_index),
                "no_integrator_state": bool(hps.model.no_integrator_state),
                "physical_state_dim": int(hps.model.physical_state_dim),
                "state_dim": int(hps.model.state_dim),
                "epsilon_dim": int(base_epsilon.shape[-1]),
                "horizon_steps": int(base_epsilon.shape[-2]),
            },
        },
        "batch": {
            "construction": (
                "repeat_single_validation_trial(pair.task.validation_trials, batch_size), "
                "the fixed +x 15 cm nominal validation reach used by the prior local replays"
            ),
            "batch_size": int(args.batch_size),
            "replicate_handling": (
                "single frozen replicate selected from the five-replicate ensemble for CPU cost"
            ),
            "replicate_index": int(args.replicate_index),
            "seed": int(args.seed),
            "paired_clean_and_adversarial_rollout_keys": True,
            "target_position_m": np.asarray(target_position).tolist(),
        },
        "lambda_source": {
            "path": repo_rel(LAMBDA_SOURCE_PATH),
            "lambda_curv_p90": lambda_curv,
            "lambda0_beta_1p05": lambda0,
            "cap_independent_source_validated": True,
            "source_objective_energy_convention": lambda_source["objective_contract"]["energy"],
        },
        "damage_references": {
            "path": repo_rel(DAMAGE_SOURCE_PATH),
            "deterministic_output_feedback": DETERMINISTIC_DAMAGE_REF,
            "paired_nominal_noise_output_feedback": PAIRED_NOISE_DAMAGE_REF,
            "source_values": damage_source,
        },
        "optimizer": optimizer_json(opt),
        "cap_free_contract": {
            "uses_run_broad_epsilon_pgd_inner_maximizer": False,
            "uses_projection": False,
            "uses_radius_or_trust_region_bound": False,
            "epsilon_scale_role": "recorded output only, not a guard criterion",
            "stabilization_controls": [
                "Adam learning rate",
                "finite optimizer step count",
                "gradient L2 clipping",
                "nonfinite proposal rejection",
            ],
        },
        "clean_cost": clean_summary,
        "activity_bracket": bracket,
        "activity_bracket_rows": bracket_rows,
        "adaptive_targets": target_rows,
        "overall_assessment": {
            "all_targets_pass": all(bool(target["assessment"]["pass"]) for target in target_rows),
            "target_labels": [target["target_label"] for target in target_rows],
        },
    }

    OUTPUT_JSON.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    update_marked_section(OUTPUT_MD, "cap_free_direct_epsilon_adaptive_replay", render_markdown(payload))
    print(json.dumps(summary_for_stdout(payload), indent=2, sort_keys=True))


def optimize_direct_epsilon(
    *,
    lambda_value: float,
    label: str,
    zero_delta: jnp.ndarray,
    objective_and_grad: Any,
    terms_fn: Any,
    costs_for_delta: Any,
    clean_task_cost: float,
    opt: OptimizerConfig,
) -> dict[str, Any]:
    """Run cap-free Adam ascent and select the best finite soft objective."""

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
        if step in {1, int(opt.n_steps)} or step % max(1, int(opt.n_steps) // 5) == 0:
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

    selected_costs = summarize_costs(costs_for_delta(best_delta))
    epsilon = summarize_epsilon_cap_free(best_delta)
    paired_damage = float(selected_costs["total"]["mean"] - clean_task_cost)
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
        "clean_cost": clean_task_cost,
        "selected_cost": selected_costs["total"]["mean"],
        "selected_cost_components": selected_costs,
        "paired_damage": paired_damage,
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
    lambda0: float,
    evaluate: Any,
    iterations: int,
    eta: float,
    max_log_step: float,
) -> dict[str, Any]:
    rows = []
    lambda_value = float(lambda0)
    for iteration in range(iterations):
        row = evaluate(lambda_value, f"{target_label}_adaptive_{iteration:02d}")
        damage = max(float(row["paired_damage"]), float(damage_ref) * 1e-12)
        unclipped = float(eta) * math.log(damage / float(damage_ref))
        applied = float(np.clip(unclipped, -float(max_log_step), float(max_log_step)))
        next_lambda = float(lambda_value * math.exp(applied))
        row.update(
            {
                "iteration": iteration,
                "damage_ref": float(damage_ref),
                "damage_over_ref": float(row["paired_damage"]) / float(damage_ref),
                "next_lambda": next_lambda,
                "update": {
                    "formula": (
                        "log(lambda_next)=log(lambda)+clip(eta*log(max(damage,eps)/D_ref), "
                        "+/-max_log_step)"
                    ),
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
        "rows": rows,
        "assessment": assess_rows(rows, damage_ref),
    }


def assess_rows(rows: list[Mapping[str, Any]], damage_ref: float) -> dict[str, Any]:
    first_damage = float(rows[0]["paired_damage"])
    last_damage = float(rows[-1]["paired_damage"])
    finite_all = all(bool(row["finite"]) for row in rows)
    nonzero_all = all(bool(row["nonzero"]) for row in rows)
    moved_toward = abs(last_damage - damage_ref) < abs(first_damage - damage_ref)
    damage_finite = np.isfinite(last_damage)
    passed = bool(finite_all and nonzero_all and moved_toward and damage_finite)
    blockers = []
    if not finite_all:
        blockers.append("nonfinite optimizer or selected cost occurred")
    if not nonzero_all:
        blockers.append("the cap-free optimizer selected zero epsilon on at least one row")
    if not moved_toward:
        blockers.append("damage did not move closer to the target over the adaptive sequence")
    if not damage_finite:
        blockers.append("damage was nonfinite")
    return {
        "pass": passed,
        "finite_all": bool(finite_all),
        "nonzero_all": bool(nonzero_all),
        "moved_toward_target": bool(moved_toward),
        "first_damage": first_damage,
        "last_damage": last_damage,
        "damage_ref": float(damage_ref),
        "first_abs_error": abs(first_damage - damage_ref),
        "last_abs_error": abs(last_damage - damage_ref),
        "blockers": blockers,
    }


def estimate_activity_bracket(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: float(row["lambda"]))
    active = [row for row in sorted_rows if bool(row["nonzero"])]
    inactive = [row for row in sorted_rows if not bool(row["nonzero"])]
    lower_active = max((float(row["lambda"]) for row in active), default=None)
    upper_inactive = min(
        (
            float(row["lambda"])
            for row in inactive
            if lower_active is None or float(row["lambda"]) > lower_active
        ),
        default=None,
    )
    return {
        "method": "coarse optimizer-dependent activity threshold under cap-free Adam",
        "thresholds": {
            "energy_mean_per_trial": 1e-20,
            "l2_norm_mean_per_trial": 1e-12,
        },
        "lower_active_lambda": lower_active,
        "upper_inactive_lambda": upper_inactive,
        "bracketed": bool(lower_active is not None and upper_inactive is not None),
        "interpretation": (
            "This is a batch- and optimizer-dependent activity bracket, not a theoretical "
            "lambda_zero estimate."
        ),
    }


def summarize_costs(costs: Mapping[str, jnp.ndarray]) -> dict[str, dict[str, Any]]:
    return {key: stats(np.asarray(value, dtype=np.float64)) for key, value in costs.items()}


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


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Cap-Free Direct-Epsilon Adaptive Replay",
        "",
        "## Headline",
        "",
        (
            "This replay loads the clean 6D no-PGD H0 `const_band16` baseline and runs a "
            "cap-free direct-epsilon inner optimizer with frozen controller weights."
        ),
        "",
        "No projection, inherited radius, trust-region value, or cap/guard criterion is used. "
        "Selected epsilon energy and norm are reported only as outcomes.",
        "",
        "## Inputs",
        "",
        f"- Run spec: `{payload['baseline']['run_spec_path']}`.",
        f"- Checkpoint: `{payload['baseline']['checkpoint_path']}`.",
        f"- Lambda source: `{payload['lambda_source']['path']}`.",
        f"- Damage references: `{payload['damage_references']['path']}`.",
        f"- Batch: {payload['batch']['batch_size']} repeated fixed +x 15 cm validation trials; "
        f"replicate {payload['batch']['replicate_index']} of "
        f"{payload['baseline']['model_contract']['n_replicates']} was used for CPU cost.",
        "",
        "## Optimizer",
        "",
        f"- Method: `{payload['optimizer']['method']}` from zero epsilon.",
        f"- Objective: `{payload['optimizer']['objective']}`.",
        f"- Steps: `{payload['optimizer']['n_steps']}`; Adam learning rate: "
        f"`{payload['optimizer']['learning_rate']}`; gradient clip L2: "
        f"`{payload['optimizer']['gradient_clip_l2']}`.",
        f"- `lambda_curv_p90`: `{payload['lambda_source']['lambda_curv_p90']:.9g}`; "
        f"`lambda0` beta 1.05: `{payload['lambda_source']['lambda0_beta_1p05']:.9g}`.",
        "",
        "## Activity Bracket",
        "",
        payload["activity_bracket"]["interpretation"],
        "",
        "| lambda | damage | objective gain | energy/trial | norm/trial | finite | nonzero |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in payload["activity_bracket_rows"]:
        lines.append(format_row(row, include_next=False))

    for target in payload["adaptive_targets"]:
        assessment = target["assessment"]
        lines.extend(
            [
                "",
                f"## Adaptive Target: {target['target_label']}",
                "",
                f"- Reference damage: `{target['damage_ref']:.10g}`.",
                f"- Pass frozen criterion: `{assessment['pass']}`.",
                f"- First damage: `{assessment['first_damage']:.8g}`; last damage: "
                f"`{assessment['last_damage']:.8g}`.",
                f"- First absolute error: `{assessment['first_abs_error']:.8g}`; last absolute "
                f"error: `{assessment['last_abs_error']:.8g}`.",
                "",
                "| iter | lambda | clean cost | selected cost | damage | damage/ref | energy/trial | "
                "norm/trial | finite | nonzero | objective gain | next lambda |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|",
            ]
        )
        for row in target["rows"]:
            lines.append(format_adaptive_row(row))
        if assessment["blockers"]:
            lines.extend(["", "Blockers:"])
            lines.extend(f"- {blocker}." for blocker in assessment["blockers"])
        else:
            lines.extend(["", "No blocker under the stated frozen criterion."])

    lines.extend(
        [
            "",
            "## Residual Uncertainties",
            "",
            "- This is a small local frozen replay, not a training launch.",
            "- The activity bracket is optimizer- and batch-dependent.",
            "- CPU cost kept the replay to one replicate and a small repeated validation batch.",
            "- The direct-epsilon optimizer is unconstrained; if damage grows sharply at lower lambda, "
            "that is reported as optimizer behavior rather than clipped away.",
            "",
        ]
    )
    return "\n".join(lines)


def format_row(row: Mapping[str, Any], *, include_next: bool) -> str:
    base = (
        f"| {row['lambda']:.6g} | {row['paired_damage']:.6g} | "
        f"{row['objective_gain']:.6g} | {row['epsilon']['energy_mean_per_trial']:.6g} | "
        f"{row['epsilon']['l2_norm_mean_per_trial']:.6g} | {row['finite']} | "
        f"{row['nonzero']} |"
    )
    if include_next:
        return base[:-1] + f" {row['next_lambda']:.6g} |"
    return base


def format_adaptive_row(row: Mapping[str, Any]) -> str:
    return (
        f"| {row['iteration']} | {row['lambda']:.6g} | {row['clean_cost']:.6g} | "
        f"{row['selected_cost']:.6g} | {row['paired_damage']:.6g} | "
        f"{row['damage_over_ref']:.5g} | {row['epsilon']['energy_mean_per_trial']:.6g} | "
        f"{row['epsilon']['l2_norm_mean_per_trial']:.6g} | {row['finite']} | "
        f"{row['nonzero']} | {row['objective_gain']:.6g} | {row['next_lambda']:.6g} |"
    )


def summary_for_stdout(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "json": repo_rel(OUTPUT_JSON),
        "markdown": repo_rel(OUTPUT_MD),
        "all_targets_pass": payload["overall_assessment"]["all_targets_pass"],
        "targets": {
            target["target_label"]: {
                "pass": target["assessment"]["pass"],
                "first_damage": target["assessment"]["first_damage"],
                "last_damage": target["assessment"]["last_damage"],
                "damage_ref": target["damage_ref"],
            }
            for target in payload["adaptive_targets"]
        },
    }


def _replicate_model(model: Any, hps: Any, replicate_index: int) -> Any:
    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
    if not 0 <= int(replicate_index) < n_replicates:
        raise ValueError(f"replicate_index must be in [0, {n_replicates}); got {replicate_index}")
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_axis_array(leaf, n_replicates),
    )
    replicate_arrays = jt.map(
        lambda leaf: None if leaf is None else leaf[int(replicate_index)],
        model_arrays,
        is_leaf=lambda leaf: leaf is None,
    )
    model_replicate = eqx.combine(replicate_arrays, model_other)
    return _with_single_replicate_state_initializers(
        model_replicate,
        n_replicates=n_replicates,
        replicate_index=int(replicate_index),
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_rel(path: Path) -> str:
    try:
        return str(Path(path).absolute().relative_to(REPO_ROOT.absolute()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
