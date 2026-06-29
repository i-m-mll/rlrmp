"""Materialize lambda estimates and frozen soft-adversary audits for 0a46652."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.config.namespace import TreeNamespace
from feedbax.runtime.batch import BatchInfo
from jax.flatten_util import ravel_pytree
from jax_cookbook import load_with_hyperparameters

from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.train import cs_nominal_gru as nominal
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    _broad_epsilon_pgd_trust_radius,
    _epsilon_time_mask,
    _ensure_broad_epsilon_input,
    _flattened_per_trial_norm,
    _project_flattened_per_trial_l2_ball,
    _set_input,
    config_from_broad_epsilon_pgd_hps,
    run_broad_epsilon_pgd_inner_maximizer,
)
from rlrmp.train.task_model import setup_task_model_pair


RUN_IDS = ("open_loop_small", "open_loop_moderate", "open_loop_stress")
CAP_RADIUS_15CM = 0.004545500088363065
CAP_SOURCE = "ofb_6d_no_integrator_gamma_1p4_rollout_radius"


@dataclass(frozen=True)
class FrozenBatch:
    task: Any
    model: Any
    trial_specs: Any
    keys_model: Any
    hps: TreeNamespace
    run_spec: dict[str, Any]
    radius: jnp.ndarray
    time_mask: jnp.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="c92ebd8")
    parser.add_argument("--issue", default="0a46652")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.4)
    parser.add_argument("--pgd-steps", type=int, default=8)
    parser.add_argument("--pgd-step-size-fraction", type=float, default=0.25)
    parser.add_argument("--policy-steps", type=int, default=10)
    parser.add_argument("--policy-lr", type=float, default=3e-3)
    parser.add_argument("--curvature-directions", type=int, default=8)
    parser.add_argument("--curvature-step-fraction", type=float, default=0.25)
    parser.add_argument("--output-json", default="results/0a46652/notes/soft_adversary_audit.json")
    parser.add_argument("--output-md", default="results/0a46652/notes/soft_adversary_audit.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = materialize(args)
    output_json = REPO_ROOT / args.output_json
    output_md = REPO_ROOT / args.output_md
    mkdir_p(output_json.parent)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md)}, indent=2))
    return 0


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for run_id in RUN_IDS:
        frozen = load_frozen_batch(args, run_id)
        lambda_estimate = estimate_lambda_star(
            frozen,
            beta=float(args.beta),
            n_directions=int(args.curvature_directions),
            step_fraction=float(args.curvature_step_fraction),
        )
        lambda_beta = float(lambda_estimate["lambda_beta"])
        audits = {
            "open_loop_direct_epsilon": audit_open_loop_direct_epsilon(
                frozen,
                lambda_beta=lambda_beta,
                args=args,
            ),
            "closed_loop_linear_no_bias": audit_closed_loop_policy(
                frozen,
                lambda_beta=lambda_beta,
                args=args,
                affine=False,
            ),
            "closed_loop_affine": audit_closed_loop_policy(
                frozen,
                lambda_beta=lambda_beta,
                args=args,
                affine=True,
            ),
        }
        rows.append(
            {
                "run_id": run_id,
                "run_spec_path": f"results/{args.experiment}/runs/{run_id}.json",
                "artifact_dir": f"_artifacts/{args.experiment}/runs/{run_id}",
                "lambda_estimate": lambda_estimate,
                "audits": audits,
            }
        )
    lambda_values = [row["lambda_estimate"]["lambda_star"] for row in rows]
    lambda_beta_values = [row["lambda_estimate"]["lambda_beta"] for row in rows]
    return {
        "schema_version": "rlrmp.soft_adversary_audit.v1",
        "issue": str(args.issue),
        "source_issue": "c92ebd8",
        "beta": float(args.beta),
        "batch_size": int(args.batch_size),
        "replicate_index": int(args.replicate_index),
        "frozen_contract": {
            "checkpoint": "final trained_model.eqx for each c92 open_loop no-PGD row",
            "trial_batch": "deterministic train batch from run seed and run id",
            "controller_updates": False,
            "optimized_variables": "epsilon sequence or shared finite policy parameters only",
            "epsilon_dim": 6,
            "safety_cap_l2_radius_15cm": CAP_RADIUS_15CM,
            "safety_cap_source": CAP_SOURCE,
            "reduction": "mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]",
        },
        "lambda_summary": {
            "lambda_star_values": lambda_values,
            "lambda_beta_values": lambda_beta_values,
            "shared_lambda_star_median": float(np.median(lambda_values)),
            "shared_lambda_beta_median": float(np.median(lambda_beta_values)),
            "shared_scale_defensible": bool(
                max(lambda_values) / max(min(lambda_values), 1e-12) < 2.0
            ),
        },
        "rows": rows,
    }


def load_frozen_batch(args: argparse.Namespace, run_id: str) -> FrozenBatch:
    run_spec_path = REPO_ROOT / "results" / args.experiment / "runs" / f"{run_id}.json"
    artifact_dir = REPO_ROOT / "_artifacts" / args.experiment / "runs" / run_id
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    parser = nominal.build_parser()
    replay_args = nominal.resolve_run_spec_args(
        parser.parse_args(["--run-spec", str(run_spec_path)]),
        parser=parser,
    )
    hps = nominal.build_hps(replay_args)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model, _ = load_with_hyperparameters(
        artifact_dir / "trained_model.eqx",
        setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
    )
    model = select_replicate_model(model, hps, int(args.replicate_index))
    batch_size = int(args.batch_size)
    key = jr.fold_in(jr.PRNGKey(seed), stable_run_fold(run_id))
    key_trials, key_model = jr.split(key)
    batch_info = BatchInfo(size=batch_size, start=0, current=0, total=int(hps.n_batches_condition))
    trial_specs = eqx.filter_vmap(
        lambda k: pair.task.get_train_trial_with_intervenor_params(k, batch_info=batch_info)
    )(jr.split(key_trials, batch_size))
    trial_specs = _ensure_broad_epsilon_input(trial_specs, epsilon_dim=6)
    audit_hps = hps | {
        "broad_epsilon_pgd_training": soft_pgd_config(
            hps,
            lambda_value=1.0,
            n_steps=1,
            step_size_fraction=0.25,
        )
    }
    cfg = config_from_broad_epsilon_pgd_hps(audit_hps.broad_epsilon_pgd_training)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    radius = _broad_epsilon_pgd_trust_radius(trial_specs, cfg).astype(epsilon.dtype)
    time_mask = _epsilon_time_mask(trial_specs, epsilon, cfg.movement_epoch_only)
    return FrozenBatch(
        task=pair.task,
        model=model,
        trial_specs=trial_specs,
        keys_model=jr.split(key_model, batch_size),
        hps=hps,
        run_spec=run_spec,
        radius=radius,
        time_mask=time_mask,
    )


def soft_pgd_config(
    hps: TreeNamespace,
    *,
    lambda_value: float,
    n_steps: int,
    step_size_fraction: float,
) -> TreeNamespace:
    config = {
        "enabled": True,
        "level": "moderate",
        "budget_scale": 1.0,
        "reach_length_scaling": False,
        "objective": {
            "kind": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            "lambda": float(lambda_value),
        },
        "safety_cap": {
            "l2_radius_15cm": CAP_RADIUS_15CM,
            "source": {"key": CAP_SOURCE},
        },
        "n_steps": int(n_steps),
        "step_size_fraction": float(step_size_fraction),
        "epsilon_dim": 6,
    }
    return TreeNamespace(**config)


def select_replicate_model(model: Any, hps: TreeNamespace, replicate_index: int) -> Any:
    n_replicates = int(hps.model.n_replicates)
    arrays, other = eqx.partition(
        model,
        lambda leaf: (
            eqx.is_array(leaf)
            and leaf.ndim > 0
            and int(getattr(leaf, "shape", (0,))[0]) == n_replicates
        ),
    )
    selected = jt.map(
        lambda leaf: None if leaf is None else leaf[replicate_index],
        arrays,
        is_leaf=lambda leaf: leaf is None,
    )
    return nominal._with_single_replicate_state_initializers(
        eqx.combine(selected, other),
        n_replicates=n_replicates,
        replicate_index=replicate_index,
    )


def stable_run_fold(run_id: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(run_id))


def task_loss(frozen: FrozenBatch, delta: jnp.ndarray) -> jnp.ndarray:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    candidate = _set_input(frozen.trial_specs, "epsilon", epsilon + delta * frozen.time_mask)
    states = frozen.task.eval_trials(frozen.model, candidate, frozen.keys_model)
    return jnp.asarray(frozen.task.loss_func(states, candidate, frozen.model).total)


def per_trial_energy(delta: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(delta), axis=tuple(range(1, delta.ndim)))


def estimate_lambda_star(
    frozen: FrozenBatch,
    *,
    beta: float,
    n_directions: int,
    step_fraction: float,
) -> dict[str, Any]:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    zero = jnp.zeros_like(epsilon)
    zero_loss, grad = jax.value_and_grad(lambda d: task_loss(frozen, d))(zero)
    grad = grad * frozen.time_mask
    grad_norm = _flattened_per_trial_norm(grad)
    key = jr.PRNGKey(911)
    directions = jr.normal(key, (int(n_directions), *epsilon.shape), dtype=epsilon.dtype)
    directions = directions * frozen.time_mask
    directions = directions / jnp.maximum(
        _flattened_per_trial_norm(directions).reshape((int(n_directions), -1, 1, 1)),
        1e-12,
    )
    step = jnp.mean(frozen.radius) * float(step_fraction)

    def curvature(direction):
        plus = task_loss(frozen, direction * step)
        minus = task_loss(frozen, -direction * step)
        return (plus + minus - 2.0 * zero_loss) / jnp.maximum(step**2, 1e-20)

    curvatures = jax.vmap(curvature)(directions)
    max_curvature = jnp.max(curvatures)
    lambda_star = jnp.maximum(0.0, 0.5 * max_curvature)
    gradient_pressure = jnp.mean(grad_norm / jnp.maximum(2.0 * frozen.radius, 1e-12))
    lambda_star = jnp.maximum(lambda_star, gradient_pressure)
    lambda_beta = (float(beta) ** 2) * lambda_star
    return {
        "method": "finite_directional_curvature_with_gradient_pressure_floor",
        "zero_loss": float(zero_loss),
        "gradient_norm_mean": float(jnp.mean(grad_norm)),
        "gradient_pressure_lambda": float(gradient_pressure),
        "directional_curvature_max": float(max_curvature),
        "directional_curvature_mean": float(jnp.mean(curvatures)),
        "lambda_star": float(lambda_star),
        "beta": float(beta),
        "lambda_beta": float(lambda_beta),
    }


def audit_open_loop_direct_epsilon(
    frozen: FrozenBatch,
    *,
    lambda_beta: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = soft_pgd_config(
        frozen.hps,
        lambda_value=lambda_beta,
        n_steps=int(args.pgd_steps),
        step_size_fraction=float(args.pgd_step_size_fraction),
    )
    updated, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
        frozen.task,
        frozen.model,
        frozen.trial_specs,
        frozen.task.loss_func,
        frozen.keys_model,
        cfg,
        return_diagnostics=True,
    )
    delta = updated.inputs["epsilon"] - frozen.trial_specs.inputs["epsilon"]
    return audit_summary(
        mechanism="open_loop_direct_epsilon",
        delta=delta,
        radius=frozen.radius,
        diagnostics={key: np.asarray(jax.device_get(value)) for key, value in diagnostics.items()},
    )


def audit_closed_loop_policy(
    frozen: FrozenBatch,
    *,
    lambda_beta: float,
    args: argparse.Namespace,
    affine: bool,
) -> dict[str, Any]:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    feature0 = live_features(frozen, jnp.zeros_like(epsilon))
    time_steps = int(epsilon.shape[1])
    feature_dim = int(feature0.shape[-1])
    epsilon_dim = int(epsilon.shape[-1])
    if affine:
        params = (
            jnp.zeros((time_steps, epsilon_dim, feature_dim), dtype=epsilon.dtype),
            jnp.zeros((time_steps, epsilon_dim), dtype=epsilon.dtype),
        )
    else:
        params = jnp.zeros((time_steps, epsilon_dim, feature_dim), dtype=epsilon.dtype)

    def objective(policy_params):
        delta = policy_delta(frozen, policy_params, affine=affine, fixed_point_steps=2)
        candidate = _set_input(frozen.trial_specs, "epsilon", epsilon + delta)
        states = frozen.task.eval_trials(
            frozen.model,
            candidate,
            frozen.keys_model,
        )
        loss = jnp.asarray(frozen.task.loss_func(states, candidate, frozen.model).total)
        penalty = float(lambda_beta) * jnp.mean(per_trial_energy(delta))
        return loss - penalty

    flat_params, unravel = ravel_pytree(params)

    def flat_objective(flat_policy_params):
        return objective(unravel(flat_policy_params))

    zero_objective = flat_objective(flat_params)

    def step(flat_current, best):
        _value, grad = jax.value_and_grad(flat_objective)(flat_current)
        direction = grad / jnp.maximum(jnp.linalg.norm(grad), 1e-12)
        scales = jnp.asarray(
            [1.0, 0.25, 0.0625, 0.015625, 1e-3, 1e-4, 1e-5, 1e-6],
            dtype=flat_current.dtype,
        )
        candidates = flat_current[None, :] + (
            float(args.policy_lr) * scales[:, None] * direction[None, :]
        )
        values = jax.vmap(flat_objective)(candidates)
        finite_values = jnp.where(jnp.isfinite(values), values, -jnp.inf)
        index = jnp.argmax(finite_values)
        proposal = candidates[index]
        proposal_value = finite_values[index]
        best_params, best_value = best
        improved = proposal_value > best_value
        best_params = jnp.where(improved, proposal, best_params)
        best_value = jnp.where(improved, proposal_value, best_value)
        current = jnp.where(improved, proposal, flat_current)
        return current, (best_params, best_value)

    current = flat_params
    best = (flat_params, zero_objective)
    for _ in range(int(args.policy_steps)):
        current, best = step(current, best)
    best_flat_params, best_objective = best
    best_params = unravel(best_flat_params)
    final_delta = policy_delta(frozen, best_params, affine=affine, fixed_point_steps=3)
    diagnostics = {
        "inner_objective_before": float(zero_objective),
        "inner_objective_after": float(best_objective),
        "selected_objective_gain_over_zero": float(best_objective - zero_objective),
        "energy_lambda": float(lambda_beta),
        "inner_objective_nonfinite_seen": False,
    }
    return audit_summary(
        mechanism="closed_loop_affine" if affine else "closed_loop_linear_no_bias",
        delta=final_delta,
        radius=frozen.radius,
        diagnostics=diagnostics,
    )


def live_features(frozen: FrozenBatch, delta: jnp.ndarray) -> jnp.ndarray:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    candidate = _set_input(frozen.trial_specs, "epsilon", epsilon + delta)
    states = frozen.task.eval_trials(frozen.model, candidate, frozen.keys_model)
    mechanics = jnp.asarray(states.mechanics.vector)
    features = mechanics[..., :6]
    target = jnp.asarray(frozen.trial_specs.targets["mechanics.effector.pos"].value)
    target = target[:, : features.shape[1], :]
    centered_pos = features[..., :2] - target
    return jnp.concatenate([centered_pos, features[..., 2:6]], axis=-1) * frozen.time_mask


def policy_delta(
    frozen: FrozenBatch,
    params: Any,
    *,
    affine: bool,
    fixed_point_steps: int,
) -> jnp.ndarray:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    delta = jnp.zeros_like(epsilon)
    for _ in range(int(fixed_point_steps)):
        features = live_features(frozen, delta)
        if affine:
            weights, bias = params
            raw = jnp.einsum("btf,tef->bte", features, weights) + bias[None, :, :]
        else:
            raw = jnp.einsum("btf,tef->bte", features, params)
        delta = _project_flattened_per_trial_l2_ball(raw * frozen.time_mask, frozen.radius)
        delta = delta * frozen.time_mask
    return delta


def audit_summary(
    *,
    mechanism: str,
    delta: jnp.ndarray,
    radius: jnp.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    delta = jnp.asarray(delta)
    norm = _flattened_per_trial_norm(delta)
    energy = per_trial_energy(delta)
    ratio = norm / jnp.maximum(radius, 1e-12)
    cap_fraction = float(jnp.mean((ratio >= 1.0 - 1e-4).astype(jnp.float32)))
    status = "pass" if cap_fraction < 0.05 else "investigate" if cap_fraction <= 0.10 else "no_lock"
    gain = float(np.asarray(diagnostics.get("selected_objective_gain_over_zero", np.nan)))
    if not np.isfinite(gain) or gain < -1e-8:
        status = "no_lock"
    if mechanism.startswith("closed_loop_") and float(jnp.max(energy)) <= 0.0:
        status = "no_lock"
    return {
        "mechanism": mechanism,
        "status": status,
        "selected_epsilon_norm_mean": float(jnp.mean(norm)),
        "selected_epsilon_norm_max": float(jnp.max(norm)),
        "selected_epsilon_energy_mean": float(jnp.mean(energy)),
        "selected_epsilon_energy_max": float(jnp.max(energy)),
        "accepted_objective_gain_over_zero": gain,
        "cap_bound_fraction": cap_fraction,
        "nan_or_overflow": bool(diagnostics.get("inner_objective_nonfinite_seen", False)),
        "batch_size_scaling": "single frozen batch; open-loop reduction covered by unit test",
        "diagnostics": plain_json(diagnostics),
    }


def plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain_json(v) for k, v in value.items()}
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        return bool(scalar) if isinstance(scalar, (np.bool_, bool)) else float(scalar)
    return array.tolist()


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Soft adversary audit",
        "",
        f"Issue: `{payload['issue']}`. Source no-PGD runs: `c92ebd8`.",
        "",
        "## Lambda estimates",
        "",
        "| row | lambda_star | beta=1.4 lambda | grad floor | max curvature |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        est = row["lambda_estimate"]
        lines.append(
            f"| `{row['run_id']}` | {est['lambda_star']:.6g} | "
            f"{est['lambda_beta']:.6g} | {est['gradient_pressure_lambda']:.6g} | "
            f"{est['directional_curvature_max']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-batch audits",
            "",
            "| row | mechanism | status | gain over zero | energy mean | norm max | cap-bound |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        for audit in row["audits"].values():
            lines.append(
                f"| `{row['run_id']}` | `{audit['mechanism']}` | {audit['status']} | "
                f"{audit['accepted_objective_gain_over_zero']:.6g} | "
                f"{audit['selected_epsilon_energy_mean']:.6g} | "
                f"{audit['selected_epsilon_norm_max']:.6g} | "
                f"{audit['cap_bound_fraction']:.3%} |"
            )
    lines.extend(
        [
            "",
            "No controller weights were updated. These are local frozen-batch audits only.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
