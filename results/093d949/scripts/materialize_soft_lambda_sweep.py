"""Materialize repaired soft-lambda estimates and direct-epsilon sweeps for 093d949."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.config.namespace import TreeNamespace
from feedbax.runtime.batch import BatchInfo
from jax_cookbook import load_with_hyperparameters

from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.train import cs_nominal_gru as nominal
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    _broad_epsilon_pgd_trust_radius,
    _epsilon_time_mask,
    _ensure_broad_epsilon_input,
    _flattened_per_trial_norm,
    _set_input,
    config_from_broad_epsilon_pgd_hps,
    run_broad_epsilon_pgd_inner_maximizer,
)
from rlrmp.train.task_model import setup_task_model_pair


RUN_IDS = ("open_loop_small", "open_loop_moderate", "open_loop_stress")
SWEEP_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
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
    parser.add_argument("--issue", default="093d949")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.4)
    parser.add_argument("--pgd-steps", type=int, default=8)
    parser.add_argument("--pgd-step-size-fraction", type=float, default=0.25)
    parser.add_argument("--curvature-directions", type=int, default=8)
    parser.add_argument("--curvature-step-fraction", type=float, default=0.25)
    parser.add_argument(
        "--sweep-multipliers",
        type=float,
        nargs="+",
        default=list(SWEEP_MULTIPLIERS),
    )
    parser.add_argument(
        "--output-json",
        default="results/093d949/soft_lambda_sweep.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/093d949/soft_lambda_sweep.csv",
    )
    parser.add_argument(
        "--output-md",
        default="results/093d949/notes/soft_lambda_sweep.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = materialize(args)
    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    output_md = REPO_ROOT / args.output_md
    mkdir_p(output_json.parent)
    mkdir_p(output_csv.parent)
    mkdir_p(output_md.parent)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_sweep_csv(output_csv, payload)
    update_marked_section(output_md, "soft_lambda_sweep", render_markdown(payload))
    print(json.dumps({"json": str(output_json), "csv": str(output_csv), "markdown": str(output_md)}, indent=2))
    return 0


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for run_id in RUN_IDS:
        frozen = load_frozen_batch(args, run_id)
        estimates = estimate_lambdas(
            frozen,
            beta=float(args.beta),
            n_directions=int(args.curvature_directions),
            step_fraction=float(args.curvature_step_fraction),
        )
        center_lambda = float(estimates["per_trial_p90"]["lambda_beta"])
        sweep_rows = [
            audit_open_loop_direct_epsilon(
                frozen,
                lambda_value=center_lambda * float(multiplier),
                multiplier=float(multiplier),
                args=args,
            )
            for multiplier in args.sweep_multipliers
        ]
        rows.append(
            {
                "run_id": run_id,
                "run_spec_path": f"results/{args.experiment}/runs/{run_id}.json",
                "artifact_dir": f"_artifacts/{args.experiment}/runs/{run_id}",
                "estimates": estimates,
                "sweep_center": {
                    "source": "per_trial_p90.lambda_beta",
                    "lambda": center_lambda,
                    "multipliers": [float(value) for value in args.sweep_multipliers],
                },
                "sweep": sweep_rows,
                "transition": classify_transition(sweep_rows),
            }
        )
    return {
        "schema_version": "rlrmp.soft_lambda_sweep.v1",
        "issue": str(args.issue),
        "parent_umbrella": "54389a4",
        "source_experiment": str(args.experiment),
        "source_audit_issue": "0a46652",
        "beta": float(args.beta),
        "batch_size": int(args.batch_size),
        "replicate_index": int(args.replicate_index),
        "curvature": {
            "method": "finite_directional_curvature",
            "directions": int(args.curvature_directions),
            "step_fraction": float(args.curvature_step_fraction),
            "hvp_power_curvature": "not_used; finite-direction curvature kept this lane bounded",
        },
        "direct_sweep": {
            "objective": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            "pgd_steps": int(args.pgd_steps),
            "pgd_step_size_fraction": float(args.pgd_step_size_fraction),
            "multipliers": [float(value) for value in args.sweep_multipliers],
            "centered_on": "per_trial_p90.lambda_beta",
        },
        "frozen_contract": {
            "checkpoint": "final trained_model.eqx for each c92 open_loop no-PGD row",
            "trial_batch": "deterministic train batch from run seed and run id",
            "controller_updates": False,
            "optimized_variables": "epsilon sequence only",
            "epsilon_dim": 6,
            "safety_cap_l2_radius_15cm": CAP_RADIUS_15CM,
            "safety_cap_source": CAP_SOURCE,
            "reduction": "mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]",
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
    *,
    lambda_value: float,
    n_steps: int,
    step_size_fraction: float,
) -> TreeNamespace:
    return TreeNamespace(
        **{
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
    )


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


def estimate_lambdas(
    frozen: FrozenBatch,
    *,
    beta: float,
    n_directions: int,
    step_fraction: float,
) -> dict[str, Any]:
    old = estimate_batch_scalar_lambda(
        frozen,
        beta=beta,
        n_directions=n_directions,
        step_fraction=step_fraction,
        batch_correct_gradient=False,
    )
    batch_corrected = estimate_batch_scalar_lambda(
        frozen,
        beta=beta,
        n_directions=n_directions,
        step_fraction=step_fraction,
        batch_correct_gradient=True,
    )
    per_trial = estimate_per_trial_lambda(
        frozen,
        beta=beta,
        n_directions=n_directions,
        step_fraction=step_fraction,
    )
    return {
        "old_batch_mean_scalar": old,
        "batch_corrected_comparison": batch_corrected,
        "per_trial_p90": per_trial,
    }


def task_loss(frozen: FrozenBatch, delta: jnp.ndarray) -> jnp.ndarray:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    candidate = _set_input(frozen.trial_specs, "epsilon", epsilon + delta * frozen.time_mask)
    states = frozen.task.eval_trials(frozen.model, candidate, frozen.keys_model)
    return jnp.asarray(frozen.task.loss_func(states, candidate, frozen.model).total)


def estimate_batch_scalar_lambda(
    frozen: FrozenBatch,
    *,
    beta: float,
    n_directions: int,
    step_fraction: float,
    batch_correct_gradient: bool,
) -> dict[str, Any]:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    zero = jnp.zeros_like(epsilon)
    zero_loss, grad = jax.value_and_grad(lambda d: task_loss(frozen, d))(zero)
    grad = grad * frozen.time_mask
    grad_norm = _flattened_per_trial_norm(grad)
    batch_size = int(epsilon.shape[0])
    gradient_pressure = jnp.mean(grad_norm / jnp.maximum(2.0 * frozen.radius, 1e-12))
    if batch_correct_gradient:
        gradient_pressure = gradient_pressure * float(batch_size)
    curvatures = batch_directional_curvatures(
        frozen,
        zero_loss=zero_loss,
        n_directions=n_directions,
        step_fraction=step_fraction,
    )
    max_curvature = jnp.max(curvatures)
    curvature_lambda = jnp.maximum(0.0, 0.5 * max_curvature)
    lambda_star = jnp.maximum(curvature_lambda, gradient_pressure)
    return {
        "method": (
            "batch_mean_scalar_finite_directional_curvature_with_gradient_pressure_floor"
            if not batch_correct_gradient
            else "batch_mean_scalar_with_B_corrected_gradient_pressure_floor"
        ),
        "zero_loss": float(zero_loss),
        "batch_size": batch_size,
        "gradient_norm_mean": float(jnp.mean(grad_norm)),
        "gradient_pressure_lambda": float(gradient_pressure),
        "directional_curvature_lambda": float(curvature_lambda),
        "directional_curvature_max": float(max_curvature),
        "directional_curvature_mean": float(jnp.mean(curvatures)),
        "lambda_star": float(lambda_star),
        "beta": float(beta),
        "lambda_beta": float((float(beta) ** 2) * lambda_star),
    }


def batch_directional_curvatures(
    frozen: FrozenBatch,
    *,
    zero_loss: jnp.ndarray,
    n_directions: int,
    step_fraction: float,
) -> jnp.ndarray:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    key = jr.PRNGKey(911)
    directions = jr.normal(key, (int(n_directions), *epsilon.shape), dtype=epsilon.dtype)
    directions = directions * frozen.time_mask
    norm_shape = (int(n_directions), -1, 1, 1)
    directions = directions / jnp.maximum(
        _flattened_per_trial_norm(directions).reshape(norm_shape),
        1e-12,
    )
    step = jnp.mean(frozen.radius) * float(step_fraction)

    def curvature(direction: jnp.ndarray) -> jnp.ndarray:
        plus = task_loss(frozen, direction * step)
        minus = task_loss(frozen, -direction * step)
        return (plus + minus - 2.0 * zero_loss) / jnp.maximum(step**2, 1e-20)

    return jax.vmap(curvature)(directions)


def estimate_per_trial_lambda(
    frozen: FrozenBatch,
    *,
    beta: float,
    n_directions: int,
    step_fraction: float,
) -> dict[str, Any]:
    candidates = [
        estimate_single_trial_lambda(
            slice_frozen_batch(frozen, trial_index),
            beta=beta,
            n_directions=n_directions,
            step_fraction=step_fraction,
            key=jr.fold_in(jr.PRNGKey(1907), trial_index),
            trial_index=trial_index,
        )
        for trial_index in range(int(frozen.radius.shape[0]))
    ]
    lambda_star_values = np.asarray([row["lambda_star"] for row in candidates], dtype=float)
    lambda_beta_values = np.asarray([row["lambda_beta"] for row in candidates], dtype=float)
    gradient_values = np.asarray([row["gradient_pressure_lambda"] for row in candidates], dtype=float)
    curvature_values = np.asarray([row["directional_curvature_lambda"] for row in candidates], dtype=float)
    finite = np.isfinite(lambda_beta_values)
    return {
        "method": "per_trial_finite_directional_curvature_with_gradient_pressure_p90",
        "aggregation": "p90_across_trial_level_lambda_candidates",
        "n_trials": len(candidates),
        "trial_candidates": candidates,
        "gradient_pressure_lambda_p90": percentile_or_nan(gradient_values, 90.0),
        "directional_curvature_lambda_p90": percentile_or_nan(curvature_values, 90.0),
        "lambda_star": percentile_or_nan(lambda_star_values, 90.0),
        "beta": float(beta),
        "lambda_beta": percentile_or_nan(lambda_beta_values, 90.0),
        "lambda_beta_median": percentile_or_nan(lambda_beta_values, 50.0),
        "finite_fraction": float(np.mean(finite)) if len(finite) else 0.0,
    }


def estimate_single_trial_lambda(
    frozen: FrozenBatch,
    *,
    beta: float,
    n_directions: int,
    step_fraction: float,
    key: jax.Array,
    trial_index: int,
) -> dict[str, Any]:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    zero = jnp.zeros_like(epsilon)
    zero_loss, grad = jax.value_and_grad(lambda d: task_loss(frozen, d))(zero)
    grad = grad * frozen.time_mask
    grad_norm = _flattened_per_trial_norm(grad)
    radius = jnp.asarray(frozen.radius)[0]
    gradient_pressure = grad_norm[0] / jnp.maximum(2.0 * radius, 1e-12)
    directions = jr.normal(key, (int(n_directions), *epsilon.shape), dtype=epsilon.dtype)
    directions = directions * frozen.time_mask
    directions = directions / jnp.maximum(
        _flattened_per_trial_norm(directions).reshape((int(n_directions), -1, 1, 1)),
        1e-12,
    )
    step = radius * float(step_fraction)

    def curvature(direction: jnp.ndarray) -> jnp.ndarray:
        plus = task_loss(frozen, direction * step)
        minus = task_loss(frozen, -direction * step)
        return (plus + minus - 2.0 * zero_loss) / jnp.maximum(step**2, 1e-20)

    curvatures = jax.vmap(curvature)(directions)
    max_curvature = jnp.max(curvatures)
    curvature_lambda = jnp.maximum(0.0, 0.5 * max_curvature)
    lambda_star = jnp.maximum(curvature_lambda, gradient_pressure)
    return {
        "trial_index": int(trial_index),
        "zero_loss": float(zero_loss),
        "radius": float(radius),
        "gradient_norm": float(grad_norm[0]),
        "gradient_pressure_lambda": float(gradient_pressure),
        "directional_curvature_lambda": float(curvature_lambda),
        "directional_curvature_max": float(max_curvature),
        "directional_curvature_mean": float(jnp.mean(curvatures)),
        "lambda_star": float(lambda_star),
        "beta": float(beta),
        "lambda_beta": float((float(beta) ** 2) * lambda_star),
    }


def slice_frozen_batch(frozen: FrozenBatch, trial_index: int) -> FrozenBatch:
    def take_trial(leaf: Any) -> Any:
        if eqx.is_array(leaf) and leaf.ndim > 0 and int(leaf.shape[0]) == int(frozen.radius.shape[0]):
            return leaf[trial_index : trial_index + 1]
        return leaf

    return FrozenBatch(
        task=frozen.task,
        model=frozen.model,
        trial_specs=jt.map(take_trial, frozen.trial_specs),
        keys_model=frozen.keys_model[trial_index : trial_index + 1],
        hps=frozen.hps,
        run_spec=frozen.run_spec,
        radius=frozen.radius[trial_index : trial_index + 1],
        time_mask=frozen.time_mask[trial_index : trial_index + 1],
    )


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def audit_open_loop_direct_epsilon(
    frozen: FrozenBatch,
    *,
    lambda_value: float,
    multiplier: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = soft_pgd_config(
        lambda_value=lambda_value,
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
        lambda_value=lambda_value,
        multiplier=multiplier,
        delta=delta,
        radius=frozen.radius,
        diagnostics={key: np.asarray(jax.device_get(value)) for key, value in diagnostics.items()},
    )


def per_trial_energy(delta: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(delta), axis=tuple(range(1, delta.ndim)))


def audit_summary(
    *,
    lambda_value: float,
    multiplier: float,
    delta: jnp.ndarray,
    radius: jnp.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    delta = jnp.asarray(delta)
    norm = _flattened_per_trial_norm(delta)
    energy = per_trial_energy(delta)
    ratio = norm / jnp.maximum(radius, 1e-12)
    raw_loss_gain = scalar_diagnostic(diagnostics, "raw_task_loss_selected") - scalar_diagnostic(
        diagnostics,
        "raw_task_loss_zero",
    )
    energy_penalty = scalar_diagnostic(diagnostics, "energy_penalty_term_selected")
    penalized_gain = scalar_diagnostic(diagnostics, "selected_objective_gain_over_zero")
    nonfinite_seen = bool(np.asarray(diagnostics.get("inner_objective_nonfinite_seen", False)))
    finite_values = np.isfinite([raw_loss_gain, energy_penalty, penalized_gain, float(lambda_value)])
    finite_status = "finite" if bool(np.all(finite_values)) and not nonfinite_seen else "nonfinite"
    cap_fraction = float(jnp.mean((ratio >= 1.0 - 1e-4).astype(jnp.float32)))
    return {
        "lambda": float(lambda_value),
        "multiplier": float(multiplier),
        "selected_epsilon_norm_mean": float(jnp.mean(norm)),
        "selected_epsilon_norm_max": float(jnp.max(norm)),
        "radius_mean": float(jnp.mean(radius)),
        "radius_max": float(jnp.max(radius)),
        "selected_norm_cap_ratio_mean": float(jnp.mean(ratio)),
        "selected_norm_cap_ratio_max": float(jnp.max(ratio)),
        "cap_bound_fraction": cap_fraction,
        "raw_loss_gain": float(raw_loss_gain),
        "energy_mean": float(jnp.mean(energy)),
        "energy_penalty": float(energy_penalty),
        "penalized_gain_over_zero": float(penalized_gain),
        "finite_status": finite_status,
        "direct_objective_reduction": "mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]",
        "diagnostics": plain_json(diagnostics),
    }


def scalar_diagnostic(diagnostics: dict[str, Any], key: str) -> float:
    return float(np.asarray(diagnostics.get(key, np.nan)))


def classify_transition(sweep_rows: list[dict[str, Any]]) -> dict[str, Any]:
    interior = [row for row in sweep_rows if row["cap_bound_fraction"] < 0.05]
    capped = [row for row in sweep_rows if row["cap_bound_fraction"] >= 0.95]
    center = min(sweep_rows, key=lambda row: abs(row["multiplier"] - 1.0))
    first_interior = min(interior, key=lambda row: row["lambda"]) if interior else None
    last_capped = max(capped, key=lambda row: row["lambda"]) if capped else None
    predicted = bool(first_interior and last_capped and last_capped["lambda"] < first_interior["lambda"])
    center_cap_fraction = float(center["cap_bound_fraction"])
    if center_cap_fraction >= 0.95:
        center_side = "cap_dominated"
    elif center_cap_fraction < 0.05:
        center_side = "interior"
    else:
        center_side = "intermediate"
    return {
        "has_interior_row": bool(interior),
        "has_cap_dominated_row": bool(capped),
        "transition_bracketed": predicted,
        "estimate_center_multiplier": float(center["multiplier"]),
        "estimate_center_lambda": float(center["lambda"]),
        "estimate_center_cap_bound_fraction": center_cap_fraction,
        "estimate_center_norm_cap_ratio_max": float(center["selected_norm_cap_ratio_max"]),
        "estimate_center_side": center_side,
        "last_cap_dominated_multiplier": None if last_capped is None else last_capped["multiplier"],
        "first_interior_multiplier": None if first_interior is None else first_interior["multiplier"],
        "interpretation": (
            "p90 estimate lands cap-dominated, but the narrow grid brackets the transition above it"
            if predicted and center_side == "cap_dominated"
            else "p90 estimate lands interior, and the narrow grid brackets the transition below it"
            if predicted and center_side == "interior"
            else "p90 estimate lands in the transition band"
            if predicted
            else "sweep does not bracket a clean cap-to-interior transition"
        ),
        "bracket_interpretation": (
            "sweep brackets a cap-to-interior transition"
            if predicted
            else "sweep does not bracket a clean cap-to-interior transition"
        ),
    }


def plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain_json(v) for k, v in value.items()}
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, (np.bool_, bool)):
            return bool(scalar)
        if isinstance(scalar, (np.integer, int)):
            return int(scalar)
        return float(scalar)
    return array.tolist()


def write_sweep_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "run_id",
        "lambda",
        "multiplier",
        "selected_epsilon_norm_mean",
        "selected_epsilon_norm_max",
        "radius_mean",
        "radius_max",
        "selected_norm_cap_ratio_mean",
        "selected_norm_cap_ratio_max",
        "cap_bound_fraction",
        "raw_loss_gain",
        "energy_mean",
        "energy_penalty",
        "penalized_gain_over_zero",
        "finite_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in payload["rows"]:
            for sweep in row["sweep"]:
                writer.writerow({"run_id": row["run_id"], **{key: sweep[key] for key in fieldnames[1:]}})


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Soft lambda estimator and direct-epsilon sweep",
        "",
        f"Issue: `{payload['issue']}`. Source no-PGD runs: `c92ebd8`.",
        "",
        "No controller weights were updated. This is a frozen-batch audit of existing c92 "
        "open-loop no-PGD substrates.",
        "",
        "## Lambda estimates",
        "",
        "| row | old beta lambda | B-corrected beta lambda | per-trial p90 beta lambda | old grad floor | B-corrected grad floor | per-trial p90 grad floor | per-trial p90 curvature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        estimates = row["estimates"]
        old = estimates["old_batch_mean_scalar"]
        corrected = estimates["batch_corrected_comparison"]
        per_trial = estimates["per_trial_p90"]
        lines.append(
            f"| `{row['run_id']}` | {old['lambda_beta']:.6g} | "
            f"{corrected['lambda_beta']:.6g} | {per_trial['lambda_beta']:.6g} | "
            f"{old['gradient_pressure_lambda']:.6g} | "
            f"{corrected['gradient_pressure_lambda']:.6g} | "
            f"{per_trial['gradient_pressure_lambda_p90']:.6g} | "
            f"{per_trial['directional_curvature_lambda_p90']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Direct-epsilon sweep",
            "",
            "The sweep is centered on `per_trial_p90.lambda_beta`; the batch-corrected value is "
            "reported only as a comparison.",
            "",
            "| row | multiplier | lambda | norm/cap max | cap-bound | raw loss gain | energy penalty | penalized gain | finite |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        for sweep in row["sweep"]:
            lines.append(
                f"| `{row['run_id']}` | {sweep['multiplier']:.3g} | "
                f"{sweep['lambda']:.6g} | {sweep['selected_norm_cap_ratio_max']:.6g} | "
                f"{sweep['cap_bound_fraction']:.3%} | {sweep['raw_loss_gain']:.6g} | "
                f"{sweep['energy_penalty']:.6g} | {sweep['penalized_gain_over_zero']:.6g} | "
                f"{sweep['finite_status']} |"
            )
    lines.extend(["", "## Transition read", ""])
    for row in payload["rows"]:
        transition = row["transition"]
        lines.append(
            f"- `{row['run_id']}`: {transition['interpretation']}; "
            f"center cap-bound = `{transition['estimate_center_cap_bound_fraction']:.3%}`, "
            f"last cap-dominated multiplier = `{transition['last_cap_dominated_multiplier']}`, "
            f"first interior multiplier = `{transition['first_interior_multiplier']}`."
        )
    lines.extend(
        [
            "",
            "## Runtime caveats",
            "",
            "- Curvature uses bounded finite directions, not HVP/power iteration. HVP remains a later "
            "estimator-strengthening option rather than a separate adversary mechanism.",
            "- The direct soft-energy objective reduction remains the existing "
            "`mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]` code path.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
