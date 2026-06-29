"""Materialize raw-vs-clipped closed-loop policy audits for 3b850d6."""

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
from rlrmp.paths import mkdir_p
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
DEFAULT_AMPLITUDES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_LAMBDA_MULTIPLIERS = (2.0, 4.0)
CAP_RADIUS_15CM = 0.004545500088363065
CAP_SOURCE = "ofb_6d_no_integrator_gamma_1p4_rollout_radius"
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_LAMBDA_SWEEP = REPO_ROOT / "results" / "093d949" / "soft_lambda_sweep.json"


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


@dataclass(frozen=True)
class PolicyDirection:
    label: str
    mechanism: str
    params: Any
    fit_diagnostics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="c92ebd8")
    parser.add_argument("--issue", default="3b850d6")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--pgd-steps", type=int, default=8)
    parser.add_argument("--pgd-step-size-fraction", type=float, default=0.25)
    parser.add_argument("--fixed-point-steps", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument(
        "--lambda-multipliers",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDA_MULTIPLIERS),
    )
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=list(DEFAULT_AMPLITUDES),
    )
    parser.add_argument(
        "--output-json",
        default="results/3b850d6/closed_loop_policy_audit.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/3b850d6/closed_loop_policy_audit.csv",
    )
    parser.add_argument(
        "--output-md",
        default="results/3b850d6/notes/closed_loop_policy_audit.md",
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
    write_csv(output_csv, payload)
    update_marked_section(output_md, "closed_loop_policy_audit", render_markdown(payload))
    print(
        json.dumps(
            {"json": str(output_json), "csv": str(output_csv), "markdown": str(output_md)},
            indent=2,
        )
    )
    return 0


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    lambda_source = load_lambda_source(SOURCE_LAMBDA_SWEEP)
    rows = []
    for run_id in RUN_IDS:
        frozen = load_frozen_batch(args, run_id)
        center_lambda = float(lambda_source[run_id]["center_lambda"])
        transition_multiplier = float(lambda_source[run_id]["transition_multiplier"])
        multipliers = sorted(set(float(value) for value in args.lambda_multipliers) | {transition_multiplier})
        lambda_rows = []
        for multiplier in multipliers:
            lambda_value = center_lambda * multiplier
            direct_delta, direct_diagnostics = direct_epsilon_direction(
                frozen,
                lambda_value=lambda_value,
                args=args,
            )
            directions = build_policy_directions(
                frozen,
                direct_delta=direct_delta,
                ridge_alpha=float(args.ridge_alpha),
            )
            audits = [
                evaluate_policy_direction(
                    frozen,
                    direction=direction,
                    lambda_value=lambda_value,
                    multiplier=multiplier,
                    amplitude=float(amplitude),
                    fixed_point_steps=int(args.fixed_point_steps),
                )
                for direction in directions
                for amplitude in args.amplitudes
            ]
            lambda_rows.append(
                {
                    "lambda": float(lambda_value),
                    "multiplier": float(multiplier),
                    "lambda_source": lambda_source[run_id],
                    "direct_epsilon": direct_summary(
                        direct_delta,
                        frozen.radius,
                        diagnostics=direct_diagnostics,
                    ),
                    "directions": [direction_summary(direction) for direction in directions],
                    "audits": audits,
                    "best_by_direction": best_rows_by_direction(audits),
                }
            )
        rows.append(
            {
                "run_id": run_id,
                "run_spec_path": f"results/{args.experiment}/runs/{run_id}.json",
                "artifact_dir": f"_artifacts/{args.experiment}/runs/{run_id}",
                "lambda_rows": lambda_rows,
                "interpretation": interpret_run(lambda_rows),
            }
        )
    return {
        "schema_version": "rlrmp.closed_loop_policy_audit.v1",
        "issue": str(args.issue),
        "parent_umbrella": "54389a4",
        "source_experiment": str(args.experiment),
        "source_direct_lambda_issue": "093d949",
        "source_closed_loop_issue": "0a46652",
        "batch_size": int(args.batch_size),
        "replicate_index": int(args.replicate_index),
        "policy_contract": {
            "controller_updates": False,
            "optimized_variables": "none; scalar line search over known policy directions",
            "raw_policy_objective": "J(raw_epsilon) - lambda * E(raw_epsilon)",
            "selected_policy_diagnostic": "per-trial L2 projection used only after raw output for cap diagnostics",
            "lambda_multipliers": [float(value) for value in args.lambda_multipliers],
            "amplitudes": [float(value) for value in args.amplitudes],
            "epsilon_dim": 6,
            "safety_cap_l2_radius_15cm": CAP_RADIUS_15CM,
            "safety_cap_source": CAP_SOURCE,
        },
        "rows": rows,
        "overall_interpretation": interpret_overall(rows),
    }


def load_lambda_source(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = {}
    for row in payload["rows"]:
        transition = row["transition"]
        transition_multiplier = transition["first_interior_multiplier"]
        if transition_multiplier is None:
            transition_multiplier = transition["estimate_center_multiplier"]
        source[row["run_id"]] = {
            "center_lambda": float(row["sweep_center"]["lambda"]),
            "center_source": row["sweep_center"]["source"],
            "transition_multiplier": float(transition_multiplier),
            "transition_interpretation": transition["interpretation"],
            "last_cap_dominated_multiplier": transition["last_cap_dominated_multiplier"],
            "first_interior_multiplier": transition["first_interior_multiplier"],
        }
    return source


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


def direct_epsilon_direction(
    frozen: FrozenBatch,
    *,
    lambda_value: float,
    args: argparse.Namespace,
) -> tuple[jnp.ndarray, dict[str, Any]]:
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
    delta = (updated.inputs["epsilon"] - frozen.trial_specs.inputs["epsilon"]) * frozen.time_mask
    plain_diagnostics = {key: np.asarray(jax.device_get(value)) for key, value in diagnostics.items()}
    return delta, plain_diagnostics


def build_policy_directions(
    frozen: FrozenBatch,
    *,
    direct_delta: jnp.ndarray,
    ridge_alpha: float,
) -> list[PolicyDirection]:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    zero_features = live_features(frozen, jnp.zeros_like(epsilon))
    affine_bias = jnp.mean(direct_delta, axis=0)
    affine_params = (
        jnp.zeros((epsilon.shape[1], epsilon.shape[2], zero_features.shape[-1]), dtype=epsilon.dtype),
        affine_bias,
    )
    linear_weights, linear_fit = fit_linear_no_bias_policy(
        features=zero_features,
        target_delta=direct_delta,
        time_mask=frozen.time_mask,
        ridge_alpha=ridge_alpha,
    )
    zero_linear = jnp.zeros_like(linear_weights)
    return [
        PolicyDirection(
            label="zero_linear_no_bias",
            mechanism="linear_no_bias",
            params=zero_linear,
            fit_diagnostics={"fit_method": "zero_policy"},
        ),
        PolicyDirection(
            label="zero_affine",
            mechanism="affine",
            params=(
                jnp.zeros_like(linear_weights),
                jnp.zeros((epsilon.shape[1], epsilon.shape[2]), dtype=epsilon.dtype),
            ),
            fit_diagnostics={"fit_method": "zero_policy"},
        ),
        PolicyDirection(
            label="affine_mean_direct",
            mechanism="affine",
            params=affine_params,
            fit_diagnostics={
                "fit_method": "batch_mean_direct_epsilon_bias",
                "bias_norm": float(jnp.linalg.norm(affine_bias)),
            },
        ),
        PolicyDirection(
            label="linear_ridge_direct",
            mechanism="linear_no_bias",
            params=linear_weights,
            fit_diagnostics=linear_fit,
        ),
    ]


def fit_linear_no_bias_policy(
    *,
    features: jnp.ndarray,
    target_delta: jnp.ndarray,
    time_mask: jnp.ndarray,
    ridge_alpha: float,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    features_np = np.asarray(features)
    target_np = np.asarray(target_delta * time_mask)
    mask_np = np.asarray(time_mask[..., 0] > 0.0)
    time_steps = int(features_np.shape[1])
    feature_dim = int(features_np.shape[2])
    epsilon_dim = int(target_np.shape[2])
    weights = np.zeros((time_steps, epsilon_dim, feature_dim), dtype=features_np.dtype)
    predictions = np.zeros_like(target_np)
    ranks = []
    condition_numbers = []
    for time_index in range(time_steps):
        active = mask_np[:, time_index]
        if not np.any(active):
            ranks.append(0)
            condition_numbers.append(float("nan"))
            continue
        design = features_np[active, time_index, :]
        target = target_np[active, time_index, :]
        gram = design.T @ design
        scale = float(np.trace(gram) / max(feature_dim, 1))
        ridge = max(float(ridge_alpha) * max(scale, 1e-12), 1e-12)
        regularized = gram + ridge * np.eye(feature_dim, dtype=features_np.dtype)
        solution = np.linalg.solve(regularized, design.T @ target)
        weights[time_index, :, :] = solution.T
        predictions[active, time_index, :] = design @ solution
        ranks.append(int(np.linalg.matrix_rank(design)))
        condition_numbers.append(float(np.linalg.cond(regularized)))
    residual = target_np - predictions
    target_norm = float(np.linalg.norm(target_np))
    residual_norm = float(np.linalg.norm(residual))
    active_steps = int(np.sum(mask_np))
    return jnp.asarray(weights), {
        "fit_method": "per_time_ridge_from_zero_live_features_to_direct_epsilon",
        "ridge_alpha": float(ridge_alpha),
        "active_trial_steps": active_steps,
        "feature_rms": float(np.sqrt(np.mean(np.square(features_np[mask_np])))) if active_steps else 0.0,
        "target_norm": target_norm,
        "residual_norm": residual_norm,
        "relative_residual_norm": residual_norm / max(target_norm, 1e-12),
        "rank_min": int(np.min(ranks)) if ranks else 0,
        "rank_max": int(np.max(ranks)) if ranks else 0,
        "condition_number_max": max_or_nan(condition_numbers),
    }


def live_features(frozen: FrozenBatch, delta: jnp.ndarray) -> jnp.ndarray:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    candidate = _set_input(frozen.trial_specs, "epsilon", epsilon + delta * frozen.time_mask)
    states = frozen.task.eval_trials(frozen.model, candidate, frozen.keys_model)
    mechanics = jnp.asarray(states.mechanics.vector)
    features = mechanics[..., :6]
    target = jnp.asarray(frozen.trial_specs.targets["mechanics.effector.pos"].value)
    target = target[:, : features.shape[1], :]
    centered_pos = features[..., :2] - target
    return jnp.concatenate([centered_pos, features[..., 2:6]], axis=-1) * frozen.time_mask


def evaluate_policy_direction(
    frozen: FrozenBatch,
    *,
    direction: PolicyDirection,
    lambda_value: float,
    multiplier: float,
    amplitude: float,
    fixed_point_steps: int,
) -> dict[str, Any]:
    raw_delta = policy_raw_delta(
        frozen,
        direction.params,
        mechanism=direction.mechanism,
        amplitude=amplitude,
        fixed_point_steps=fixed_point_steps,
    )
    selected_delta = _project_flattened_per_trial_l2_ball(
        raw_delta * frozen.time_mask,
        frozen.radius,
    ) * frozen.time_mask
    zero_delta = jnp.zeros_like(raw_delta)
    raw_metrics = score_delta(frozen, raw_delta, lambda_value=lambda_value)
    selected_metrics = score_delta(frozen, selected_delta, lambda_value=lambda_value)
    zero_metrics = score_delta(frozen, zero_delta, lambda_value=lambda_value)
    raw_norm = _flattened_per_trial_norm(raw_delta)
    selected_norm = _flattened_per_trial_norm(selected_delta)
    raw_energy = per_trial_energy(raw_delta)
    selected_energy = per_trial_energy(selected_delta)
    raw_ratio = raw_norm / jnp.maximum(frozen.radius, 1e-12)
    selected_ratio = selected_norm / jnp.maximum(frozen.radius, 1e-12)
    return {
        "direction": direction.label,
        "mechanism": direction.mechanism,
        "lambda": float(lambda_value),
        "lambda_multiplier": float(multiplier),
        "amplitude": float(amplitude),
        "raw_policy_norm_mean": float(jnp.mean(raw_norm)),
        "raw_policy_norm_max": float(jnp.max(raw_norm)),
        "selected_clipped_norm_mean": float(jnp.mean(selected_norm)),
        "selected_clipped_norm_max": float(jnp.max(selected_norm)),
        "radius_mean": float(jnp.mean(frozen.radius)),
        "raw_to_selected_norm_ratio_mean": safe_mean_ratio(raw_norm, selected_norm),
        "raw_norm_cap_ratio_mean": float(jnp.mean(raw_ratio)),
        "raw_norm_cap_ratio_max": float(jnp.max(raw_ratio)),
        "selected_norm_cap_ratio_mean": float(jnp.mean(selected_ratio)),
        "selected_norm_cap_ratio_max": float(jnp.max(selected_ratio)),
        "cap_violation_fraction_before_projection": float(
            jnp.mean((raw_ratio > 1.0 + 1e-6).astype(jnp.float32))
        ),
        "cap_bound_fraction_after_projection": float(
            jnp.mean((selected_ratio >= 1.0 - 1e-4).astype(jnp.float32))
        ),
        "raw_energy_mean": float(jnp.mean(raw_energy)),
        "selected_clipped_energy_mean": float(jnp.mean(selected_energy)),
        "raw_energy_penalty": float(lambda_value * jnp.mean(raw_energy)),
        "selected_clipped_energy_penalty": float(lambda_value * jnp.mean(selected_energy)),
        "raw_task_loss_gain_over_zero": float(raw_metrics["task_loss"] - zero_metrics["task_loss"]),
        "selected_task_loss_gain_over_zero": float(
            selected_metrics["task_loss"] - zero_metrics["task_loss"]
        ),
        "raw_objective_gain_over_zero": float(raw_metrics["objective"] - zero_metrics["objective"]),
        "selected_objective_gain_over_zero": float(
            selected_metrics["objective"] - zero_metrics["objective"]
        ),
        "raw_finite_status": finite_status(raw_metrics),
        "selected_finite_status": finite_status(selected_metrics),
        "fit_diagnostics": plain_json(direction.fit_diagnostics),
    }


def policy_raw_delta(
    frozen: FrozenBatch,
    params: Any,
    *,
    mechanism: str,
    amplitude: float,
    fixed_point_steps: int,
) -> jnp.ndarray:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    delta = jnp.zeros_like(epsilon)
    for _ in range(int(fixed_point_steps)):
        features = live_features(frozen, delta)
        if mechanism == "affine":
            weights, bias = params
            raw = jnp.einsum("btf,tef->bte", features, weights) + bias[None, :, :]
        elif mechanism == "linear_no_bias":
            raw = jnp.einsum("btf,tef->bte", features, params)
        else:
            raise ValueError(f"Unknown mechanism {mechanism!r}")
        delta = float(amplitude) * raw * frozen.time_mask
    return delta * frozen.time_mask


def score_delta(
    frozen: FrozenBatch,
    delta: jnp.ndarray,
    *,
    lambda_value: float,
) -> dict[str, jnp.ndarray]:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    candidate = _set_input(frozen.trial_specs, "epsilon", epsilon + delta * frozen.time_mask)
    states = frozen.task.eval_trials(frozen.model, candidate, frozen.keys_model)
    task_loss_value = jnp.asarray(frozen.task.loss_func(states, candidate, frozen.model).total)
    energy = jnp.mean(per_trial_energy(delta * frozen.time_mask))
    objective = task_loss_value - float(lambda_value) * energy
    return {"task_loss": task_loss_value, "energy": energy, "objective": objective}


def per_trial_energy(delta: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(delta), axis=tuple(range(1, delta.ndim)))


def safe_mean_ratio(raw_norm: jnp.ndarray, selected_norm: jnp.ndarray) -> float:
    ratio = raw_norm / jnp.maximum(selected_norm, 1e-12)
    inactive = (raw_norm <= 1e-12) & (selected_norm <= 1e-12)
    ratio = jnp.where(inactive, 1.0, ratio)
    return float(jnp.mean(ratio))


def finite_status(metrics: dict[str, jnp.ndarray]) -> str:
    values = jnp.asarray([metrics["task_loss"], metrics["energy"], metrics["objective"]])
    return "finite" if bool(jnp.all(jnp.isfinite(values))) else "nonfinite"


def direct_summary(
    delta: jnp.ndarray,
    radius: jnp.ndarray,
    *,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    norm = _flattened_per_trial_norm(delta)
    energy = per_trial_energy(delta)
    ratio = norm / jnp.maximum(radius, 1e-12)
    return {
        "selected_epsilon_norm_mean": float(jnp.mean(norm)),
        "selected_epsilon_norm_max": float(jnp.max(norm)),
        "selected_norm_cap_ratio_max": float(jnp.max(ratio)),
        "cap_bound_fraction": float(jnp.mean((ratio >= 1.0 - 1e-4).astype(jnp.float32))),
        "energy_mean": float(jnp.mean(energy)),
        "raw_task_loss_gain": scalar_diagnostic(diagnostics, "raw_task_loss_selected")
        - scalar_diagnostic(diagnostics, "raw_task_loss_zero"),
        "penalized_gain_over_zero": scalar_diagnostic(
            diagnostics,
            "selected_objective_gain_over_zero",
        ),
        "finite_status": (
            "nonfinite"
            if bool(np.asarray(diagnostics.get("inner_objective_nonfinite_seen", False)))
            else "finite"
        ),
    }


def direction_summary(direction: PolicyDirection) -> dict[str, Any]:
    return {
        "direction": direction.label,
        "mechanism": direction.mechanism,
        "fit_diagnostics": plain_json(direction.fit_diagnostics),
    }


def best_rows_by_direction(audits: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best = {}
    for row in audits:
        key = row["direction"]
        if key not in best or row["raw_objective_gain_over_zero"] > best[key]["raw_objective_gain_over_zero"]:
            best[key] = row
    return best


def interpret_run(lambda_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_affine = best_across(lambda_rows, "affine_mean_direct")
    best_linear = best_across(lambda_rows, "linear_ridge_direct")
    zero_rows = [
        audit
        for lambda_row in lambda_rows
        for audit in lambda_row["audits"]
        if audit["direction"].startswith("zero_")
    ]
    zero_all_zero = all(abs(row["raw_objective_gain_over_zero"]) < 1e-9 for row in zero_rows)
    if best_affine and best_affine["raw_objective_gain_over_zero"] > 0.0:
        verdict = "prior_zero_policy_result_invalidated_as_optimizer_or_projection_artifact"
    elif best_linear and best_linear["raw_objective_gain_over_zero"] > 0.0:
        verdict = "linear_direction_improves_but_affine_not_needed"
    elif zero_all_zero:
        verdict = "unresolved_or_expressivity_limited"
    else:
        verdict = "unresolved"
    return {
        "verdict": verdict,
        "zero_policy_rows_are_zero": zero_all_zero,
        "best_affine_raw_objective_gain": None
        if best_affine is None
        else best_affine["raw_objective_gain_over_zero"],
        "best_linear_raw_objective_gain": None
        if best_linear is None
        else best_linear["raw_objective_gain_over_zero"],
        "best_affine": compact_best(best_affine),
        "best_linear": compact_best(best_linear),
    }


def interpret_overall(rows: list[dict[str, Any]]) -> str:
    verdicts = {row["interpretation"]["verdict"] for row in rows}
    if "prior_zero_policy_result_invalidated_as_optimizer_or_projection_artifact" in verdicts:
        return (
            "Known affine mean-direct directions improve the raw objective on frozen c92 rows, "
            "so the old zero closed-loop policy result should not be read as evidence that "
            "finite closed-loop policies lack useful directions. It is best interpreted as an "
            "optimizer/projection artifact; linear no-bias expressivity remains basis/scaling "
            "dependent."
        )
    if "linear_direction_improves_but_affine_not_needed" in verdicts:
        return (
            "Known linear directions improve at least one row, so zero-policy results are not "
            "supported as a closed-loop expressivity limit."
        )
    return (
        "The known directions did not improve the raw objective on this bounded audit; the "
        "zero-policy result remains unresolved rather than confirmed."
    )


def best_across(lambda_rows: list[dict[str, Any]], direction: str) -> dict[str, Any] | None:
    candidates = [
        audit
        for lambda_row in lambda_rows
        for audit in lambda_row["audits"]
        if audit["direction"] == direction
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["raw_objective_gain_over_zero"])


def compact_best(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keys = (
        "direction",
        "mechanism",
        "lambda_multiplier",
        "lambda",
        "amplitude",
        "raw_objective_gain_over_zero",
        "selected_objective_gain_over_zero",
        "raw_energy_penalty",
        "selected_clipped_energy_penalty",
        "raw_norm_cap_ratio_max",
        "cap_violation_fraction_before_projection",
        "raw_finite_status",
    )
    return {key: row[key] for key in keys}


def scalar_diagnostic(diagnostics: dict[str, Any], key: str) -> float:
    return float(np.asarray(diagnostics.get(key, np.nan)))


def max_or_nan(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return max(finite) if finite else float("nan")


def plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain_json(item) for key, item in value.items()}
    if isinstance(value, str):
        return value
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, (np.bool_, bool)):
            return bool(scalar)
        if isinstance(scalar, (np.integer, int)):
            return int(scalar)
        return float(scalar)
    return array.tolist()


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = [
        "run_id",
        "direction",
        "mechanism",
        "lambda_multiplier",
        "lambda",
        "amplitude",
        "raw_policy_norm_mean",
        "selected_clipped_norm_mean",
        "raw_to_selected_norm_ratio_mean",
        "cap_violation_fraction_before_projection",
        "raw_energy_penalty",
        "selected_clipped_energy_penalty",
        "raw_task_loss_gain_over_zero",
        "raw_objective_gain_over_zero",
        "selected_objective_gain_over_zero",
        "raw_finite_status",
        "selected_finite_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in payload["rows"]:
            for lambda_row in row["lambda_rows"]:
                for audit in lambda_row["audits"]:
                    writer.writerow({"run_id": row["run_id"], **{key: audit[key] for key in fields[1:]}})


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Closed-loop policy audit",
        "",
        f"Issue: `{payload['issue']}`. Source frozen no-PGD runs: `c92ebd8`.",
        "",
        "No controller weights were updated and no training was launched. The repaired audit "
        "scores raw finite-policy output with `J(raw_epsilon) - lambda * E(raw_epsilon)`. "
        "The per-trial L2 cap is then applied only to produce selected/clipped diagnostics.",
        "",
        "## Lambda choices",
        "",
        "Lambda values use the `093d949` per-trial p90 sweep center and compact multipliers "
        "around the observed cap-to-interior transition.",
        "",
        "| row | multiplier | lambda | direct cap-bound | direct gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        for lambda_row in row["lambda_rows"]:
            direct = lambda_row["direct_epsilon"]
            lines.append(
                f"| `{row['run_id']}` | {lambda_row['multiplier']:.6g} | "
                f"{lambda_row['lambda']:.6g} | {direct['cap_bound_fraction']:.3%} | "
                f"{direct['penalized_gain_over_zero']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Best known directions",
            "",
            "The table shows the best scalar-amplitude row for each known direction. "
            "Full per-amplitude raw norms, selected/clipped norms, raw-to-selected ratios, "
            "cap-violation fractions, raw energy penalties, selected/clipped energy penalties, "
            "objective gains, and finite/nonfinite statuses are in the tracked JSON and CSV.",
            "",
            "| row | direction | lambda mult | amplitude | raw gain | selected gain | raw energy penalty | clipped energy penalty | raw/cap max | cap violations | finite |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        best_rows = [
            row["interpretation"].get("best_affine"),
            row["interpretation"].get("best_linear"),
        ]
        for best in best_rows:
            if best is None:
                continue
            lines.append(
                f"| `{row['run_id']}` | `{best['direction']}` | "
                f"{best['lambda_multiplier']:.6g} | {best['amplitude']:.6g} | "
                f"{best['raw_objective_gain_over_zero']:.6g} | "
                f"{best['selected_objective_gain_over_zero']:.6g} | "
                f"{best['raw_energy_penalty']:.6g} | "
                f"{best['selected_clipped_energy_penalty']:.6g} | "
                f"{best['raw_norm_cap_ratio_max']:.6g} | "
                f"{best['cap_violation_fraction_before_projection']:.3%} | "
                f"{best['raw_finite_status']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["overall_interpretation"],
            "",
            "The affine mean-direct direction is an expressivity check, not a scientific success "
            "criterion. A positive known-direction line-search result means the previous zero "
            "closed-loop policy result cannot by itself support a no-useful-policy claim. The "
            "linear no-bias ridge direction is still sensitive to the live feature basis and "
            "feature scaling; weak fits should be read as unresolved basis/scaling evidence, "
            "not as proof that no-bias policies are dead.",
            "",
            "## Runtime caveats",
            "",
            "- The line search uses frozen deterministic c92 train batches and existing final checkpoints.",
            "- The linear ridge fit is per-time from zero-policy live features to direct epsilon; it does not optimize policy weights.",
            "- Heavy arrays were not persisted; tracked JSON/CSV carry scalar summaries only.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
