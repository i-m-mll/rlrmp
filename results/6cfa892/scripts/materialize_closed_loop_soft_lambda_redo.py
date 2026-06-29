"""Materialize closed-loop soft-lambda redo rows for 6cfa892."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT


RUN_IDS = ("open_loop_small", "open_loop_moderate", "open_loop_stress")
BETA_VALUES = (0.95, 1.05, 1.2, 1.4, 1.8)
MECHANISMS = ("linear_no_bias", "affine")
OPTIMIZERS = ("line_search_known_direction", "adam")
LINE_SEARCH_AMPLITUDES = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
HVP_SOURCE_JSON = "results/06a4dc8/canonical_soft_lambda_hvp.json"
DIRECT_CONTEXT_JSON = "results/7180984/direct_epsilon_soft_lambda_redo.json"
REFERENCE_POLICY_AUDIT = (
    REPO_ROOT / "results" / "3b850d6" / "scripts" / "materialize_closed_loop_policy_audit.py"
)
REFERENCE_CRITICAL_SEARCH = (
    REPO_ROOT / "results" / "1697bdc" / "scripts" / "materialize_critical_lambda_search.py"
)
REFERENCE_DIRECT_REDO = (
    REPO_ROOT / "results" / "7180984" / "scripts" / "materialize_direct_epsilon_soft_lambda_redo.py"
)
PRIMARY_LAMBDA_SUMMARY = "lambda_star_p90"
CAP_RADIUS_15CM = 0.004545500088363065
CAP_SOURCE = "ofb_6d_no_integrator_gamma_1p4_rollout_radius"
OBJECTIVE_GAIN_TOL = 1e-5
TASK_GAIN_TOL = 1e-9
NONZERO_NORM_TOL = 1e-12


@dataclass(frozen=True)
class AuditRow:
    run_id: str
    mechanism: str
    optimizer: str
    beta: float
    beta_role: str
    lambda_value: float
    lambda_mapping: str
    lambda_source: str
    lambda_star_summary: str
    lambda_star_summary_value: float
    finite_status: str
    gradient_status: str
    optimizer_success: bool
    optimizer_status: str
    optimizer_iterations: int
    optimizer_evaluations: int
    selected_nonzero: bool
    classification: str
    objective_level_success: bool
    penalized_gain_over_zero: float
    task_loss_gain: float
    energy_mean: float
    energy_max: float
    energy_penalty: float
    penalty_minus_task_gain: float
    penalty_over_task_gain_abs: float
    selected_policy_norm_mean: float
    selected_policy_norm_max: float
    old_cap_ratio_mean_sidecar: float
    old_cap_ratio_max_sidecar: float
    old_cap_boundary_fraction_sidecar: float
    old_cap_used_as_criterion: bool
    gradient_norm: float | None
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mechanism": self.mechanism,
            "optimizer": self.optimizer,
            "beta": self.beta,
            "beta_role": self.beta_role,
            "lambda": self.lambda_value,
            "lambda_mapping": self.lambda_mapping,
            "lambda_source": self.lambda_source,
            "lambda_star_summary": self.lambda_star_summary,
            "lambda_star_summary_value": self.lambda_star_summary_value,
            "finite_status": self.finite_status,
            "gradient_status": self.gradient_status,
            "optimizer_success": self.optimizer_success,
            "optimizer_status": self.optimizer_status,
            "optimizer_iterations": self.optimizer_iterations,
            "optimizer_evaluations": self.optimizer_evaluations,
            "selected_nonzero": self.selected_nonzero,
            "classification": self.classification,
            "objective_level_success": self.objective_level_success,
            "penalized_gain_over_zero": self.penalized_gain_over_zero,
            "task_loss_gain": self.task_loss_gain,
            "energy_mean": self.energy_mean,
            "energy_max": self.energy_max,
            "energy_penalty": self.energy_penalty,
            "penalty_minus_task_gain": self.penalty_minus_task_gain,
            "penalty_over_task_gain_abs": self.penalty_over_task_gain_abs,
            "selected_policy_norm_mean": self.selected_policy_norm_mean,
            "selected_policy_norm_max": self.selected_policy_norm_max,
            "old_cap_ratio_mean_sidecar": self.old_cap_ratio_mean_sidecar,
            "old_cap_ratio_max_sidecar": self.old_cap_ratio_max_sidecar,
            "old_cap_boundary_fraction_sidecar": self.old_cap_boundary_fraction_sidecar,
            "old_cap_used_as_criterion": self.old_cap_used_as_criterion,
            "gradient_norm": self.gradient_norm,
            "details": self.details,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen c92 no-PGD closed-loop linear no-bias and affine policies "
            "at corrected HVP/Lanczos p90 soft-lambda beta scales."
        )
    )
    parser.add_argument("--experiment", default="c92ebd8")
    parser.add_argument("--issue", default="6cfa892")
    parser.add_argument("--run-ids", nargs="+", default=list(RUN_IDS))
    parser.add_argument("--hvp-source-json", default=HVP_SOURCE_JSON)
    parser.add_argument("--direct-context-json", default=DIRECT_CONTEXT_JSON)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETA_VALUES))
    parser.add_argument("--mechanisms", nargs="+", default=list(MECHANISMS), choices=MECHANISMS)
    parser.add_argument("--optimizers", nargs="+", default=list(OPTIMIZERS))
    parser.add_argument("--pgd-steps", type=int, default=8)
    parser.add_argument("--pgd-step-size-fraction", type=float, default=0.25)
    parser.add_argument("--fixed-point-steps", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--line-search-reference-beta", type=float, default=1.4)
    parser.add_argument(
        "--line-search-amplitudes",
        type=float,
        nargs="+",
        default=list(LINE_SEARCH_AMPLITUDES),
    )
    parser.add_argument("--adam-steps", type=int, default=8)
    parser.add_argument("--adam-learning-rate", type=float, default=5e-5)
    parser.add_argument("--lbfgsb-maxiter", type=int, default=4)
    parser.add_argument(
        "--output-json",
        default="results/6cfa892/closed_loop_soft_lambda_redo.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/6cfa892/closed_loop_soft_lambda_redo.csv",
    )
    parser.add_argument(
        "--output-md",
        default="results/6cfa892/notes/closed_loop_soft_lambda_redo.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = materialize(args)
    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    output_md = REPO_ROOT / args.output_md
    write_compact_json(output_json, payload)
    write_csv(output_csv, payload)
    update_marked_section(output_md, "closed_loop_soft_lambda_redo", render_markdown(payload))
    print(
        json.dumps(
            {
                "json": str(output_json),
                "csv": str(output_csv),
                "markdown": str(output_md),
                "hvp_source_json": str((REPO_ROOT / args.hvp_source_json).resolve()),
            },
            indent=2,
        )
    )
    return 0


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_module("closed_loop_policy_audit_reference_3b850d6", REFERENCE_POLICY_AUDIT)
    search = load_module("critical_lambda_search_reference_1697bdc", REFERENCE_CRITICAL_SEARCH)
    direct = load_module("direct_epsilon_soft_lambda_redo_reference_7180984", REFERENCE_DIRECT_REDO)
    hvp_payload = direct.read_hvp_payload(REPO_ROOT / args.hvp_source_json)
    direct_context = read_optional_json(REPO_ROOT / args.direct_context_json)
    source_rows = {row["run_id"]: row for row in hvp_payload["rows"]}
    run_rows = []
    flat_rows = []
    for run_id in args.run_ids:
        if run_id not in source_rows:
            raise ValueError(f"Run {run_id!r} is missing from {args.hvp_source_json}")
        source_row = source_rows[run_id]
        frozen = reference.load_frozen_batch(args, run_id)
        beta_rows = direct.beta_mapping_from_source(source_row, args.betas)
        direction_cache = build_direction_cache(
            reference,
            frozen,
            beta_rows=beta_rows,
            reference_beta=float(args.line_search_reference_beta),
            ridge_alpha=float(args.ridge_alpha),
            args=args,
        )
        row_dicts = []
        for mechanism in args.mechanisms:
            codec = search.make_theta_codec(reference, frozen, mechanism)
            value_and_grad = search.make_closed_loop_value_and_grad(reference, frozen, codec, args)
            for optimizer in args.optimizers:
                for beta_row in beta_rows:
                    audit_row = evaluate_closed_loop_row(
                        reference=reference,
                        search=search,
                        frozen=frozen,
                        codec=codec,
                        value_and_grad=value_and_grad,
                        direction=direction_cache[mechanism],
                        run_id=run_id,
                        mechanism=mechanism,
                        optimizer=optimizer,
                        beta_row=beta_row,
                        args=args,
                    )
                    row_dict = audit_row.as_dict()
                    row_dicts.append(row_dict)
                    flat_rows.append(row_dict)
        run_rows.append(
            {
                "run_id": run_id,
                "run_spec_path": f"results/{args.experiment}/runs/{run_id}.json",
                "artifact_dir": f"_artifacts/{args.experiment}/runs/{run_id}",
                "checkpoint": f"_artifacts/{args.experiment}/runs/{run_id}/trained_model.eqx",
                "hvp_source": direct.hvp_row_source_summary(source_row, str(args.hvp_source_json)),
                "beta_mapping": beta_rows,
                "direct_epsilon_context": direct_context_for_run(direct_context, run_id),
                "closed_loop_rows": row_dicts,
                "objective_classification_counts": classification_counts(row_dicts),
                "best_by_mechanism_optimizer": best_rows_by_mechanism_optimizer(row_dicts),
            }
        )
    return {
        "schema_version": "rlrmp.closed_loop_soft_lambda_redo.v1",
        "issue": str(args.issue),
        "parent_umbrella": "54389a4",
        "coordinator_issue": "31e67ff",
        "source_experiment": str(args.experiment),
        "command": {"argv": list(sys.argv), "cwd": "."},
        "hvp_source": {
            "path": str(args.hvp_source_json),
            "schema_version": hvp_payload.get("schema_version"),
            "issue": hvp_payload.get("issue"),
            "estimator_method": hvp_payload.get("estimator", {}).get("method"),
            "primary_continuity_summary": PRIMARY_LAMBDA_SUMMARY,
            "lambda_convention": hvp_payload.get("estimator", {}).get("lambda_star_convention"),
            "pooled_summary": hvp_payload.get("pooled_summary"),
            "pooled_beta_mapping": hvp_payload.get("pooled_beta_mapping"),
        },
        "direct_epsilon_context_path": str(args.direct_context_json),
        "beta_policy": {
            "formula": "lambda(beta) = beta^2 * substrate_p90(lambda_star_i)",
            "values": [float(beta) for beta in args.betas],
            "diagnostic_only_beta_values": [
                float(beta) for beta in args.betas if float(beta) < 1.0
            ],
            "primary_beta_values": [float(beta) for beta in args.betas if float(beta) >= 1.0],
        },
        "closed_loop_contract": {
            "mechanisms": list(args.mechanisms),
            "optimizers": list(args.optimizers),
            "objective": "mean_i[J_i(raw_policy_epsilon_i) - lambda * E_i(raw_policy_epsilon_i)]",
            "controller_updates": False,
            "optimized_variables": (
                "closed-loop policy parameters only for adam/lbfgsb; scalar amplitude only for "
                "line_search_known_direction"
            ),
            "line_search_reference_beta": float(args.line_search_reference_beta),
            "line_search_amplitudes": [float(value) for value in args.line_search_amplitudes],
            "fixed_point_steps": int(args.fixed_point_steps),
            "batch_size": int(args.batch_size),
            "replicate_index": int(args.replicate_index),
        },
        "classification_contract": {
            "criterion": "objective-level behavior only",
            "uses_cap_or_interiority_as_criterion": False,
            "finite_required": True,
            "nonzero_threshold_l2": NONZERO_NORM_TOL,
            "positive_penalized_gain_threshold": OBJECTIVE_GAIN_TOL,
            "positive_task_gain_threshold": TASK_GAIN_TOL,
            "old_cap_role": "sidecar diagnostic only; not pass/fail and not lambda selection",
        },
        "old_cap_sidecar": {
            "radius_15cm": CAP_RADIUS_15CM,
            "source": CAP_SOURCE,
        },
        "rows": run_rows,
        "flat_rows": flat_rows,
        "overall_interpretation": interpret_overall(run_rows),
    }


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reference module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_direction_cache(
    reference: Any,
    frozen: Any,
    *,
    beta_rows: list[dict[str, Any]],
    reference_beta: float,
    ridge_alpha: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    beta_row = nearest_beta_row(beta_rows, reference_beta)
    direct_delta, _diagnostics = reference.direct_epsilon_direction(
        frozen,
        lambda_value=float(beta_row["lambda"]),
        args=args,
    )
    directions = reference.build_policy_directions(
        frozen,
        direct_delta=direct_delta,
        ridge_alpha=ridge_alpha,
    )
    selected = {
        direction.mechanism: direction
        for direction in directions
        if direction.label in {"linear_ridge_direct", "affine_mean_direct"}
    }
    missing = set(MECHANISMS) - set(selected)
    if missing:
        raise RuntimeError(f"Missing closed-loop directions: {sorted(missing)}")
    return selected


def nearest_beta_row(beta_rows: list[dict[str, Any]], beta: float) -> dict[str, Any]:
    return min(beta_rows, key=lambda row: abs(float(row["beta"]) - float(beta)))


def evaluate_closed_loop_row(
    *,
    reference: Any,
    search: Any,
    frozen: Any,
    codec: Any,
    value_and_grad: Any,
    direction: Any,
    run_id: str,
    mechanism: str,
    optimizer: str,
    beta_row: dict[str, Any],
    args: argparse.Namespace,
) -> AuditRow:
    lambda_value = float(beta_row["lambda"])
    zero = jnp.zeros_like(frozen.trial_specs.inputs["epsilon"])
    zero_metrics = search.score_closed_loop_delta(
        reference, frozen, zero, jnp.asarray(lambda_value)
    )
    if optimizer == "line_search_known_direction":
        result = evaluate_line_search(
            reference=reference,
            search=search,
            frozen=frozen,
            direction=direction,
            lambda_value=lambda_value,
            zero_metrics=zero_metrics,
            args=args,
        )
    elif optimizer == "adam":
        opt_result = search.optimize_with_adam(codec, value_and_grad, lambda_value, args)
        delta = search.closed_loop_raw_delta(
            reference,
            frozen,
            codec,
            jnp.asarray(opt_result["theta"]),
            int(args.fixed_point_steps),
        )
        metrics = search.score_closed_loop_delta(
            reference, frozen, delta, jnp.asarray(lambda_value)
        )
        grad = jnp.asarray(opt_result["grad"])
        result = {
            "delta": delta,
            "metrics": metrics,
            "gradient_norm": float(jnp.linalg.norm(grad)),
            "gradient_status": "finite" if bool(jnp.all(jnp.isfinite(grad))) else "nonfinite",
            "optimizer_success": bool(opt_result["success"]),
            "optimizer_status": str(opt_result["status"]),
            "optimizer_iterations": int(opt_result["iterations"]),
            "optimizer_evaluations": int(opt_result["evaluations"]),
            "details": {"theta_size": codec.size, "fixed_point_steps": int(args.fixed_point_steps)},
        }
    elif optimizer == "lbfgsb":
        opt_result = search.optimize_with_lbfgsb(codec, value_and_grad, lambda_value, args)
        delta = search.closed_loop_raw_delta(
            reference,
            frozen,
            codec,
            jnp.asarray(opt_result["theta"]),
            int(args.fixed_point_steps),
        )
        metrics = search.score_closed_loop_delta(
            reference, frozen, delta, jnp.asarray(lambda_value)
        )
        grad = jnp.asarray(opt_result["grad"])
        result = {
            "delta": delta,
            "metrics": metrics,
            "gradient_norm": float(jnp.linalg.norm(grad)),
            "gradient_status": "finite" if bool(jnp.all(jnp.isfinite(grad))) else "nonfinite",
            "optimizer_success": bool(opt_result["success"]),
            "optimizer_status": str(opt_result["status"]),
            "optimizer_iterations": int(opt_result["iterations"]),
            "optimizer_evaluations": int(opt_result["evaluations"]),
            "details": {"theta_size": codec.size, "fixed_point_steps": int(args.fixed_point_steps)},
        }
    else:
        raise ValueError(f"Unknown optimizer {optimizer!r}")
    return summarize_delta(
        reference=reference,
        frozen=frozen,
        run_id=run_id,
        mechanism=mechanism,
        optimizer=optimizer,
        beta_row=beta_row,
        lambda_value=lambda_value,
        delta=result["delta"],
        metrics=result["metrics"],
        zero_metrics=zero_metrics,
        gradient_norm=result["gradient_norm"],
        gradient_status=result["gradient_status"],
        optimizer_success=result["optimizer_success"],
        optimizer_status=result["optimizer_status"],
        optimizer_iterations=result["optimizer_iterations"],
        optimizer_evaluations=result["optimizer_evaluations"],
        details=result["details"],
    )


def evaluate_line_search(
    *,
    reference: Any,
    search: Any,
    frozen: Any,
    direction: Any,
    lambda_value: float,
    zero_metrics: dict[str, jnp.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for amplitude in args.line_search_amplitudes:
        delta = reference.policy_raw_delta(
            frozen,
            direction.params,
            mechanism=direction.mechanism,
            amplitude=float(amplitude),
            fixed_point_steps=int(args.fixed_point_steps),
        )
        metrics = search.score_closed_loop_delta(
            reference, frozen, delta, jnp.asarray(lambda_value)
        )
        objective_gain = float(metrics["objective"] - zero_metrics["objective"])
        if best is None or objective_gain > best["objective_gain"]:
            best = {
                "amplitude": float(amplitude),
                "delta": delta,
                "metrics": metrics,
                "objective_gain": objective_gain,
            }
    if best is None:
        raise RuntimeError("Line-search amplitudes were empty")
    gradient_norm = search.line_search_gradient_norm(
        reference,
        frozen,
        direction,
        lambda_value,
        float(best["amplitude"]),
        int(args.fixed_point_steps),
    )
    return {
        "delta": best["delta"],
        "metrics": best["metrics"],
        "gradient_norm": gradient_norm,
        "gradient_status": "finite" if np.isfinite(gradient_norm) else "nonfinite",
        "optimizer_success": np.isfinite(float(best["metrics"]["objective"])),
        "optimizer_status": "best_grid_amplitude",
        "optimizer_iterations": len(args.line_search_amplitudes),
        "optimizer_evaluations": len(args.line_search_amplitudes),
        "details": {
            "direction": direction.label,
            "best_amplitude": best["amplitude"],
            "fit_diagnostics": plain_json(direction.fit_diagnostics),
            "reference_only": True,
        },
    }


def summarize_delta(
    *,
    reference: Any,
    frozen: Any,
    run_id: str,
    mechanism: str,
    optimizer: str,
    beta_row: dict[str, Any],
    lambda_value: float,
    delta: jnp.ndarray,
    metrics: dict[str, jnp.ndarray],
    zero_metrics: dict[str, jnp.ndarray],
    gradient_norm: float | None,
    gradient_status: str,
    optimizer_success: bool,
    optimizer_status: str,
    optimizer_iterations: int,
    optimizer_evaluations: int,
    details: dict[str, Any],
) -> AuditRow:
    delta = delta * frozen.time_mask
    norm = reference._flattened_per_trial_norm(delta)
    energy = per_trial_energy(delta)
    ratio = norm / jnp.maximum(frozen.radius, 1e-12)
    task_gain = float(metrics["task_loss"] - zero_metrics["task_loss"])
    penalized_gain = float(metrics["objective"] - zero_metrics["objective"])
    energy_mean = float(jnp.mean(energy))
    energy_penalty = float(lambda_value * energy_mean)
    finite = bool(
        jnp.all(
            jnp.isfinite(
                jnp.asarray(
                    [
                        metrics["task_loss"],
                        metrics["energy"],
                        metrics["objective"],
                        zero_metrics["objective"],
                        lambda_value,
                        energy_mean,
                    ]
                )
            )
        )
    )
    finite_status = "finite" if finite else "nonfinite"
    selected_norm_max = float(jnp.max(norm))
    selected_nonzero = selected_norm_max > NONZERO_NORM_TOL
    classification = classify_objective_behavior(
        finite_status=finite_status,
        gradient_status=gradient_status,
        selected_nonzero=selected_nonzero,
        penalized_gain=penalized_gain,
        task_gain=task_gain,
    )
    objective_level_success = (
        finite_status == "finite"
        and gradient_status == "finite"
        and selected_nonzero
        and penalized_gain > OBJECTIVE_GAIN_TOL
    )
    task_gain_denominator = max(abs(task_gain), 1e-30)
    return AuditRow(
        run_id=run_id,
        mechanism=mechanism,
        optimizer=optimizer,
        beta=float(beta_row["beta"]),
        beta_role=str(beta_row["role"]),
        lambda_value=lambda_value,
        lambda_mapping=str(beta_row["mapping"]),
        lambda_source=str(beta_row["lambda_source"]),
        lambda_star_summary=str(beta_row["lambda_star_summary"]),
        lambda_star_summary_value=float(beta_row["lambda_star_summary_value"]),
        finite_status=finite_status,
        gradient_status=gradient_status,
        optimizer_success=bool(optimizer_success),
        optimizer_status=str(optimizer_status),
        optimizer_iterations=int(optimizer_iterations),
        optimizer_evaluations=int(optimizer_evaluations),
        selected_nonzero=bool(selected_nonzero),
        classification=classification,
        objective_level_success=objective_level_success,
        penalized_gain_over_zero=penalized_gain,
        task_loss_gain=task_gain,
        energy_mean=energy_mean,
        energy_max=float(jnp.max(energy)),
        energy_penalty=energy_penalty,
        penalty_minus_task_gain=float(energy_penalty - task_gain),
        penalty_over_task_gain_abs=float(energy_penalty / task_gain_denominator),
        selected_policy_norm_mean=float(jnp.mean(norm)),
        selected_policy_norm_max=selected_norm_max,
        old_cap_ratio_mean_sidecar=float(jnp.mean(ratio)),
        old_cap_ratio_max_sidecar=float(jnp.max(ratio)),
        old_cap_boundary_fraction_sidecar=float(
            jnp.mean((ratio >= 1.0 - 1e-4).astype(jnp.float32))
        ),
        old_cap_used_as_criterion=False,
        gradient_norm=None if gradient_norm is None else float(gradient_norm),
        details=plain_json(details),
    )


def classify_objective_behavior(
    *,
    finite_status: str,
    gradient_status: str,
    selected_nonzero: bool,
    penalized_gain: float,
    task_gain: float,
) -> str:
    if finite_status != "finite" or gradient_status != "finite":
        return "optimizer_nonfinite"
    if selected_nonzero and penalized_gain > OBJECTIVE_GAIN_TOL:
        if task_gain > TASK_GAIN_TOL:
            return "nonzero_positive_penalized_and_task_gain"
        return "nonzero_positive_penalized_gain_without_task_gain"
    if selected_nonzero and penalized_gain < -OBJECTIVE_GAIN_TOL:
        return "nonzero_negative_penalized_gain"
    if selected_nonzero:
        return "nonzero_flat_penalized_gain"
    if penalized_gain > OBJECTIVE_GAIN_TOL:
        return "zero_selected_positive_penalized_gain"
    return "zero_selected_no_positive_penalized_gain"


def per_trial_energy(delta: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(delta), axis=tuple(range(1, delta.ndim)))


def direct_context_for_run(payload: dict[str, Any] | None, run_id: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    for row in payload.get("rows", []):
        if row.get("run_id") == run_id:
            return {
                "source_issue": payload.get("issue"),
                "path": DIRECT_CONTEXT_JSON,
                "objective_classification_counts": row.get("objective_classification_counts"),
            }
    return None


def classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["classification"])
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def best_rows_by_mechanism_optimizer(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['mechanism']}::{row['optimizer']}"
        if (
            key not in best
            or row["penalized_gain_over_zero"] > best[key]["penalized_gain_over_zero"]
        ):
            best[key] = compact_best(row)
    return best


def compact_best(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "mechanism",
        "optimizer",
        "beta",
        "beta_role",
        "lambda",
        "finite_status",
        "gradient_status",
        "optimizer_success",
        "classification",
        "objective_level_success",
        "penalized_gain_over_zero",
        "task_loss_gain",
        "energy_mean",
        "energy_penalty",
        "selected_policy_norm_max",
        "old_cap_ratio_max_sidecar",
        "old_cap_boundary_fraction_sidecar",
        "old_cap_used_as_criterion",
    )
    return {key: row[key] for key in keys}


def interpret_overall(run_rows: list[dict[str, Any]]) -> str:
    flat_rows = [row for run in run_rows for row in run["closed_loop_rows"]]
    successes = [row for row in flat_rows if row["objective_level_success"]]
    finite_rows = [
        row
        for row in flat_rows
        if row["finite_status"] == "finite" and row["gradient_status"] == "finite"
    ]
    diagnostic_successes = [row for row in successes if row["beta_role"] == "diagnostic_only"]
    primary_successes = [row for row in successes if row["beta_role"] != "diagnostic_only"]
    return (
        f"Closed-loop objective-level rows were finite on {len(finite_rows)}/{len(flat_rows)} "
        f"evaluations and produced positive nonzero penalized-gain behavior on "
        f"{len(successes)}/{len(flat_rows)} evaluations. Of those successes, "
        f"{len(diagnostic_successes)} were beta<1 diagnostic rows and "
        f"{len(primary_successes)} were beta>=1 candidate-scale rows. Old-cap ratios were "
        "reported only as sidecars and did not enter classification."
    )


def plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    if isinstance(value, str) or value is None:
        return value
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "mechanism",
        "optimizer",
        "beta",
        "beta_role",
        "lambda",
        "lambda_star_summary",
        "lambda_star_summary_value",
        "finite_status",
        "gradient_status",
        "optimizer_success",
        "optimizer_status",
        "optimizer_iterations",
        "optimizer_evaluations",
        "selected_nonzero",
        "classification",
        "objective_level_success",
        "penalized_gain_over_zero",
        "task_loss_gain",
        "energy_mean",
        "energy_max",
        "energy_penalty",
        "penalty_minus_task_gain",
        "penalty_over_task_gain_abs",
        "selected_policy_norm_mean",
        "selected_policy_norm_max",
        "old_cap_ratio_mean_sidecar",
        "old_cap_ratio_max_sidecar",
        "old_cap_boundary_fraction_sidecar",
        "old_cap_used_as_criterion",
        "gradient_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in payload["flat_rows"]:
            writer.writerow({key: row[key] for key in fieldnames})


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Closed-loop soft-lambda redo",
        "",
        f"Issue: `{payload['issue']}`. Source no-PGD substrates: `c92ebd8`.",
        "",
        (
            "No training was launched and no controller weights were updated. This deterministic "
            "local materializer loads the frozen c92 substrates and evaluates closed-loop "
            "linear no-bias and affine mechanisms at beta-scaled lambda values from the "
            "corrected HVP/Lanczos p90 source."
        ),
        "",
        "## Source contract",
        "",
        (
            f"HVP source: `{payload['hvp_source']['path']}` "
            f"(`{payload['hvp_source']['schema_version']}`). Primary scale: "
            f"`{payload['hvp_source']['primary_continuity_summary']}`."
        ),
        "",
        (
            "Beta mapping: `lambda(beta) = beta^2 * substrate_p90(lambda_star_i)`. "
            "Beta `0.95` is diagnostic only. Cap/interiority is not used as a criterion; "
            "old-cap ratios below are sidecars only."
        ),
        "",
        "## HVP/p90 beta mapping",
        "",
        "| substrate | beta | role | lambda_star p90 | lambda | source |",
        "|---|---:|---|---:|---:|---|",
    ]
    for run in payload["rows"]:
        for mapping in run["beta_mapping"]:
            lines.append(
                f"| `{run['run_id']}` | {mapping['beta']:.3g} | {mapping['role']} | "
                f"{mapping['lambda_star_summary_value']:.6g} | {mapping['lambda']:.6g} | "
                f"{mapping['lambda_source']} |"
            )
    lines.extend(
        [
            "",
            "## Objective-level rows",
            "",
            "| substrate | mechanism | optimizer | beta | finite | grad | class | objective success | "
            "penalized gain | task gain | energy | penalty | norm | old-cap ratio |",
            "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["flat_rows"]:
        lines.append(
            f"| `{row['run_id']}` | `{row['mechanism']}` | `{row['optimizer']}` | "
            f"{row['beta']:.3g} | {row['finite_status']} | {row['gradient_status']} | "
            f"`{row['classification']}` | {str(row['objective_level_success']).lower()} | "
            f"{row['penalized_gain_over_zero']:.6g} | {row['task_loss_gain']:.6g} | "
            f"{row['energy_mean']:.6g} | {row['energy_penalty']:.6g} | "
            f"{row['selected_policy_norm_max']:.6g} | {row['old_cap_ratio_max_sidecar']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Best objective rows",
            "",
            "| substrate | mechanism | optimizer | beta | class | penalized gain | task gain | "
            "energy penalty | norm | old-cap ratio |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in payload["rows"]:
        for best in run["best_by_mechanism_optimizer"].values():
            lines.append(
                f"| `{run['run_id']}` | `{best['mechanism']}` | `{best['optimizer']}` | "
                f"{best['beta']:.3g} | `{best['classification']}` | "
                f"{best['penalized_gain_over_zero']:.6g} | {best['task_loss_gain']:.6g} | "
                f"{best['energy_penalty']:.6g} | {best['selected_policy_norm_max']:.6g} | "
                f"{best['old_cap_ratio_max_sidecar']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Classification counts",
            "",
            "| substrate | counts |",
            "|---|---|",
        ]
    )
    for run in payload["rows"]:
        counts = ", ".join(
            f"`{label}`: {count}" for label, count in run["objective_classification_counts"].items()
        )
        lines.append(f"| `{run['run_id']}` | {counts} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["overall_interpretation"],
            "",
            "The old hard cap is retained only as `old_cap_*_sidecar` provenance. It is not used "
            "to select lambda, to define success, or to classify any row.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run --no-sync python results/6cfa892/scripts/materialize_closed_loop_soft_lambda_redo.py",
            "```",
            "",
            "For a fast local smoke:",
            "",
            "```bash",
            "uv run --no-sync python results/6cfa892/scripts/materialize_closed_loop_soft_lambda_redo.py \\",
            "  --run-ids open_loop_small --betas 0.95 1.4 --optimizers line_search_known_direction \\",
            "  --output-json results/6cfa892/smoke/closed_loop_soft_lambda_redo.json \\",
            "  --output-csv results/6cfa892/smoke/closed_loop_soft_lambda_redo.csv \\",
            "  --output-md results/6cfa892/smoke/closed_loop_soft_lambda_redo.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
