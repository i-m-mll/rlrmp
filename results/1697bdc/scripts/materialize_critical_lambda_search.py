"""Materialize practical frozen-audit critical lambda estimates for 1697bdc."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT, mkdir_p


RUN_IDS = ("open_loop_small", "open_loop_moderate", "open_loop_stress")
DIRECT_BRACKETS = {
    "open_loop_small": (2.0, 4.0),
    "open_loop_moderate": (2.0, 4.0),
    "open_loop_stress": (1.0, 2.0),
}
DEFAULT_CLOSED_LOOP_PROBES = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_LINE_SEARCH_AMPLITUDES = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MECHANISMS = ("direct_epsilon", "linear_no_bias", "affine")
CLOSED_LOOP_MECHANISMS = ("linear_no_bias", "affine")
CLOSED_LOOP_OPTIMIZERS = ("line_search_known_direction", "adam", "lbfgsb")
SOURCE_LAMBDA_SWEEP = REPO_ROOT / "results" / "093d949" / "soft_lambda_sweep.json"
REFERENCE_POLICY_AUDIT = (
    REPO_ROOT / "results" / "3b850d6" / "scripts" / "materialize_closed_loop_policy_audit.py"
)


@dataclass(frozen=True)
class SearchPoint:
    run_id: str
    mechanism: str
    optimizer: str
    phase: str
    point_index: int
    lambda_multiplier: float
    lambda_value: float
    objective_gain_over_zero: float
    task_loss_gain_over_zero: float
    energy_penalty: float
    energy_mean: float
    max_norm_over_cap: float
    mean_norm_over_cap: float
    cap_bound_fraction: float
    finite_status: str
    gradient_status: str
    gradient_norm: float | None
    useful: bool
    interior: bool
    valid: bool
    failure_mode: str
    optimizer_success: bool
    optimizer_status: str
    optimizer_iterations: int
    optimizer_evaluations: int
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mechanism": self.mechanism,
            "optimizer": self.optimizer,
            "phase": self.phase,
            "point_index": self.point_index,
            "lambda_multiplier": self.lambda_multiplier,
            "lambda": self.lambda_value,
            "objective_gain_over_zero": self.objective_gain_over_zero,
            "task_loss_gain_over_zero": self.task_loss_gain_over_zero,
            "energy_penalty": self.energy_penalty,
            "energy_mean": self.energy_mean,
            "max_norm_over_cap": self.max_norm_over_cap,
            "mean_norm_over_cap": self.mean_norm_over_cap,
            "cap_bound_fraction": self.cap_bound_fraction,
            "finite_status": self.finite_status,
            "gradient_status": self.gradient_status,
            "gradient_norm": self.gradient_norm,
            "useful": self.useful,
            "interior": self.interior,
            "valid": self.valid,
            "failure_mode": self.failure_mode,
            "optimizer_success": self.optimizer_success,
            "optimizer_status": self.optimizer_status,
            "optimizer_iterations": self.optimizer_iterations,
            "optimizer_evaluations": self.optimizer_evaluations,
            "details": self.details,
        }


@dataclass(frozen=True)
class ThetaCodec:
    mechanism: str
    weight_shape: tuple[int, int, int]
    bias_shape: tuple[int, int] | None
    size: int

    def zeros(self) -> jnp.ndarray:
        return jnp.zeros((self.size,), dtype=jnp.float32)

    def unpack(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        weight_size = math.prod(self.weight_shape)
        weights = theta[:weight_size].reshape(self.weight_shape)
        if self.bias_shape is None:
            return weights, None
        bias = theta[weight_size:].reshape(self.bias_shape)
        return weights, bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="c92ebd8")
    parser.add_argument("--issue", default="1697bdc")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--pgd-steps", type=int, default=8)
    parser.add_argument("--pgd-step-size-fraction", type=float, default=0.25)
    parser.add_argument("--fixed-point-steps", type=int, default=2)
    parser.add_argument("--adam-steps", type=int, default=12)
    parser.add_argument("--adam-learning-rate", type=float, default=5e-5)
    parser.add_argument("--lbfgsb-maxiter", type=int, default=8)
    parser.add_argument("--bisection-rel-tol", type=float, default=1.10)
    parser.add_argument("--max-bisection-steps", type=int, default=8)
    parser.add_argument(
        "--closed-loop-probes",
        type=float,
        nargs="+",
        default=list(DEFAULT_CLOSED_LOOP_PROBES),
    )
    parser.add_argument(
        "--line-search-amplitudes",
        type=float,
        nargs="+",
        default=list(DEFAULT_LINE_SEARCH_AMPLITUDES),
    )
    parser.add_argument(
        "--output-json",
        default="results/1697bdc/critical_lambda_search.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/1697bdc/critical_lambda_search.csv",
    )
    parser.add_argument(
        "--output-md",
        default="results/1697bdc/notes/critical_lambda_search.md",
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
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_csv, payload)
    update_marked_section(output_md, "critical_lambda_search", render_markdown(payload))
    print(
        json.dumps(
            {"json": str(output_json), "csv": str(output_csv), "markdown": str(output_md)},
            indent=2,
        )
    )
    return 0


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_reference_module()
    lambda_source = load_lambda_source(SOURCE_LAMBDA_SWEEP)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run_id in RUN_IDS:
        frozen = reference.load_frozen_batch(args, run_id)
        center_lambda = float(lambda_source[run_id]["center_lambda"])
        direct_points = run_direct_search(reference, frozen, run_id, center_lambda, args)
        all_rows.extend(point.as_dict() for point in direct_points)
        summaries.append(summarize_search(run_id, "direct_epsilon", "pgd_projected_epsilon", direct_points))
        direction_cache = build_line_search_direction_cache(reference, frozen, center_lambda, args)
        for mechanism in CLOSED_LOOP_MECHANISMS:
            for optimizer in CLOSED_LOOP_OPTIMIZERS:
                points = run_closed_loop_search(
                    reference,
                    frozen,
                    run_id,
                    mechanism,
                    optimizer,
                    center_lambda,
                    direction_cache,
                    args,
                )
                all_rows.extend(point.as_dict() for point in points)
                summaries.append(summarize_search(run_id, mechanism, optimizer, points))
    return {
        "schema_version": "rlrmp.critical_lambda_search.v1",
        "issue": str(args.issue),
        "parent_umbrella": "54389a4",
        "source_experiment": str(args.experiment),
        "source_direct_lambda_issue": "093d949",
        "source_closed_loop_reference_issue": "3b850d6",
        "batch_size": int(args.batch_size),
        "replicate_index": int(args.replicate_index),
        "definition": {
            "lambda_crit": (
                "smallest tested lambda multiplier where the optimized adversary is interior "
                "and useful"
            ),
            "interior": "cap_bound_fraction = 0.0 and max_norm_over_cap <= 0.99",
            "useful": (
                "finite objective/gradients and positive objective gain over zero after "
                "the soft-energy penalty"
            ),
            "status": "practical frozen-audit threshold, not an exact analytical H-infinity value",
        },
        "search_contract": {
            "direct_epsilon_brackets": DIRECT_BRACKETS,
            "closed_loop_probe_multipliers": [float(value) for value in args.closed_loop_probes],
            "closed_loop_probe_rationale": (
                "mechanism-specific probes span 0.5x to 8x around the direct-epsilon "
                "per-trial-p90 region before any closed-loop bisection"
            ),
            "bisection_rel_tol": float(args.bisection_rel_tol),
            "max_bisection_steps": int(args.max_bisection_steps),
            "closed_loop_objective": "J(raw_epsilon) - lambda * E(raw_epsilon)",
            "cap_handling": "cap is diagnostic only for closed-loop raw policies",
            "optimizers": {
                "pgd_projected_epsilon": "existing direct-epsilon inner maximizer",
                "line_search_known_direction": (
                    "scalar line search over 3b850d6-style direct-fitted directions; "
                    "reference evidence, not the final closed-loop optimizer"
                ),
                "adam": "bounded full-parameter Adam ascent from zero policy parameters",
                "lbfgsb": "bounded full-parameter SciPy L-BFGS-B ascent from zero policy parameters",
            },
        },
        "rows": all_rows,
        "summary": summaries,
        "overall_interpretation": interpret_overall(summaries),
    }


def load_reference_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "closed_loop_policy_audit_reference_3b850d6",
        REFERENCE_POLICY_AUDIT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reference module at {REFERENCE_POLICY_AUDIT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_lambda_source(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["run_id"]: {
            "center_lambda": float(row["sweep_center"]["lambda"]),
            "first_interior_multiplier": row["transition"]["first_interior_multiplier"],
            "last_cap_dominated_multiplier": row["transition"]["last_cap_dominated_multiplier"],
        }
        for row in payload["rows"]
    }


def run_direct_search(
    reference: Any,
    frozen: Any,
    run_id: str,
    center_lambda: float,
    args: argparse.Namespace,
) -> list[SearchPoint]:
    low, high = DIRECT_BRACKETS[run_id]
    points: list[SearchPoint] = []
    point_index = 0
    low_point = evaluate_direct_point(
        reference, frozen, run_id, center_lambda, low, "bracket", point_index, args
    )
    point_index += 1
    high_point = evaluate_direct_point(
        reference, frozen, run_id, center_lambda, high, "bracket", point_index, args
    )
    point_index += 1
    points.extend([low_point, high_point])
    while not high_point.valid and high < 16.0:
        high *= 2.0
        high_point = evaluate_direct_point(
            reference, frozen, run_id, center_lambda, high, "bracket_expand", point_index, args
        )
        point_index += 1
        points.append(high_point)
    if low_point.valid or not high_point.valid:
        return points
    bisected, _low, _high = log_bisect(
        lambda multiplier, phase, index: evaluate_direct_point(
            reference, frozen, run_id, center_lambda, multiplier, phase, index, args
        ),
        low_point,
        high_point,
        point_index,
        rel_tol=float(args.bisection_rel_tol),
        max_steps=int(args.max_bisection_steps),
    )
    points.extend(bisected)
    return points


def evaluate_direct_point(
    reference: Any,
    frozen: Any,
    run_id: str,
    center_lambda: float,
    multiplier: float,
    phase: str,
    point_index: int,
    args: argparse.Namespace,
) -> SearchPoint:
    lambda_value = center_lambda * float(multiplier)
    delta, diagnostics = reference.direct_epsilon_direction(frozen, lambda_value=lambda_value, args=args)
    summary = reference.direct_summary(delta, frozen.radius, diagnostics=diagnostics)
    gradient_status = "finite" if summary["finite_status"] == "finite" else "nonfinite"
    return make_point(
        run_id=run_id,
        mechanism="direct_epsilon",
        optimizer="pgd_projected_epsilon",
        phase=phase,
        point_index=point_index,
        lambda_multiplier=multiplier,
        lambda_value=lambda_value,
        objective_gain_over_zero=float(summary["penalized_gain_over_zero"]),
        task_loss_gain_over_zero=float(summary["raw_task_loss_gain"]),
        energy_penalty=lambda_value * float(summary["energy_mean"]),
        energy_mean=float(summary["energy_mean"]),
        max_norm_over_cap=float(summary["selected_norm_cap_ratio_max"]),
        mean_norm_over_cap=float(
            summary["selected_epsilon_norm_mean"] / max(float(jnp.mean(frozen.radius)), 1e-12)
        ),
        cap_bound_fraction=float(summary["cap_bound_fraction"]),
        finite_status=str(summary["finite_status"]),
        gradient_status=gradient_status,
        gradient_norm=None,
        optimizer_success=summary["finite_status"] == "finite",
        optimizer_status="direct_epsilon_inner_maximizer",
        optimizer_iterations=int(args.pgd_steps),
        optimizer_evaluations=0,
        details={"diagnostics": plain_json(summary)},
    )


def build_line_search_direction_cache(
    reference: Any,
    frozen: Any,
    center_lambda: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference_multiplier = 4.0
    delta, _diagnostics = reference.direct_epsilon_direction(
        frozen,
        lambda_value=center_lambda * reference_multiplier,
        args=args,
    )
    directions = reference.build_policy_directions(
        frozen,
        direct_delta=delta,
        ridge_alpha=1e-3,
    )
    return {
        direction.mechanism: direction
        for direction in directions
        if direction.label in {"linear_ridge_direct", "affine_mean_direct"}
    }


def run_closed_loop_search(
    reference: Any,
    frozen: Any,
    run_id: str,
    mechanism: str,
    optimizer: str,
    center_lambda: float,
    direction_cache: dict[str, Any],
    args: argparse.Namespace,
) -> list[SearchPoint]:
    evaluator = build_closed_loop_evaluator(
        reference, frozen, run_id, mechanism, optimizer, center_lambda, direction_cache, args
    )
    points: list[SearchPoint] = []
    for index, multiplier in enumerate(sorted(float(value) for value in args.closed_loop_probes)):
        points.append(evaluator(multiplier, "probe", index))
    bracket = find_first_valid_bracket(points)
    if bracket is None:
        return points
    low_point, high_point = bracket
    bisected, _low, _high = log_bisect(
        evaluator,
        low_point,
        high_point,
        len(points),
        rel_tol=float(args.bisection_rel_tol),
        max_steps=int(args.max_bisection_steps),
    )
    points.extend(bisected)
    return points


def build_closed_loop_evaluator(
    reference: Any,
    frozen: Any,
    run_id: str,
    mechanism: str,
    optimizer: str,
    center_lambda: float,
    direction_cache: dict[str, Any],
    args: argparse.Namespace,
) -> Callable[[float, str, int], SearchPoint]:
    if optimizer == "line_search_known_direction":
        direction = direction_cache[mechanism]

        def line_search_evaluator(multiplier: float, phase: str, index: int) -> SearchPoint:
            lambda_value = center_lambda * float(multiplier)
            return evaluate_line_search_point(
                reference, frozen, run_id, direction, lambda_value, multiplier, phase, index, args
            )

        return line_search_evaluator

    codec = make_theta_codec(reference, frozen, mechanism)
    value_and_grad = make_closed_loop_value_and_grad(reference, frozen, codec, args)
    zero_metrics = score_closed_loop_delta(
        reference,
        frozen,
        jnp.zeros_like(frozen.trial_specs.inputs["epsilon"]),
        jnp.asarray(0.0),
    )

    def evaluator(multiplier: float, phase: str, index: int) -> SearchPoint:
        lambda_value = center_lambda * float(multiplier)
        if optimizer == "adam":
            result = optimize_with_adam(codec, value_and_grad, lambda_value, args)
        elif optimizer == "lbfgsb":
            result = optimize_with_lbfgsb(codec, value_and_grad, lambda_value, args)
        else:
            raise ValueError(f"Unknown optimizer {optimizer!r}")
        return summarize_theta_point(
            reference,
            frozen,
            codec,
            run_id,
            mechanism,
            optimizer,
            phase,
            index,
            lambda_value,
            multiplier,
            result,
            zero_metrics,
            args,
        )

    return evaluator


def make_theta_codec(reference: Any, frozen: Any, mechanism: str) -> ThetaCodec:
    zero = jnp.zeros_like(frozen.trial_specs.inputs["epsilon"])
    features = reference.live_features(frozen, zero)
    time_steps = int(features.shape[1])
    feature_dim = int(features.shape[2])
    epsilon_dim = int(zero.shape[2])
    weight_shape = (time_steps, epsilon_dim, feature_dim)
    if mechanism == "linear_no_bias":
        return ThetaCodec(mechanism=mechanism, weight_shape=weight_shape, bias_shape=None, size=math.prod(weight_shape))
    if mechanism == "affine":
        bias_shape = (time_steps, epsilon_dim)
        return ThetaCodec(
            mechanism=mechanism,
            weight_shape=weight_shape,
            bias_shape=bias_shape,
            size=math.prod(weight_shape) + math.prod(bias_shape),
        )
    raise ValueError(f"Unknown closed-loop mechanism {mechanism!r}")


def make_closed_loop_value_and_grad(
    reference: Any,
    frozen: Any,
    codec: ThetaCodec,
    args: argparse.Namespace,
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    def objective(theta: jnp.ndarray, lambda_value: jnp.ndarray) -> jnp.ndarray:
        delta = closed_loop_raw_delta(reference, frozen, codec, theta, int(args.fixed_point_steps))
        return score_closed_loop_delta(reference, frozen, delta, lambda_value)["objective"]

    return jax.jit(jax.value_and_grad(objective, argnums=0))


def closed_loop_raw_delta(
    reference: Any,
    frozen: Any,
    codec: ThetaCodec,
    theta: jnp.ndarray,
    fixed_point_steps: int,
) -> jnp.ndarray:
    weights, bias = codec.unpack(theta)
    delta = jnp.zeros_like(frozen.trial_specs.inputs["epsilon"])
    for _ in range(int(fixed_point_steps)):
        features = reference.live_features(frozen, delta)
        raw = jnp.einsum("btf,tef->bte", features, weights)
        if bias is not None:
            raw = raw + bias[None, :, :]
        delta = raw * frozen.time_mask
    return delta * frozen.time_mask


def score_closed_loop_delta(
    reference: Any,
    frozen: Any,
    delta: jnp.ndarray,
    lambda_value: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    candidate = reference._set_input(frozen.trial_specs, "epsilon", epsilon + delta * frozen.time_mask)
    states = frozen.task.eval_trials(frozen.model, candidate, frozen.keys_model)
    task_loss_value = jnp.asarray(frozen.task.loss_func(states, candidate, frozen.model).total)
    energy = jnp.mean(per_trial_energy(delta * frozen.time_mask))
    objective = task_loss_value - lambda_value * energy
    return {"task_loss": task_loss_value, "energy": energy, "objective": objective}


def optimize_with_adam(
    codec: ThetaCodec,
    value_and_grad: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    lambda_value: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    theta = codec.zeros()
    optimizer = optax.adam(float(args.adam_learning_rate))
    opt_state = optimizer.init(theta)
    best_theta = theta
    best_value = -jnp.inf
    all_finite = True
    final_grad = jnp.zeros_like(theta)
    lam = jnp.asarray(lambda_value, dtype=jnp.float64)
    for step in range(int(args.adam_steps)):
        value, grad = value_and_grad(theta, lam)
        finite = bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(grad)))
        all_finite = all_finite and finite
        if finite and float(value) > float(best_value):
            best_value = value
            best_theta = theta
        updates, opt_state = optimizer.update(jax.tree.map(lambda item: -item, grad), opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        final_grad = grad
        if not finite:
            break
    return {
        "theta": best_theta,
        "value": best_value,
        "grad": final_grad,
        "success": all_finite,
        "status": "adam_all_finite" if all_finite else "adam_nonfinite_seen",
        "iterations": int(args.adam_steps),
        "evaluations": int(args.adam_steps),
    }


def optimize_with_lbfgsb(
    codec: ThetaCodec,
    value_and_grad: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    lambda_value: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        import scipy.optimize as scipy_opt
    except ImportError:
        theta = codec.zeros()
        value, grad = value_and_grad(theta, jnp.asarray(lambda_value, dtype=jnp.float64))
        return {
            "theta": theta,
            "value": value,
            "grad": grad,
            "success": False,
            "status": "scipy_unavailable",
            "iterations": 0,
            "evaluations": 1,
        }

    lam = jnp.asarray(lambda_value, dtype=jnp.float64)

    def scipy_value_and_grad(theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        theta = jnp.asarray(theta_np, dtype=jnp.float32)
        value, grad = value_and_grad(theta, lam)
        return -float(value), -np.asarray(grad, dtype=np.float64)

    theta0 = np.zeros((codec.size,), dtype=np.float64)
    result = scipy_opt.minimize(
        scipy_value_and_grad,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(args.lbfgsb_maxiter), "maxls": 10},
    )
    theta = jnp.asarray(result.x, dtype=jnp.float32)
    value, grad = value_and_grad(theta, lam)
    return {
        "theta": theta,
        "value": value,
        "grad": grad,
        "success": bool(result.success) or np.isfinite(float(value)),
        "status": str(result.message),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
    }


def summarize_theta_point(
    reference: Any,
    frozen: Any,
    codec: ThetaCodec,
    run_id: str,
    mechanism: str,
    optimizer: str,
    phase: str,
    index: int,
    lambda_value: float,
    multiplier: float,
    result: dict[str, Any],
    zero_metrics_at_zero_lambda: dict[str, jnp.ndarray],
    args: argparse.Namespace,
) -> SearchPoint:
    del zero_metrics_at_zero_lambda
    lam = jnp.asarray(lambda_value, dtype=jnp.float64)
    theta = jnp.asarray(result["theta"])
    delta = closed_loop_raw_delta(reference, frozen, codec, theta, int(args.fixed_point_steps))
    metrics = score_closed_loop_delta(reference, frozen, delta, lam)
    zero = jnp.zeros_like(delta)
    zero_metrics = score_closed_loop_delta(reference, frozen, zero, lam)
    grad = jnp.asarray(result["grad"])
    gradient_norm = float(jnp.linalg.norm(grad))
    return point_from_delta(
        reference=reference,
        frozen=frozen,
        run_id=run_id,
        mechanism=mechanism,
        optimizer=optimizer,
        phase=phase,
        point_index=index,
        lambda_value=lambda_value,
        multiplier=multiplier,
        delta=delta,
        metrics=metrics,
        zero_metrics=zero_metrics,
        gradient_norm=gradient_norm,
        gradient_status="finite" if bool(jnp.all(jnp.isfinite(grad))) else "nonfinite",
        optimizer_success=bool(result["success"]),
        optimizer_status=str(result["status"]),
        optimizer_iterations=int(result["iterations"]),
        optimizer_evaluations=int(result["evaluations"]),
        details={"theta_size": codec.size, "fixed_point_steps": int(args.fixed_point_steps)},
    )


def evaluate_line_search_point(
    reference: Any,
    frozen: Any,
    run_id: str,
    direction: Any,
    lambda_value: float,
    multiplier: float,
    phase: str,
    index: int,
    args: argparse.Namespace,
) -> SearchPoint:
    zero = jnp.zeros_like(frozen.trial_specs.inputs["epsilon"])
    zero_metrics = score_closed_loop_delta(reference, frozen, zero, jnp.asarray(lambda_value))
    best: tuple[float, jnp.ndarray, dict[str, jnp.ndarray]] | None = None
    for amplitude in args.line_search_amplitudes:
        delta = reference.policy_raw_delta(
            frozen,
            direction.params,
            mechanism=direction.mechanism,
            amplitude=float(amplitude),
            fixed_point_steps=int(args.fixed_point_steps),
        )
        metrics = score_closed_loop_delta(reference, frozen, delta, jnp.asarray(lambda_value))
        objective_gain = float(metrics["objective"] - zero_metrics["objective"])
        if best is None or objective_gain > best[0]:
            best = (float(amplitude), delta, metrics)
    assert best is not None
    amplitude, delta, metrics = best
    gradient_norm = line_search_gradient_norm(
        reference,
        frozen,
        direction,
        lambda_value,
        amplitude,
        int(args.fixed_point_steps),
    )
    return point_from_delta(
        reference=reference,
        frozen=frozen,
        run_id=run_id,
        mechanism=direction.mechanism,
        optimizer="line_search_known_direction",
        phase=phase,
        point_index=index,
        lambda_value=lambda_value,
        multiplier=multiplier,
        delta=delta,
        metrics=metrics,
        zero_metrics=zero_metrics,
        gradient_norm=gradient_norm,
        gradient_status="finite" if np.isfinite(gradient_norm) else "nonfinite",
        optimizer_success=np.isfinite(float(metrics["objective"])),
        optimizer_status="best_grid_amplitude",
        optimizer_iterations=len(args.line_search_amplitudes),
        optimizer_evaluations=len(args.line_search_amplitudes),
        details={
            "direction": direction.label,
            "best_amplitude": amplitude,
            "fit_diagnostics": plain_json(direction.fit_diagnostics),
            "reference_only": True,
        },
    )


def line_search_gradient_norm(
    reference: Any,
    frozen: Any,
    direction: Any,
    lambda_value: float,
    amplitude: float,
    fixed_point_steps: int,
) -> float:
    def scalar_objective(scale: jnp.ndarray) -> jnp.ndarray:
        delta = known_direction_raw_delta(
            reference, frozen, direction, scale, fixed_point_steps
        )
        return score_closed_loop_delta(reference, frozen, delta, jnp.asarray(lambda_value))["objective"]

    grad = jax.grad(scalar_objective)(jnp.asarray(amplitude, dtype=jnp.float32))
    return float(jnp.abs(grad))


def known_direction_raw_delta(
    reference: Any,
    frozen: Any,
    direction: Any,
    amplitude: jnp.ndarray,
    fixed_point_steps: int,
) -> jnp.ndarray:
    delta = jnp.zeros_like(frozen.trial_specs.inputs["epsilon"])
    for _ in range(int(fixed_point_steps)):
        features = reference.live_features(frozen, delta)
        if direction.mechanism == "affine":
            weights, bias = direction.params
            raw = jnp.einsum("btf,tef->bte", features, weights) + bias[None, :, :]
        elif direction.mechanism == "linear_no_bias":
            raw = jnp.einsum("btf,tef->bte", features, direction.params)
        else:
            raise ValueError(f"Unknown mechanism {direction.mechanism!r}")
        delta = amplitude * raw * frozen.time_mask
    return delta * frozen.time_mask


def point_from_delta(
    *,
    reference: Any,
    frozen: Any,
    run_id: str,
    mechanism: str,
    optimizer: str,
    phase: str,
    point_index: int,
    lambda_value: float,
    multiplier: float,
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
) -> SearchPoint:
    norm = reference._flattened_per_trial_norm(delta * frozen.time_mask)
    ratio = norm / jnp.maximum(frozen.radius, 1e-12)
    energy = per_trial_energy(delta * frozen.time_mask)
    task_gain = float(metrics["task_loss"] - zero_metrics["task_loss"])
    objective_gain = float(metrics["objective"] - zero_metrics["objective"])
    energy_penalty = float(jnp.asarray(lambda_value) * jnp.mean(energy))
    finite = bool(
        jnp.all(
            jnp.isfinite(
                jnp.asarray(
                    [
                        metrics["task_loss"],
                        metrics["energy"],
                        metrics["objective"],
                        zero_metrics["objective"],
                    ]
                )
            )
        )
    )
    finite_status = "finite" if finite else "nonfinite"
    return make_point(
        run_id=run_id,
        mechanism=mechanism,
        optimizer=optimizer,
        phase=phase,
        point_index=point_index,
        lambda_multiplier=multiplier,
        lambda_value=lambda_value,
        objective_gain_over_zero=objective_gain,
        task_loss_gain_over_zero=task_gain,
        energy_penalty=energy_penalty,
        energy_mean=float(jnp.mean(energy)),
        max_norm_over_cap=float(jnp.max(ratio)),
        mean_norm_over_cap=float(jnp.mean(ratio)),
        cap_bound_fraction=float(jnp.mean((ratio >= 1.0 - 1e-4).astype(jnp.float32))),
        finite_status=finite_status,
        gradient_status=gradient_status,
        gradient_norm=gradient_norm,
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        optimizer_iterations=optimizer_iterations,
        optimizer_evaluations=optimizer_evaluations,
        details=details,
    )


def make_point(
    *,
    run_id: str,
    mechanism: str,
    optimizer: str,
    phase: str,
    point_index: int,
    lambda_multiplier: float,
    lambda_value: float,
    objective_gain_over_zero: float,
    task_loss_gain_over_zero: float,
    energy_penalty: float,
    energy_mean: float,
    max_norm_over_cap: float,
    mean_norm_over_cap: float,
    cap_bound_fraction: float,
    finite_status: str,
    gradient_status: str,
    gradient_norm: float | None,
    optimizer_success: bool,
    optimizer_status: str,
    optimizer_iterations: int,
    optimizer_evaluations: int,
    details: dict[str, Any],
) -> SearchPoint:
    useful = (
        finite_status == "finite"
        and gradient_status == "finite"
        and bool(np.isfinite(objective_gain_over_zero))
        and objective_gain_over_zero > 0.0
    )
    interior = cap_bound_fraction == 0.0 and max_norm_over_cap <= 0.99
    valid = useful and interior
    return SearchPoint(
        run_id=run_id,
        mechanism=mechanism,
        optimizer=optimizer,
        phase=phase,
        point_index=point_index,
        lambda_multiplier=float(lambda_multiplier),
        lambda_value=float(lambda_value),
        objective_gain_over_zero=float(objective_gain_over_zero),
        task_loss_gain_over_zero=float(task_loss_gain_over_zero),
        energy_penalty=float(energy_penalty),
        energy_mean=float(energy_mean),
        max_norm_over_cap=float(max_norm_over_cap),
        mean_norm_over_cap=float(mean_norm_over_cap),
        cap_bound_fraction=float(cap_bound_fraction),
        finite_status=finite_status,
        gradient_status=gradient_status,
        gradient_norm=None if gradient_norm is None else float(gradient_norm),
        useful=useful,
        interior=interior,
        valid=valid,
        failure_mode=classify_failure(
            useful=useful,
            interior=interior,
            finite_status=finite_status,
            gradient_status=gradient_status,
            objective_gain=objective_gain_over_zero,
            cap_bound_fraction=cap_bound_fraction,
            max_norm_over_cap=max_norm_over_cap,
        ),
        optimizer_success=bool(optimizer_success),
        optimizer_status=optimizer_status,
        optimizer_iterations=int(optimizer_iterations),
        optimizer_evaluations=int(optimizer_evaluations),
        details=plain_json(details),
    )


def classify_failure(
    *,
    useful: bool,
    interior: bool,
    finite_status: str,
    gradient_status: str,
    objective_gain: float,
    cap_bound_fraction: float,
    max_norm_over_cap: float,
) -> str:
    if useful and interior:
        return "valid"
    if finite_status != "finite" or gradient_status != "finite":
        return "nonfinite"
    if objective_gain <= 0.0:
        return "not_useful"
    if cap_bound_fraction > 0.0:
        return "cap_bound"
    if max_norm_over_cap > 0.99:
        return "near_cap"
    return "invalid_unspecified"


def find_first_valid_bracket(points: list[SearchPoint]) -> tuple[SearchPoint, SearchPoint] | None:
    ordered = sorted(points, key=lambda point: point.lambda_multiplier)
    for lower, upper in zip(ordered, ordered[1:], strict=False):
        if not lower.valid and upper.valid:
            return lower, upper
    return None


def log_bisect(
    evaluator: Callable[[float, str, int], SearchPoint],
    low_point: SearchPoint,
    high_point: SearchPoint,
    start_index: int,
    *,
    rel_tol: float,
    max_steps: int,
) -> tuple[list[SearchPoint], SearchPoint, SearchPoint]:
    points = []
    low = low_point
    high = high_point
    index = int(start_index)
    for _step in range(int(max_steps)):
        if high.lambda_multiplier / max(low.lambda_multiplier, 1e-12) <= float(rel_tol):
            break
        midpoint = math.sqrt(low.lambda_multiplier * high.lambda_multiplier)
        point = evaluator(midpoint, "bisection", index)
        index += 1
        points.append(point)
        if point.valid:
            high = point
        else:
            low = point
    return points, low, high


def summarize_search(
    run_id: str,
    mechanism: str,
    optimizer: str,
    points: list[SearchPoint],
) -> dict[str, Any]:
    valid_points = [point for point in points if point.valid]
    invalid_below_valid = find_first_valid_bracket(points)
    lowest_valid = min(valid_points, key=lambda point: point.lambda_multiplier) if valid_points else None
    probe_points = [point for point in points if point.phase == "probe"]
    bracket_found = invalid_below_valid is not None
    failure_mode = summarize_failure_mode(points, probe_points, lowest_valid, bracket_found)
    reliability = optimizer_reliability(points, lowest_valid)
    return {
        "run_id": run_id,
        "mechanism": mechanism,
        "optimizer": optimizer,
        "bracket_found": bracket_found,
        "lowest_valid_lambda_multiplier": None if lowest_valid is None else lowest_valid.lambda_multiplier,
        "lowest_valid_lambda": None if lowest_valid is None else lowest_valid.lambda_value,
        "gain_at_lowest_valid": None if lowest_valid is None else lowest_valid.objective_gain_over_zero,
        "max_norm_over_cap_at_lowest_valid": None if lowest_valid is None else lowest_valid.max_norm_over_cap,
        "cap_bound_fraction_at_lowest_valid": None if lowest_valid is None else lowest_valid.cap_bound_fraction,
        "finite_status_at_lowest_valid": None if lowest_valid is None else lowest_valid.finite_status,
        "optimizer_reliability": reliability,
        "failure_mode": failure_mode,
        "n_points": len(points),
        "n_valid_points": len(valid_points),
        "bracket_low_multiplier": None if invalid_below_valid is None else invalid_below_valid[0].lambda_multiplier,
        "bracket_high_multiplier": None if invalid_below_valid is None else invalid_below_valid[1].lambda_multiplier,
    }


def summarize_failure_mode(
    points: list[SearchPoint],
    probe_points: list[SearchPoint],
    lowest_valid: SearchPoint | None,
    bracket_found: bool,
) -> str:
    if lowest_valid is not None:
        if not bracket_found and probe_points and min(
            probe_points,
            key=lambda point: point.lambda_multiplier,
        ).valid:
            return "valid_at_lowest_probe_threshold_below_range"
        if not bracket_found:
            return "valid_without_clean_lower_invalid_bracket"
        return "bracketed_or_valid"
    if not points:
        return "not_run"
    if all(point.failure_mode in {"cap_bound", "near_cap"} for point in points):
        return "likely_above_probe_range"
    if all(point.failure_mode == "not_useful" for point in points):
        return "optimizer_or_parameterization_not_useful"
    if any(point.failure_mode == "nonfinite" for point in points):
        return "nonfinite_optimizer_failure"
    if probe_points and probe_points[0].valid:
        return "valid_at_lowest_probe_threshold_below_range"
    return "mixed_invalid_or_nonmonotone"


def optimizer_reliability(points: list[SearchPoint], lowest_valid: SearchPoint | None) -> str:
    if any(point.finite_status != "finite" or point.gradient_status != "finite" for point in points):
        return "nonfinite_seen"
    if any(not point.optimizer_success for point in points):
        return "optimizer_reported_failure_but_values_materialized"
    if lowest_valid is None:
        return "no_valid_point"
    if lowest_valid.optimizer == "line_search_known_direction":
        return "reference_direction_only"
    return "bounded_optimizer_materialized_valid_point"


def interpret_overall(summaries: list[dict[str, Any]]) -> str:
    direct = [row for row in summaries if row["mechanism"] == "direct_epsilon"]
    closed = [row for row in summaries if row["mechanism"] != "direct_epsilon"]
    direct_ok = sum(1 for row in direct if row["lowest_valid_lambda_multiplier"] is not None)
    closed_ok = sum(1 for row in closed if row["lowest_valid_lambda_multiplier"] is not None)
    return (
        f"Direct epsilon produced practical lambda_crit estimates on {direct_ok}/{len(direct)} "
        f"substrate rows. Closed-loop optimizer/mechanism searches produced valid practical "
        f"points on {closed_ok}/{len(closed)} optimizer-specific rows; missing brackets are "
        "reported as outside-range, nonfinite, or optimizer/parameterization failures rather "
        "than interpreted as exact critical values."
    )


def per_trial_energy(delta: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(delta), axis=tuple(range(1, delta.ndim)))


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
    fields = [
        "run_id",
        "mechanism",
        "optimizer",
        "phase",
        "point_index",
        "lambda_multiplier",
        "lambda",
        "objective_gain_over_zero",
        "task_loss_gain_over_zero",
        "energy_penalty",
        "energy_mean",
        "max_norm_over_cap",
        "mean_norm_over_cap",
        "cap_bound_fraction",
        "finite_status",
        "gradient_status",
        "gradient_norm",
        "useful",
        "interior",
        "valid",
        "failure_mode",
        "optimizer_success",
        "optimizer_status",
        "optimizer_iterations",
        "optimizer_evaluations",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({key: row[key] for key in fields})


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Critical lambda frozen adversary search",
        "",
        f"Issue: `{payload['issue']}`. Source frozen no-PGD runs: `c92ebd8`.",
        "",
        "This audit defines practical `lambda_crit` as the smallest tested lambda multiplier "
        "where the optimized adversary is both interior and useful. Interior means "
        "`cap_bound_fraction = 0.0` and `max_norm_over_cap <= 0.99`; useful means finite "
        "objective/gradients and positive soft-energy objective gain over zero. It is not an "
        "analytical H-infinity threshold.",
        "",
        "Closed-loop policy rows optimize the raw objective `J(raw_epsilon) - lambda * "
        "E(raw_epsilon)`. Cap behavior is computed afterward as a diagnostic.",
        "",
        "## Summary",
        "",
        "| row | mechanism | optimizer | bracket | lowest valid multiplier | gain | max norm/cap | cap-bound | finite | reliability | failure mode |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["summary"]:
        lowest = row["lowest_valid_lambda_multiplier"]
        gain = row["gain_at_lowest_valid"]
        max_ratio = row["max_norm_over_cap_at_lowest_valid"]
        cap = row["cap_bound_fraction_at_lowest_valid"]
        finite = row["finite_status_at_lowest_valid"] or "n/a"
        lines.append(
            f"| `{row['run_id']}` | `{row['mechanism']}` | `{row['optimizer']}` | "
            f"{'yes' if row['bracket_found'] else 'no'} | "
            f"{format_optional(lowest)} | {format_optional(gain)} | "
            f"{format_optional(max_ratio)} | {format_optional_percent(cap)} | "
            f"{finite} | {row['optimizer_reliability']} | {row['failure_mode']} |"
        )
    lines.extend(
        [
            "",
            "## Probe and bisection rows",
            "",
            "| row | mechanism | optimizer | phase | multiplier | gain | max norm/cap | cap-bound | useful | interior | valid | failure |",
            "|---|---|---|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| `{row['run_id']}` | `{row['mechanism']}` | `{row['optimizer']}` | "
            f"{row['phase']} | {row['lambda_multiplier']:.6g} | "
            f"{row['objective_gain_over_zero']:.6g} | {row['max_norm_over_cap']:.6g} | "
            f"{row['cap_bound_fraction']:.3%} | {row['useful']} | {row['interior']} | "
            f"{row['valid']} | {row['failure_mode']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["overall_interpretation"],
            "",
            "Rows without a valid bracket are deliberately not assigned an exact lambda. "
            "`likely_above_probe_range` means useful adversaries remained too close to or over "
            "the cap through the tested range. `optimizer_or_parameterization_not_useful` means "
            "the optimizer did not find positive soft-energy gain over zero. "
            "`line_search_known_direction` is included as a reference optimizer comparison; the "
            "full-parameter Adam and L-BFGS-B rows are the closed-loop optimization rows.",
            "",
        ]
    )
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def format_optional_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3%}"


if __name__ == "__main__":
    raise SystemExit(main())
