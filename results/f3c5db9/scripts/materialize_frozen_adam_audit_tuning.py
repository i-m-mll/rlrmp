"""Materialize frozen-batch Adam reliability tuning for f3c5db9."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

from rlrmp.io import update_marked_section
from rlrmp.paths import mkdir_p


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SEARCH = REPO_ROOT / "results" / "1697bdc" / "scripts" / "materialize_critical_lambda_search.py"
SOURCE_REFERENCE_JSON = REPO_ROOT / "results" / "1697bdc" / "critical_lambda_search.json"
SOURCE_LAMBDA_SWEEP = REPO_ROOT / "results" / "093d949" / "soft_lambda_sweep.json"
REFERENCE_POLICY_AUDIT = (
    REPO_ROOT / "results" / "3b850d6" / "scripts" / "materialize_closed_loop_policy_audit.py"
)
RUN_IDS = ("open_loop_small", "open_loop_moderate", "open_loop_stress")
MECHANISMS = ("direct_epsilon", "linear_no_bias", "affine")
REFERENCE_OPTIMIZER_BY_MECHANISM = {
    "direct_epsilon": "pgd_projected_epsilon",
    "linear_no_bias": "lbfgsb",
    "affine": "lbfgsb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="c92ebd8")
    parser.add_argument("--issue", default="f3c5db9")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--pgd-steps", type=int, default=8)
    parser.add_argument("--pgd-step-size-fraction", type=float, default=0.25)
    parser.add_argument("--fixed-point-steps", type=int, default=2)
    parser.add_argument("--stage1-steps", type=int, nargs="+", default=[12, 32, 64, 128])
    parser.add_argument(
        "--stage1-learning-rates",
        type=float,
        nargs="+",
        default=[1e-5, 3e-5, 1e-4, 3e-4],
    )
    parser.add_argument("--stage2-steps", type=int, default=128)
    parser.add_argument("--stage2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument(
        "--output-json",
        default="results/f3c5db9/frozen_adam_audit_tuning.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/f3c5db9/frozen_adam_audit_tuning.csv",
    )
    parser.add_argument(
        "--output-md",
        default="results/f3c5db9/notes/frozen_adam_audit_tuning.md",
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
    write_csv(output_csv, payload["rows"])
    update_marked_section(output_md, "frozen_adam_audit_tuning", render_markdown(payload))
    print(
        json.dumps(
            {"json": str(output_json), "csv": str(output_csv), "markdown": str(output_md)},
            indent=2,
        )
    )
    return 0


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    search = load_search_module()
    reference = search.load_reference_module()
    lambda_source = search.load_lambda_source(SOURCE_LAMBDA_SWEEP)
    reference_by_key = load_reference_summary(SOURCE_REFERENCE_JSON)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run_id in RUN_IDS:
        frozen = reference.load_frozen_batch(args, run_id)
        center_lambda = float(lambda_source[run_id]["center_lambda"])
        direction_cache = search.build_line_search_direction_cache(reference, frozen, center_lambda, args)
        for mechanism in MECHANISMS:
            reference_row = reference_by_key[(run_id, mechanism)]
            stage1_rows = evaluate_stage1(
                search, reference, frozen, direction_cache, run_id, mechanism, reference_row, args
            )
            rows = list(stage1_rows)
            if not args.skip_stage2 and not any(row.valid for row in stage1_rows):
                rows.extend(
                    evaluate_stage2(
                        search,
                        reference,
                        frozen,
                        direction_cache,
                        run_id,
                        mechanism,
                        reference_row,
                        args,
                    )
                )
            row_dicts = [serialize_row(row, reference_row) for row in rows]
            all_rows.extend(row_dicts)
            summaries.append(summarize_tuning(run_id, mechanism, reference_row, row_dicts))
    common_settings = common_stage1_settings(all_rows)
    return {
        "schema_version": "rlrmp.frozen_adam_audit_tuning.v1",
        "issue": str(args.issue),
        "parent_umbrella": "54389a4",
        "source_experiment": str(args.experiment),
        "source_reference_issue": "1697bdc",
        "source_direct_lambda_issue": "093d949",
        "source_closed_loop_reference_issue": "3b850d6",
        "batch_size": int(args.batch_size),
        "replicate_index": int(args.replicate_index),
        "stage1_grid": {
            "steps": [int(value) for value in args.stage1_steps],
            "learning_rates": [float(value) for value in args.stage1_learning_rates],
        },
        "stage2": {
            "enabled": not bool(args.skip_stage2),
            "kind": "known_reference_direction_initialization",
            "steps": int(args.stage2_steps),
            "learning_rate": float(args.stage2_learning_rate),
        },
        "match_definition": (
            "Adam matches when it finds a finite, useful, interior point at the 1697bdc "
            "reference lambda multiplier for the same row and mechanism."
        ),
        "rows": all_rows,
        "summary": summaries,
        "common_stage1_matching_settings": common_settings,
        "recommendation": recommend(summaries, common_settings),
    }


def load_search_module() -> Any:
    spec = importlib.util.spec_from_file_location("critical_lambda_search_1697bdc", SOURCE_SEARCH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SOURCE_SEARCH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.REPO_ROOT = REPO_ROOT
    module.SOURCE_LAMBDA_SWEEP = SOURCE_LAMBDA_SWEEP
    module.REFERENCE_POLICY_AUDIT = REFERENCE_POLICY_AUDIT
    return module


def load_reference_summary(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload["summary"]:
        mechanism = str(row["mechanism"])
        if row["optimizer"] != REFERENCE_OPTIMIZER_BY_MECHANISM.get(mechanism):
            continue
        if row["lowest_valid_lambda_multiplier"] is None:
            raise ValueError(f"Reference row lacks a valid lambda: {row}")
        rows[(str(row["run_id"]), mechanism)] = row
    missing = [(run_id, mechanism) for run_id in RUN_IDS for mechanism in MECHANISMS]
    missing = [key for key in missing if key not in rows]
    if missing:
        raise ValueError(f"Missing reference summary rows: {missing}")
    return rows


def evaluate_stage1(
    search: Any,
    reference: Any,
    frozen: Any,
    direction_cache: dict[str, Any],
    run_id: str,
    mechanism: str,
    reference_row: dict[str, Any],
    args: argparse.Namespace,
) -> list[Any]:
    rows = []
    multiplier = float(reference_row["lowest_valid_lambda_multiplier"])
    lambda_value = float(reference_row["lowest_valid_lambda"])
    for steps in args.stage1_steps:
        for learning_rate in args.stage1_learning_rates:
            rows.append(
                evaluate_adam_point(
                    search,
                    reference,
                    frozen,
                    direction_cache,
                    run_id,
                    mechanism,
                    multiplier,
                    lambda_value,
                    stage="stage1_grid",
                    point_index=len(rows),
                    steps=int(steps),
                    learning_rate=float(learning_rate),
                    initial_theta=None,
                    initialization="zero",
                    args=args,
                )
            )
    return rows


def evaluate_stage2(
    search: Any,
    reference: Any,
    frozen: Any,
    direction_cache: dict[str, Any],
    run_id: str,
    mechanism: str,
    reference_row: dict[str, Any],
    args: argparse.Namespace,
) -> list[Any]:
    multiplier = float(reference_row["lowest_valid_lambda_multiplier"])
    lambda_value = float(reference_row["lowest_valid_lambda"])
    initial_theta, initialization = known_reference_initialization(
        search, reference, frozen, direction_cache, mechanism, lambda_value, multiplier, args
    )
    point = evaluate_adam_point(
        search,
        reference,
        frozen,
        direction_cache,
        run_id,
        mechanism,
        multiplier,
        lambda_value,
        stage="stage2_known_direction_init",
        point_index=0,
        steps=int(args.stage2_steps),
        learning_rate=float(args.stage2_learning_rate),
        initial_theta=initial_theta,
        initialization=initialization,
        args=args,
    )
    return [point]


def evaluate_adam_point(
    search: Any,
    reference: Any,
    frozen: Any,
    direction_cache: dict[str, Any],
    run_id: str,
    mechanism: str,
    multiplier: float,
    lambda_value: float,
    *,
    stage: str,
    point_index: int,
    steps: int,
    learning_rate: float,
    initial_theta: Any,
    initialization: str,
    args: argparse.Namespace,
) -> Any:
    if mechanism == "direct_epsilon":
        value_and_grad = make_direct_value_and_grad(search, reference, frozen)
        zero_metrics = search.score_closed_loop_delta(
            reference,
            frozen,
            jnp.zeros_like(frozen.trial_specs.inputs["epsilon"]),
            jnp.asarray(lambda_value, dtype=jnp.float64),
        )
        theta0 = (
            jnp.zeros_like(frozen.trial_specs.inputs["epsilon"])
            if initial_theta is None
            else jnp.asarray(initial_theta, dtype=jnp.float32)
        )
        result = optimize_with_adam(value_and_grad, theta0, lambda_value, steps, learning_rate)
        delta = jnp.asarray(result["theta"]) * frozen.time_mask
        metrics = search.score_closed_loop_delta(reference, frozen, delta, jnp.asarray(lambda_value))
        return search.point_from_delta(
            reference=reference,
            frozen=frozen,
            run_id=run_id,
            mechanism=mechanism,
            optimizer="adam",
            phase=stage,
            point_index=point_index,
            lambda_value=lambda_value,
            multiplier=multiplier,
            delta=delta,
            metrics=metrics,
            zero_metrics=zero_metrics,
            gradient_norm=float(jnp.linalg.norm(jnp.asarray(result["grad"]))),
            gradient_status="finite" if bool(jnp.all(jnp.isfinite(result["grad"]))) else "nonfinite",
            optimizer_success=bool(result["success"]),
            optimizer_status=str(result["status"]),
            optimizer_iterations=int(result["iterations"]),
            optimizer_evaluations=int(result["evaluations"]),
            details={
                "adam_steps": steps,
                "adam_learning_rate": learning_rate,
                "initialization": initialization,
                "stage": stage,
            },
        )

    del direction_cache
    codec = search.make_theta_codec(reference, frozen, mechanism)
    value_and_grad = search.make_closed_loop_value_and_grad(reference, frozen, codec, args)
    theta0 = codec.zeros() if initial_theta is None else jnp.asarray(initial_theta, dtype=jnp.float32)
    result = optimize_with_adam(value_and_grad, theta0, lambda_value, steps, learning_rate)
    point = search.summarize_theta_point(
        reference,
        frozen,
        codec,
        run_id,
        mechanism,
        "adam",
        stage,
        point_index,
        lambda_value,
        multiplier,
        result,
        {},
        args,
    )
    details = dict(point.details)
    details.update(
        {
            "adam_steps": steps,
            "adam_learning_rate": learning_rate,
            "initialization": initialization,
            "stage": stage,
        }
    )
    return search.make_point(
        run_id=point.run_id,
        mechanism=point.mechanism,
        optimizer=point.optimizer,
        phase=point.phase,
        point_index=point.point_index,
        lambda_multiplier=point.lambda_multiplier,
        lambda_value=point.lambda_value,
        objective_gain_over_zero=point.objective_gain_over_zero,
        task_loss_gain_over_zero=point.task_loss_gain_over_zero,
        energy_penalty=point.energy_penalty,
        energy_mean=point.energy_mean,
        max_norm_over_cap=point.max_norm_over_cap,
        mean_norm_over_cap=point.mean_norm_over_cap,
        cap_bound_fraction=point.cap_bound_fraction,
        finite_status=point.finite_status,
        gradient_status=point.gradient_status,
        gradient_norm=point.gradient_norm,
        optimizer_success=point.optimizer_success,
        optimizer_status=point.optimizer_status,
        optimizer_iterations=point.optimizer_iterations,
        optimizer_evaluations=point.optimizer_evaluations,
        details=details,
    )


def make_direct_value_and_grad(search: Any, reference: Any, frozen: Any) -> Callable:
    def objective(theta: jnp.ndarray, lambda_value: jnp.ndarray) -> jnp.ndarray:
        delta = theta * frozen.time_mask
        return search.score_closed_loop_delta(reference, frozen, delta, lambda_value)["objective"]

    return jax.jit(jax.value_and_grad(objective, argnums=0))


def optimize_with_adam(
    value_and_grad: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    initial_theta: jnp.ndarray,
    lambda_value: float,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    theta = jnp.asarray(initial_theta, dtype=jnp.float32)
    optimizer = optax.adam(float(learning_rate))
    opt_state = optimizer.init(theta)
    best_theta = theta
    best_value = -jnp.inf
    all_finite = True
    final_grad = jnp.zeros_like(theta)
    lam = jnp.asarray(lambda_value, dtype=jnp.float64)
    for _step in range(int(steps)):
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
        "iterations": int(steps),
        "evaluations": int(steps),
    }


def known_reference_initialization(
    search: Any,
    reference: Any,
    frozen: Any,
    direction_cache: dict[str, Any],
    mechanism: str,
    lambda_value: float,
    multiplier: float,
    args: argparse.Namespace,
) -> tuple[jnp.ndarray, str]:
    if mechanism == "direct_epsilon":
        delta, _diagnostics = reference.direct_epsilon_direction(
            frozen,
            lambda_value=lambda_value,
            args=args,
        )
        return jnp.asarray(delta, dtype=jnp.float32), "direct_epsilon_pgd_reference"

    direction = direction_cache[mechanism]
    point = search.evaluate_line_search_point(
        reference,
        frozen,
        run_id="stage2_init",
        direction=direction,
        lambda_value=lambda_value,
        multiplier=multiplier,
        phase="stage2_init_probe",
        index=0,
        args=args,
    )
    amplitude = float(point.details["best_amplitude"])
    if mechanism == "linear_no_bias":
        return jnp.ravel(jnp.asarray(direction.params) * amplitude), "known_linear_direction"
    weights, bias = direction.params
    theta = jnp.concatenate(
        [
            jnp.ravel(jnp.asarray(weights) * amplitude),
            jnp.ravel(jnp.asarray(bias) * amplitude),
        ]
    )
    return theta, "known_affine_direction"


def serialize_row(point: Any, reference_row: dict[str, Any]) -> dict[str, Any]:
    row = point.as_dict()
    details = row["details"]
    return {
        **row,
        "stage": row["phase"],
        "adam_steps": int(details["adam_steps"]),
        "adam_learning_rate": float(details["adam_learning_rate"]),
        "initialization": str(details["initialization"]),
        "reference_optimizer": str(reference_row["optimizer"]),
        "reference_lambda_multiplier": float(reference_row["lowest_valid_lambda_multiplier"]),
        "reference_lambda": float(reference_row["lowest_valid_lambda"]),
        "reference_bracket_low_multiplier": reference_row["bracket_low_multiplier"],
        "reference_bracket_high_multiplier": reference_row["bracket_high_multiplier"],
        "reference_gain": reference_row["gain_at_lowest_valid"],
        "reference_max_norm_over_cap": reference_row["max_norm_over_cap_at_lowest_valid"],
        "reference_cap_bound_fraction": reference_row["cap_bound_fraction_at_lowest_valid"],
        "match_reference_region": matches_reference_region(row),
    }


def matches_reference_region(row: dict[str, Any]) -> bool:
    return (
        bool(row["valid"])
        and row["finite_status"] == "finite"
        and row["gradient_status"] == "finite"
        and bool(row["useful"])
        and bool(row["interior"])
    )


def summarize_tuning(
    run_id: str,
    mechanism: str,
    reference_row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best = select_best_row(rows)
    stage1_matches = [row for row in rows if row["stage"] == "stage1_grid" and row["match_reference_region"]]
    any_match = best is not None
    return {
        "run_id": run_id,
        "mechanism": mechanism,
        "reference_optimizer": reference_row["optimizer"],
        "reference_lambda_multiplier": reference_row["lowest_valid_lambda_multiplier"],
        "reference_lambda_range": lambda_range_label(reference_row),
        "reference_gain": reference_row["gain_at_lowest_valid"],
        "match": any_match,
        "stage1_match": bool(stage1_matches),
        "stage2_required": any_match and not bool(stage1_matches),
        "best_stage": None if best is None else best["stage"],
        "best_adam_steps": None if best is None else best["adam_steps"],
        "best_adam_learning_rate": None if best is None else best["adam_learning_rate"],
        "best_initialization": None if best is None else best["initialization"],
        "best_gain": None if best is None else best["objective_gain_over_zero"],
        "best_max_norm_over_cap": None if best is None else best["max_norm_over_cap"],
        "best_cap_bound_fraction": None if best is None else best["cap_bound_fraction"],
        "best_finite_status": None if best is None else best["finite_status"],
        "best_gradient_status": None if best is None else best["gradient_status"],
        "best_optimizer_status": None if best is None else best["optimizer_status"],
        "best_failure_mode": None if best is None else best["failure_mode"],
        "n_stage1_settings": sum(1 for row in rows if row["stage"] == "stage1_grid"),
        "n_stage1_matches": len(stage1_matches),
        "n_total_matches": sum(1 for row in rows if row["match_reference_region"]),
        "recommendation": row_recommendation(mechanism, any_match, bool(stage1_matches)),
    }


def select_best_row(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [row for row in rows if row["match_reference_region"]]
    if not matches:
        return None
    return max(
        matches,
        key=lambda row: (
            1 if row["stage"] == "stage1_grid" else 0,
            float(row["objective_gain_over_zero"]),
            -float(row["max_norm_over_cap"]),
        ),
    )


def common_stage1_settings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {(run_id, mechanism) for run_id in RUN_IDS for mechanism in MECHANISMS}
    grouped: dict[tuple[int, float], set[tuple[str, str]]] = {}
    for row in rows:
        if row["stage"] != "stage1_grid" or not row["match_reference_region"]:
            continue
        key = (int(row["adam_steps"]), float(row["adam_learning_rate"]))
        grouped.setdefault(key, set()).add((str(row["run_id"]), str(row["mechanism"])))
    settings = [
        {"adam_steps": steps, "adam_learning_rate": learning_rate}
        for (steps, learning_rate), matched in grouped.items()
        if matched == expected
    ]
    return sorted(settings, key=lambda row: (row["adam_steps"], row["adam_learning_rate"]))


def lambda_range_label(reference_row: dict[str, Any]) -> str:
    low = reference_row["bracket_low_multiplier"]
    high = reference_row["bracket_high_multiplier"]
    ref = reference_row["lowest_valid_lambda_multiplier"]
    if low is None or high is None:
        return f"reference={format_optional(ref)}x"
    return f"{format_optional(low)}x-{format_optional(high)}x; reference={format_optional(ref)}x"


def row_recommendation(mechanism: str, match: bool, stage1_match: bool) -> str:
    if stage1_match:
        return "Adam stage 1 is reliable enough for this frozen-audit row."
    if match:
        return "Adam needs known-reference initialization on this row; do not treat zero-start Adam as reliable."
    if mechanism == "affine":
        return "Keep affine on an L-BFGS-style inner solver for this row."
    return "No Adam setting matched the reference region for this row."


def recommend(summaries: list[dict[str, Any]], common_settings: list[dict[str, Any]]) -> str:
    direct_ok = all(row["match"] for row in summaries if row["mechanism"] == "direct_epsilon")
    linear_ok = all(row["match"] for row in summaries if row["mechanism"] == "linear_no_bias")
    affine_ok = all(row["match"] for row in summaries if row["mechanism"] == "affine")
    affine_stage1_ok = all(
        row["stage1_match"] for row in summaries if row["mechanism"] == "affine"
    )
    if direct_ok and linear_ok and affine_stage1_ok:
        if common_settings:
            setting = common_settings[0]
            return (
                "Stage 1 zero-start Adam matches all direct, linear, and affine reference "
                f"regions. The conservative common setting is steps={setting['adam_steps']} "
                f"and lr={setting['adam_learning_rate']:.1e}; use that for training-facing "
                "smoke tests before considering more aggressive per-row settings."
            )
        return (
            "Stage 1 zero-start Adam matches all direct, linear, and affine reference regions; "
            "prefer the lowest common matching step/lr setting for training-facing smoke tests."
        )
    if direct_ok and linear_ok and not affine_ok:
        return (
            "Use Adam for direct-epsilon and linear no-bias frozen inner solves, but keep affine "
            "on an L-BFGS-style inner solver."
        )
    if direct_ok and linear_ok:
        return (
            "Adam matches direct-epsilon and linear no-bias rows. Affine only matches with the "
            "harder initialization stage, so zero-start affine Adam is not reliable enough for "
            "training-facing use."
        )
    return (
        "Adam did not reliably match all simpler direct/linear reference regions; keep the "
        "L-BFGS-style frozen-audit solver as the reliability baseline."
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "mechanism",
        "stage",
        "lambda_multiplier",
        "reference_lambda_multiplier",
        "reference_bracket_low_multiplier",
        "reference_bracket_high_multiplier",
        "adam_steps",
        "adam_learning_rate",
        "initialization",
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
        "match_reference_region",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Frozen Adam audit reliability tuning",
        "",
        f"Issue: `{payload['issue']}`. Reference issue: `1697bdc`.",
        "",
        "This audit uses the same frozen c92 no-PGD rows, lambda regions, mechanisms, "
        "and strict validity rule as the 1697bdc frozen-audit reference. A match means "
        "Adam found a finite, useful, interior policy at the reference lambda multiplier; "
        "objective equality with the reference solver is not required.",
        "",
        "## Recommendation",
        "",
        payload["recommendation"],
        "",
        "Common Stage 1 matching settings: "
        + format_common_settings(payload["common_stage1_matching_settings"]),
        "",
        "## Headline",
        "",
        "| row | mechanism | lambda multiplier/range | Adam settings | finite/useful/interior | norm/cap | cap-bound | gain | optimizer status | match | recommendation |",
        "|---|---|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in payload["summary"]:
        settings = "n/a"
        finite = "n/a"
        norm = "n/a"
        cap = "n/a"
        gain = "n/a"
        status = "n/a"
        if row["best_adam_steps"] is not None:
            settings = (
                f"{row['best_stage']}; steps={row['best_adam_steps']}; "
                f"lr={row['best_adam_learning_rate']:.1e}; init={row['best_initialization']}"
            )
            finite = f"{row['best_finite_status']}/true/true"
            norm = format_optional(row["best_max_norm_over_cap"])
            cap = format_optional_percent(row["best_cap_bound_fraction"])
            gain = format_optional(row["best_gain"])
            status = str(row["best_optimizer_status"])
        lines.append(
            f"| `{row['run_id']}` | `{row['mechanism']}` | {row['reference_lambda_range']} | "
            f"{settings} | {finite} | {norm} | {cap} | {gain} | {status} | "
            f"{'match' if row['match'] else 'mismatch'} | {row['recommendation']} |"
        )
    lines.extend(
        [
            "",
            "## Stage 1 grid",
            "",
            "| row | mechanism | matching settings | best matching gain | reference solver | reference gain |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for row in payload["summary"]:
        lines.append(
            f"| `{row['run_id']}` | `{row['mechanism']}` | {row['n_stage1_matches']} / "
            f"{row['n_stage1_settings']} | {format_optional(row['best_gain'])} | "
            f"`{row['reference_optimizer']}` | {format_optional(row['reference_gain'])} |"
        )
    stage2 = [row for row in payload["summary"] if row["stage2_required"]]
    if stage2:
        lines.extend(["", "## Stage 2 harder knob", ""])
        lines.append(
            "Stage 2 ran only where zero-start Stage 1 Adam had no valid point. The harder knob "
            "was known-reference-direction initialization at the same lambda multiplier."
        )
        lines.extend(
            [
                "",
                "| row | mechanism | initialization | gain | norm/cap | match |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for row in stage2:
            lines.append(
                f"| `{row['run_id']}` | `{row['mechanism']}` | `{row['best_initialization']}` | "
                f"{format_optional(row['best_gain'])} | "
                f"{format_optional(row['best_max_norm_over_cap'])} | "
                f"{'match' if row['match'] else 'mismatch'} |"
            )
    lines.extend(
        [
            "",
            "## Machine-readable artifacts",
            "",
            "- `results/f3c5db9/frozen_adam_audit_tuning.json`",
            "- `results/f3c5db9/frozen_adam_audit_tuning.csv`",
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


def format_common_settings(settings: list[dict[str, Any]]) -> str:
    if not settings:
        return "none"
    return ", ".join(
        f"steps={row['adam_steps']}, lr={float(row['adam_learning_rate']):.1e}"
        for row in settings
    )


if __name__ == "__main__":
    raise SystemExit(main())
