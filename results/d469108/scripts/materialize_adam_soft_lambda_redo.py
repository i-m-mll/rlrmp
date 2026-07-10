"""Materialize corrected frozen Adam soft-lambda matching for d469108."""

from __future__ import annotations
from rlrmp.io import write_csv_rows
from rlrmp.io import load_named_python_module as load_module

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np

from rlrmp.io import compact_json_dumps, update_marked_section, write_compact_json


REPO_ROOT = Path(__file__).resolve().parents[3]


RUN_IDS = ("open_loop_small", "open_loop_moderate", "open_loop_stress")
BETA_VALUES = (0.95, 1.05, 1.2, 1.4, 1.8)
MECHANISMS = ("direct_epsilon", "linear_no_bias", "affine")
CLOSED_LOOP_MECHANISMS = ("linear_no_bias", "affine")
HVP_SOURCE_JSON = "results/06a4dc8/canonical_soft_lambda_hvp.json"
DIRECT_REFERENCE_JSON = "results/7180984/direct_epsilon_soft_lambda_redo.json"
CLOSED_LOOP_REFERENCE_JSON = "results/6cfa892/closed_loop_soft_lambda_redo.json"
DIRECT_REFERENCE_SCRIPT = (
    REPO_ROOT / "results" / "7180984" / "scripts" / "materialize_direct_epsilon_soft_lambda_redo.py"
)
CLOSED_LOOP_REFERENCE_SCRIPT = (
    REPO_ROOT / "results" / "6cfa892" / "scripts" / "materialize_closed_loop_soft_lambda_redo.py"
)
LEGACY_ADAM_SCRIPT = (
    REPO_ROOT / "results" / "f3c5db9" / "scripts" / "materialize_frozen_adam_audit_tuning.py"
)
REFERENCE_POLICY_AUDIT = (
    REPO_ROOT / "results" / "3b850d6" / "scripts" / "materialize_closed_loop_policy_audit.py"
)
REFERENCE_CRITICAL_SEARCH = (
    REPO_ROOT / "results" / "1697bdc" / "scripts" / "materialize_critical_lambda_search.py"
)
PRIMARY_LAMBDA_SUMMARY = "lambda_star_p90"
OBJECTIVE_GAIN_TOL = 1e-5
TASK_GAIN_TOL = 1e-9
NONZERO_NORM_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Adam inner solves against corrected HVP/p90 "
            "direct-epsilon and closed-loop soft-lambda references."
        )
    )
    parser.add_argument("--experiment", default="c92ebd8")
    parser.add_argument("--issue", default="d469108")
    parser.add_argument("--run-ids", nargs="+", default=list(RUN_IDS))
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETA_VALUES))
    parser.add_argument("--mechanisms", nargs="+", default=list(MECHANISMS), choices=MECHANISMS)
    parser.add_argument("--hvp-source-json", default=HVP_SOURCE_JSON)
    parser.add_argument("--direct-reference-json", default=DIRECT_REFERENCE_JSON)
    parser.add_argument("--closed-loop-reference-json", default=CLOSED_LOOP_REFERENCE_JSON)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--pgd-steps", type=int, default=8)
    parser.add_argument("--pgd-step-size-fraction", type=float, default=0.25)
    parser.add_argument("--fixed-point-steps", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--line-search-reference-beta", type=float, default=1.4)
    parser.add_argument("--adam-steps", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument(
        "--adam-learning-rates",
        type=float,
        nargs="+",
        default=[1e-5, 5e-5, 1e-4],
    )
    parser.add_argument(
        "--output-json",
        default="results/d469108/adam_soft_lambda_redo.json",
    )
    parser.add_argument(
        "--output-detail-json",
        default="_artifacts/d469108/adam_soft_lambda_redo_detail.json",
        help=(
            "Bulk-detail sink for the nested per-run Adam rows. Lives under the "
            "gitignored _artifacts mirror; the tracked --output-json keeps only a "
            "slim manifest plus a bulk-detail pointer."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="results/d469108/adam_soft_lambda_redo.csv",
    )
    parser.add_argument(
        "--output-md",
        default="results/d469108/notes/adam_soft_lambda_redo.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = materialize(args)
    output_json = REPO_ROOT / args.output_json
    output_detail_json = REPO_ROOT / args.output_detail_json
    output_csv = REPO_ROOT / args.output_csv
    output_md = REPO_ROOT / args.output_md
    # The full per-setting Adam table is bulk (~1.8 MB serialized twice as nested
    # rows[].adam_rows and flat_rows). Keep the tracked JSON as a slim manifest and
    # push the canonical nested detail to the gitignored _artifacts mirror; the flat
    # table is the CSV twin and is not serialized a third time.
    slim_payload, detail_sha, detail_counts = split_payload(payload, args.output_detail_json)
    write_detail_json(output_detail_json, payload, args.output_detail_json)
    write_compact_json(output_json, slim_payload)
    write_csv(output_csv, payload["flat_rows"])
    update_marked_section(output_md, "adam_soft_lambda_redo", render_markdown(payload))
    print(
        json.dumps(
            {
                "json": str(output_json),
                "detail_json": str(output_detail_json),
                "detail_sha256": detail_sha,
                "detail_counts": detail_counts,
                "csv": str(output_csv),
                "markdown": str(output_md),
                "hvp_source_json": str((REPO_ROOT / args.hvp_source_json).resolve()),
            },
            indent=2,
        )
    )
    return 0


def build_detail_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Canonical bulk-detail document: nested per-run Adam rows only.

    The flat per-setting table (``flat_rows``) is the deterministic concatenation of
    ``rows[].adam_rows`` and is covered by the tracked CSV twin, so it is not
    duplicated in the detail file.
    """

    return {
        "schema_version": payload["schema_version"],
        "issue": payload["issue"],
        "source_experiment": payload["source_experiment"],
        "detail_of": "results/d469108/adam_soft_lambda_redo.json",
        "canonical_form": "nested_rows",
        "note": (
            "Canonical bulk form is the nested per-run Adam rows (rows[].adam_rows). "
            "The flat per-setting table (flat_rows) is the deterministic concatenation "
            "of rows[].adam_rows and is covered by the tracked CSV twin "
            "results/d469108/adam_soft_lambda_redo.csv; it is not duplicated here."
        ),
        "rows": payload["rows"],
    }


def split_payload(
    payload: dict[str, Any],
    detail_rel_path: str,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    """Return the slim tracked manifest plus the detail file's sha256 and counts.

    The tracked manifest keeps every summary/manifest field and a per-run row shell
    (without the dense ``adam_rows`` payload), drops ``flat_rows`` entirely, and adds
    a ``bulk_detail_manifest`` pointer into the gitignored ``_artifacts`` mirror.
    """

    detail_bytes = compact_json_dumps(build_detail_payload(payload)).encode("utf-8")
    detail_sha = hashlib.sha256(detail_bytes).hexdigest()
    counts = {
        "rows": len(payload["rows"]),
        "adam_rows": sum(len(row["adam_rows"]) for row in payload["rows"]),
    }
    slim = {key: value for key, value in payload.items() if key not in {"rows", "flat_rows"}}
    slim["rows"] = [
        {key: value for key, value in row.items() if key != "adam_rows"}
        for row in payload["rows"]
    ]
    slim["bulk_detail_manifest"] = {
        "path": detail_rel_path,
        "format": "json",
        "contains": (
            "full nested per-run Adam rows (rows[].adam_rows); the flat per-setting "
            "table is covered by the tracked CSV twin "
            "results/d469108/adam_soft_lambda_redo.csv"
        ),
        "sha256": detail_sha,
        "counts": counts,
    }
    return slim, detail_sha, counts


def write_detail_json(path: Path, payload: dict[str, Any], detail_rel_path: str) -> None:
    """Write the canonical nested bulk-detail document to the _artifacts mirror.

    The sha256 recorded in the tracked manifest is computed over exactly these bytes
    (see ``split_payload``), so both use ``compact_json_dumps`` serialization.
    """

    del detail_rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compact_json_dumps(build_detail_payload(payload)).encode("utf-8"))


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    direct_module = load_module("direct_epsilon_soft_lambda_redo_7180984", DIRECT_REFERENCE_SCRIPT)
    closed_module = load_module(
        "closed_loop_soft_lambda_redo_6cfa892", CLOSED_LOOP_REFERENCE_SCRIPT
    )
    legacy_adam = load_module("frozen_adam_audit_tuning_f3c5db9", LEGACY_ADAM_SCRIPT)
    reference = load_module("closed_loop_policy_audit_reference_3b850d6", REFERENCE_POLICY_AUDIT)
    search = load_module("critical_lambda_search_reference_1697bdc", REFERENCE_CRITICAL_SEARCH)
    force_module_roots(closed_module, search)

    hvp_payload = direct_module.read_hvp_payload(REPO_ROOT / args.hvp_source_json)
    direct_reference_payload = read_json(REPO_ROOT / args.direct_reference_json)
    closed_reference_payload = read_json(REPO_ROOT / args.closed_loop_reference_json)
    reference_rows = build_reference_rows(direct_reference_payload, closed_reference_payload)
    source_rows = {row["run_id"]: row for row in hvp_payload["rows"]}

    run_rows = []
    flat_rows = []
    for run_id in args.run_ids:
        if run_id not in source_rows:
            raise ValueError(f"Run {run_id!r} is missing from {args.hvp_source_json}")
        source_row = source_rows[run_id]
        beta_rows = direct_module.beta_mapping_from_source(source_row, args.betas)
        frozen = reference.load_frozen_batch(args, run_id)
        row_dicts = []
        if "direct_epsilon" in args.mechanisms:
            row_dicts.extend(
                evaluate_direct_adam_rows(
                    legacy_adam=legacy_adam,
                    closed_module=closed_module,
                    search=search,
                    reference=reference,
                    frozen=frozen,
                    run_id=run_id,
                    beta_rows=beta_rows,
                    reference_rows=reference_rows,
                    args=args,
                )
            )
        closed_mechanisms = [item for item in args.mechanisms if item in CLOSED_LOOP_MECHANISMS]
        if closed_mechanisms:
            row_dicts.extend(
                evaluate_closed_loop_adam_rows(
                    closed_module=closed_module,
                    search=search,
                    reference=reference,
                    frozen=frozen,
                    run_id=run_id,
                    beta_rows=beta_rows,
                    mechanisms=closed_mechanisms,
                    reference_rows=reference_rows,
                    args=args,
                )
            )
        run_summary = summarize_run(row_dicts)
        run_rows.append(
            {
                "run_id": run_id,
                "run_spec_path": f"results/{args.experiment}/runs/{run_id}.json",
                "artifact_dir": f"_artifacts/{args.experiment}/runs/{run_id}",
                "checkpoint": f"_artifacts/{args.experiment}/runs/{run_id}/trained_model.eqx",
                "hvp_source": direct_module.hvp_row_source_summary(
                    source_row,
                    str(args.hvp_source_json),
                ),
                "beta_mapping": beta_rows,
                "adam_rows": row_dicts,
                "summary": run_summary,
            }
        )
        flat_rows.extend(row_dicts)

    payload = {
        "schema_version": "rlrmp.adam_soft_lambda_redo.v1",
        "issue": str(args.issue),
        "parent_umbrella": "54389a4",
        "coordinator_issue": "31e67ff",
        "source_experiment": str(args.experiment),
        "command": {"argv": list(sys.argv), "cwd": "."},
        "hvp_source": {
            "path": str(args.hvp_source_json),
            "schema_version": hvp_payload.get("schema_version"),
            "issue": hvp_payload.get("issue"),
            "primary_continuity_summary": PRIMARY_LAMBDA_SUMMARY,
            "pooled_summary": hvp_payload.get("pooled_summary"),
            "pooled_beta_mapping": hvp_payload.get("pooled_beta_mapping"),
        },
        "reference_sources": {
            "direct_epsilon": {
                "path": str(args.direct_reference_json),
                "issue": direct_reference_payload.get("issue"),
                "schema_version": direct_reference_payload.get("schema_version"),
                "reference_optimizer": "pgd_projected_epsilon",
            },
            "closed_loop": {
                "path": str(args.closed_loop_reference_json),
                "issue": closed_reference_payload.get("issue"),
                "schema_version": closed_reference_payload.get("schema_version"),
                "reference_optimizer": "line_search_known_direction",
            },
        },
        "beta_policy": {
            "formula": "lambda(beta) = beta^2 * substrate_p90(lambda_star_i)",
            "values": [float(beta) for beta in args.betas],
            "diagnostic_only_beta_values": [
                float(beta) for beta in args.betas if float(beta) < 1.0
            ],
            "primary_beta_values": [float(beta) for beta in args.betas if float(beta) >= 1.0],
        },
        "adam_grid": {
            "steps": [int(value) for value in args.adam_steps],
            "learning_rates": [float(value) for value in args.adam_learning_rates],
            "initialization": "zero",
        },
        "classification_contract": {
            "criterion": "objective-level behavior only",
            "uses_cap_or_interiority_as_criterion": False,
            "finite_required": True,
            "gradient_finite_required_for_adam": True,
            "nonzero_threshold_l2": NONZERO_NORM_TOL,
            "positive_penalized_gain_threshold": OBJECTIVE_GAIN_TOL,
            "positive_task_gain_threshold": TASK_GAIN_TOL,
            "agreement_rule": (
                "Adam agrees when its objective-level success flag and classification "
                "match the corrected reference row for the same run, mechanism, and beta."
            ),
            "old_cap_role": "sidecar diagnostic only; not pass/fail and not lambda selection",
        },
        "rows": run_rows,
        "flat_rows": flat_rows,
        "summary": summarize_overall(flat_rows),
        "overall_interpretation": interpret_overall(flat_rows),
    }
    payload["summary_groups"] = summarize_groups(flat_rows)
    return payload


def evaluate_direct_adam_rows(
    *,
    legacy_adam: Any,
    closed_module: Any,
    search: Any,
    reference: Any,
    frozen: Any,
    run_id: str,
    beta_rows: list[dict[str, Any]],
    reference_rows: dict[tuple[str, str, float], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    value_and_grad = legacy_adam.make_direct_value_and_grad(search, reference, frozen)
    rows = []
    for beta_row in beta_rows:
        lambda_value = float(beta_row["lambda"])
        zero = jnp.zeros_like(frozen.trial_specs.inputs["epsilon"])
        zero_metrics = search.score_closed_loop_delta(
            reference,
            frozen,
            zero,
            jnp.asarray(lambda_value),
        )
        for setting_index, setting in enumerate(adam_settings(args)):
            result = legacy_adam.optimize_with_adam(
                value_and_grad,
                zero,
                lambda_value,
                int(setting["adam_steps"]),
                float(setting["adam_learning_rate"]),
            )
            delta = jnp.asarray(result["theta"]) * frozen.time_mask
            metrics = search.score_closed_loop_delta(
                reference,
                frozen,
                delta,
                jnp.asarray(lambda_value),
            )
            grad = jnp.asarray(result["grad"])
            row = closed_module.summarize_delta(
                reference=reference,
                frozen=frozen,
                run_id=run_id,
                mechanism="direct_epsilon",
                optimizer="adam",
                beta_row=beta_row,
                lambda_value=lambda_value,
                delta=delta,
                metrics=metrics,
                zero_metrics=zero_metrics,
                gradient_norm=float(jnp.linalg.norm(grad)),
                gradient_status="finite" if bool(jnp.all(jnp.isfinite(grad))) else "nonfinite",
                optimizer_success=bool(result["success"]),
                optimizer_status=str(result["status"]),
                optimizer_iterations=int(result["iterations"]),
                optimizer_evaluations=int(result["evaluations"]),
                details={
                    "theta_size": int(np.size(np.asarray(result["theta"]))),
                    "adam_steps": int(setting["adam_steps"]),
                    "adam_learning_rate": float(setting["adam_learning_rate"]),
                    "setting_index": int(setting_index),
                    "initialization": "zero_direct_epsilon",
                },
            ).as_dict()
            rows.append(
                attach_reference_agreement(
                    row,
                    reference_rows[reference_key(run_id, "direct_epsilon", beta_row["beta"])],
                    setting,
                )
            )
    return rows


def evaluate_closed_loop_adam_rows(
    *,
    closed_module: Any,
    search: Any,
    reference: Any,
    frozen: Any,
    run_id: str,
    beta_rows: list[dict[str, Any]],
    mechanisms: list[str],
    reference_rows: dict[tuple[str, str, float], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    direction_cache = closed_module.build_direction_cache(
        reference,
        frozen,
        beta_rows=beta_rows,
        reference_beta=float(args.line_search_reference_beta),
        ridge_alpha=float(args.ridge_alpha),
        args=args,
    )
    for mechanism in mechanisms:
        codec = search.make_theta_codec(reference, frozen, mechanism)
        value_and_grad = search.make_closed_loop_value_and_grad(reference, frozen, codec, args)
        for beta_row in beta_rows:
            reference_row = reference_rows[reference_key(run_id, mechanism, beta_row["beta"])]
            for setting_index, setting in enumerate(adam_settings(args)):
                setting_args = namespace_with_setting(args, setting)
                row = closed_module.evaluate_closed_loop_row(
                    reference=reference,
                    search=search,
                    frozen=frozen,
                    codec=codec,
                    value_and_grad=value_and_grad,
                    direction=direction_cache[mechanism],
                    run_id=run_id,
                    mechanism=mechanism,
                    optimizer="adam",
                    beta_row=beta_row,
                    args=setting_args,
                ).as_dict()
                details = dict(row["details"])
                details.update(
                    {
                        "adam_steps": int(setting["adam_steps"]),
                        "adam_learning_rate": float(setting["adam_learning_rate"]),
                        "setting_index": int(setting_index),
                        "initialization": "zero_policy_parameters",
                    }
                )
                row["details"] = details
                rows.append(attach_reference_agreement(row, reference_row, setting))
    return rows


def build_reference_rows(
    direct_payload: dict[str, Any],
    closed_payload: dict[str, Any],
) -> dict[tuple[str, str, float], dict[str, Any]]:
    rows: dict[tuple[str, str, float], dict[str, Any]] = {}
    for run in direct_payload["rows"]:
        for sweep in run["sweep"]:
            reference = normalize_reference_row(
                run_id=str(run["run_id"]),
                mechanism="direct_epsilon",
                optimizer="pgd_projected_epsilon",
                row=sweep,
            )
            rows[reference_key(reference["run_id"], reference["mechanism"], reference["beta"])] = (
                reference
            )
    for row in closed_payload["flat_rows"]:
        if row["optimizer"] != "line_search_known_direction":
            continue
        reference = normalize_reference_row(
            run_id=str(row["run_id"]),
            mechanism=str(row["mechanism"]),
            optimizer="line_search_known_direction",
            row=row,
        )
        rows[reference_key(reference["run_id"], reference["mechanism"], reference["beta"])] = (
            reference
        )
    return rows


def normalize_reference_row(
    *,
    run_id: str,
    mechanism: str,
    optimizer: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    task_gain = float(row["task_loss_gain"])
    success = objective_success(row)
    return {
        "run_id": run_id,
        "mechanism": mechanism,
        "optimizer": optimizer,
        "beta": float(row["beta"]),
        "beta_role": row["beta_role"],
        "lambda": float(row["lambda"]),
        "finite_status": row["finite_status"],
        "gradient_status": row.get("gradient_status"),
        "selected_nonzero": bool(row["selected_nonzero"]),
        "classification": row["classification"],
        "objective_level_success": success,
        "task_gain_positive": task_gain > TASK_GAIN_TOL,
        "penalized_gain_over_zero": float(row["penalized_gain_over_zero"]),
        "task_loss_gain": task_gain,
        "energy_mean": float(row["energy_mean"]),
        "energy_penalty": float(row["energy_penalty"]),
        "penalty_minus_task_gain": float(row["penalty_minus_task_gain"]),
        "penalty_over_task_gain_abs": float(row["penalty_over_task_gain_abs"]),
        "selected_norm_max": float(
            row.get("selected_epsilon_norm_max", row.get("selected_policy_norm_max", np.nan))
        ),
        "old_cap_ratio_max_sidecar": float(row["old_cap_ratio_max_sidecar"]),
        "old_cap_boundary_fraction_sidecar": float(row["old_cap_boundary_fraction_sidecar"]),
        "old_cap_used_as_criterion": bool(row["old_cap_used_as_criterion"]),
    }


def attach_reference_agreement(
    row: dict[str, Any],
    reference: dict[str, Any],
    setting: dict[str, Any],
) -> dict[str, Any]:
    adam_success = objective_success(row)
    adam_task_gain_positive = float(row["task_loss_gain"]) > TASK_GAIN_TOL
    matches_success = adam_success == bool(reference["objective_level_success"])
    matches_class = str(row["classification"]) == str(reference["classification"])
    matches_selected = bool(row["selected_nonzero"]) == bool(reference["selected_nonzero"])
    if matches_success and matches_class and matches_selected:
        agreement = "agrees_with_reference_classification"
    elif matches_success:
        agreement = "same_success_different_objective_class"
    else:
        agreement = "fails_reference_objective_success"
    return {
        **row,
        "adam_steps": int(setting["adam_steps"]),
        "adam_learning_rate": float(setting["adam_learning_rate"]),
        "adam_objective_level_success": adam_success,
        "adam_task_gain_positive": adam_task_gain_positive,
        "reference_optimizer": reference["optimizer"],
        "reference_classification": reference["classification"],
        "reference_objective_level_success": bool(reference["objective_level_success"]),
        "reference_task_gain_positive": bool(reference["task_gain_positive"]),
        "reference_selected_nonzero": bool(reference["selected_nonzero"]),
        "reference_penalized_gain_over_zero": float(reference["penalized_gain_over_zero"]),
        "reference_task_loss_gain": float(reference["task_loss_gain"]),
        "reference_energy_mean": float(reference["energy_mean"]),
        "reference_energy_penalty": float(reference["energy_penalty"]),
        "reference_selected_norm_max": float(reference["selected_norm_max"]),
        "matches_reference_success": matches_success,
        "matches_reference_classification": matches_class,
        "matches_reference_selected_nonzero": matches_selected,
        "agreement": agreement,
    }


def objective_success(row: dict[str, Any]) -> bool:
    gradient_status = row.get("gradient_status", "finite")
    return (
        row["finite_status"] == "finite"
        and gradient_status in (None, "finite")
        and bool(row["selected_nonzero"])
        and float(row["penalized_gain_over_zero"]) > OBJECTIVE_GAIN_TOL
    )


def summarize_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_groups(rows)


def summarize_overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [
        row
        for row in rows
        if row["finite_status"] == "finite" and row.get("gradient_status") == "finite"
    ]
    candidate = [row for row in rows if row["beta_role"] != "diagnostic_only"]
    candidate_groups = summarize_groups(candidate)
    all_groups = summarize_groups(rows)
    return {
        "n_rows": len(rows),
        "n_finite_gradient_rows": len(finite),
        "n_candidate_rows": len(candidate),
        "candidate_reference_groups": len(candidate_groups),
        "candidate_groups_with_any_classification_agreement": sum(
            1 for group in candidate_groups if group["any_classification_agreement"]
        ),
        "candidate_groups_with_any_success_agreement": sum(
            1 for group in candidate_groups if group["any_success_agreement"]
        ),
        "all_reference_groups": len(all_groups),
        "all_groups_with_any_classification_agreement": sum(
            1 for group in all_groups if group["any_classification_agreement"]
        ),
        "all_groups_with_any_success_agreement": sum(
            1 for group in all_groups if group["any_success_agreement"]
        ),
        "classification_counts": counts(row["classification"] for row in rows),
        "agreement_counts": counts(row["agreement"] for row in rows),
    }


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[reference_key(row["run_id"], row["mechanism"], row["beta"])].append(row)
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        best_gain = max(group_rows, key=lambda row: float(row["penalized_gain_over_zero"]))
        first_class_match = first_or_none(
            row for row in group_rows if row["matches_reference_classification"]
        )
        first_success_match = first_or_none(
            row for row in group_rows if row["matches_reference_success"]
        )
        representative = first_class_match or first_success_match or best_gain
        summaries.append(
            {
                "run_id": key[0],
                "mechanism": key[1],
                "beta": key[2],
                "beta_role": representative["beta_role"],
                "reference_optimizer": representative["reference_optimizer"],
                "reference_classification": representative["reference_classification"],
                "reference_objective_level_success": representative[
                    "reference_objective_level_success"
                ],
                "reference_penalized_gain": representative["reference_penalized_gain_over_zero"],
                "n_adam_settings": len(group_rows),
                "any_classification_agreement": first_class_match is not None,
                "any_success_agreement": first_success_match is not None,
                "representative_agreement": representative["agreement"],
                "representative_classification": representative["classification"],
                "representative_steps": representative["adam_steps"],
                "representative_learning_rate": representative["adam_learning_rate"],
                "representative_penalized_gain": representative["penalized_gain_over_zero"],
                "representative_task_gain": representative["task_loss_gain"],
                "representative_energy_penalty": representative["energy_penalty"],
                "representative_norm_max": representative["selected_policy_norm_max"],
                "representative_old_cap_ratio_max": representative["old_cap_ratio_max_sidecar"],
            }
        )
    return summaries


def interpret_overall(rows: list[dict[str, Any]]) -> str:
    summary = summarize_overall(rows)
    candidate_total = summary["candidate_reference_groups"]
    candidate_class = summary["candidate_groups_with_any_classification_agreement"]
    candidate_success = summary["candidate_groups_with_any_success_agreement"]
    finite = summary["n_finite_gradient_rows"]
    total = summary["n_rows"]
    return (
        f"Adam rows were finite with finite gradients on {finite}/{total} bounded evaluations. "
        f"Across beta>=1 candidate reference groups, Adam matched the corrected reference "
        f"classification in {candidate_class}/{candidate_total} groups and matched the "
        f"objective-success flag in {candidate_success}/{candidate_total} groups. Beta 0.95 is "
        "diagnostic only. Old hard-cap ratios are sidecars and do not enter selection, success, "
        "or failure labels."
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ['run_id', 'mechanism', 'beta', 'beta_role', 'lambda', 'adam_steps', 'adam_learning_rate', 'finite_status', 'gradient_status', 'optimizer_success', 'optimizer_status', 'optimizer_iterations', 'optimizer_evaluations', 'selected_nonzero', 'classification', 'adam_objective_level_success', 'penalized_gain_over_zero', 'task_loss_gain', 'energy_mean', 'energy_max', 'energy_penalty', 'penalty_minus_task_gain', 'penalty_over_task_gain_abs', 'selected_policy_norm_mean', 'selected_policy_norm_max', 'old_cap_ratio_mean_sidecar', 'old_cap_ratio_max_sidecar', 'old_cap_boundary_fraction_sidecar', 'old_cap_used_as_criterion', 'reference_optimizer', 'reference_classification', 'reference_objective_level_success', 'reference_selected_nonzero', 'reference_penalized_gain_over_zero', 'reference_task_loss_gain', 'matches_reference_success', 'matches_reference_classification', 'matches_reference_selected_nonzero', 'agreement', 'gradient_norm']
    write_csv_rows(path, list(rows), fieldnames=fieldnames)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Adam soft-lambda redo",
        "",
        f"Issue: `{payload['issue']}`. Source no-PGD substrates: `c92ebd8`.",
        "",
        (
            "This materializer evaluates zero-start frozen Adam inner solves against the "
            "corrected HVP/p90 lambda mapping and corrected direct-epsilon / closed-loop "
            "reference rows. It does not launch training and does not update controller "
            "weights."
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
            "Reference rows: direct-epsilon Adam is compared to corrected PGD direct-epsilon "
            "rows from `7180984`; closed-loop Adam is compared to corrected "
            "line-search-known-direction rows from `6cfa892`."
        ),
        "",
        (
            "Agreement is objective-level only: finite Adam status, finite gradient, selected "
            "nonzero perturbation, positive penalized gain over zero, task-loss gain class, "
            "energy/penalty relation, and sidecar norm diagnostics. Old cap/interiority ratios "
            "are not criteria."
        ),
        "",
        "## Headline",
        "",
        payload["overall_interpretation"],
        "",
        "## Per-reference summary",
        "",
        "| substrate | mechanism | beta | role | ref class | any class agreement | any success agreement | representative Adam setting | representative class | penalized gain | task gain | energy penalty | norm | old-cap ratio |",
        "|---|---|---:|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in payload["summary_groups"]:
        lines.append(format_group_row(group))
    mismatch_groups = [
        group for group in payload["summary_groups"] if not group["any_classification_agreement"]
    ]
    lines.extend(
        [
            "",
            "## Mismatch groups",
            "",
            "| substrate | mechanism | beta | role | reference class | Adam representative class | representative setting | representative gain | reference gain | note |",
            "|---|---|---:|---|---|---|---|---:|---:|---|",
        ]
    )
    if not mismatch_groups:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | none |")
    for group in mismatch_groups:
        lines.append(format_mismatch_row(group))
    lines.extend(
        [
            "",
            "The tracked `results/d469108/adam_soft_lambda_redo.json` is a slim "
            "manifest (summary/manifest fields + a `bulk_detail_manifest` pointer). "
            "The flat per-setting Adam table is the CSV twin "
            "`results/d469108/adam_soft_lambda_redo.csv`; the canonical nested rows "
            "(`rows[].adam_rows`) live in the gitignored bulk detail file "
            "`_artifacts/d469108/adam_soft_lambda_redo_detail.json`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Representative Adam rows",
            "",
            "| substrate | mechanism | beta | steps | lr | Adam class | ref class | agreement | penalized gain | task gain | norm | old-cap ratio |",
            "|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    representative_rows = representative_rows_by_group(payload["flat_rows"])
    for row in representative_rows:
        lines.append(
            f"| `{row['run_id']}` | `{row['mechanism']}` | {row['beta']:.3g} | "
            f"{row['adam_steps']} | {row['adam_learning_rate']:.1e} | "
            f"`{row['classification']}` | `{row['reference_classification']}` | "
            f"`{row['agreement']}` | {row['penalized_gain_over_zero']:.6g} | "
            f"{row['task_loss_gain']:.6g} | {row['selected_policy_norm_max']:.6g} | "
            f"{row['old_cap_ratio_max_sidecar']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Row classifications: {format_counts(payload['summary']['classification_counts'])}",
            f"- Agreement labels: {format_counts(payload['summary']['agreement_counts'])}",
            "",
            "## Reproduction",
            "",
            "```bash",
            "PYTHONPATH=src uv run --no-sync python \\",
            "  results/d469108/scripts/materialize_adam_soft_lambda_redo.py",
            "```",
            "",
            "Fast smoke:",
            "",
            "```bash",
            "PYTHONPATH=src uv run --no-sync python \\",
            "  results/d469108/scripts/materialize_adam_soft_lambda_redo.py \\",
            "  --run-ids open_loop_small --mechanisms direct_epsilon linear_no_bias \\",
            "  --betas 0.95 1.4 --adam-steps 2 --adam-learning-rates 5e-5 \\",
            "  --output-json results/d469108/smoke/adam_soft_lambda_redo.json \\",
            "  --output-csv results/d469108/smoke/adam_soft_lambda_redo.csv \\",
            "  --output-md results/d469108/smoke/adam_soft_lambda_redo.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def format_group_row(group: dict[str, Any]) -> str:
    setting = (
        f"steps={group['representative_steps']}; "
        f"lr={float(group['representative_learning_rate']):.1e}"
    )
    return (
        f"| `{group['run_id']}` | `{group['mechanism']}` | {group['beta']:.3g} | "
        f"{group['beta_role']} | `{group['reference_classification']}` | "
        f"{str(group['any_classification_agreement']).lower()} | "
        f"{str(group['any_success_agreement']).lower()} | {setting} | "
        f"`{group['representative_classification']}` | "
        f"{group['representative_penalized_gain']:.6g} | "
        f"{group['representative_task_gain']:.6g} | "
        f"{group['representative_energy_penalty']:.6g} | "
        f"{group['representative_norm_max']:.6g} | "
        f"{group['representative_old_cap_ratio_max']:.6g} |"
    )


def format_mismatch_row(group: dict[str, Any]) -> str:
    setting = (
        f"steps={group['representative_steps']}; "
        f"lr={float(group['representative_learning_rate']):.1e}"
    )
    if group["reference_objective_level_success"]:
        note = "Adam did not recover the reference positive objective behavior."
    else:
        note = "Adam found positive objective behavior where the reference selected zero."
    return (
        f"| `{group['run_id']}` | `{group['mechanism']}` | {group['beta']:.3g} | "
        f"{group['beta_role']} | `{group['reference_classification']}` | "
        f"`{group['representative_classification']}` | {setting} | "
        f"{group['representative_penalized_gain']:.6g} | "
        f"{group['reference_penalized_gain']:.6g} | {note} |"
    )


def representative_rows_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[reference_key(row["run_id"], row["mechanism"], row["beta"])].append(row)
    selected = []
    for _key, group_rows in sorted(grouped.items()):
        class_match = first_or_none(
            row for row in group_rows if row["matches_reference_classification"]
        )
        success_match = first_or_none(row for row in group_rows if row["matches_reference_success"])
        best_gain = max(group_rows, key=lambda row: float(row["penalized_gain_over_zero"]))
        selected.append(class_match or success_match or best_gain)
    return selected




def force_module_roots(*modules: Any) -> None:
    for module in modules:
        if hasattr(module, "REPO_ROOT"):
            module.REPO_ROOT = REPO_ROOT
    for module in modules:
        if hasattr(module, "REFERENCE_POLICY_AUDIT"):
            module.REFERENCE_POLICY_AUDIT = REFERENCE_POLICY_AUDIT
        if hasattr(module, "REFERENCE_CRITICAL_SEARCH"):
            module.REFERENCE_CRITICAL_SEARCH = REFERENCE_CRITICAL_SEARCH
        if hasattr(module, "REFERENCE_DIRECT_REDO"):
            module.REFERENCE_DIRECT_REDO = DIRECT_REFERENCE_SCRIPT


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reference_key(run_id: str, mechanism: str, beta: float) -> tuple[str, str, float]:
    return (str(run_id), str(mechanism), round(float(beta), 12))


def adam_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {"adam_steps": int(steps), "adam_learning_rate": float(learning_rate)}
        for steps in args.adam_steps
        for learning_rate in args.adam_learning_rates
    ]


def namespace_with_setting(args: argparse.Namespace, setting: dict[str, Any]) -> SimpleNamespace:
    data = vars(args).copy()
    data["adam_steps"] = int(setting["adam_steps"])
    data["adam_learning_rate"] = float(setting["adam_learning_rate"])
    return SimpleNamespace(**data)


def first_or_none(items: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for item in items:
        return item
    return None


def counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items()))


def format_counts(values: dict[str, int]) -> str:
    return ", ".join(f"`{key}`: {value}" for key, value in values.items())


if __name__ == "__main__":
    raise SystemExit(main())
