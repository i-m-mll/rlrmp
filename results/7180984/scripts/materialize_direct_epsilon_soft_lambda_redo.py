"""Materialize direct-epsilon soft-lambda redo rows for 7180984."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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

from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.train import cs_nominal_gru as nominal
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    _broad_epsilon_pgd_trust_radius,
    _epsilon_time_mask,
    _ensure_broad_epsilon_input,
    _flattened_per_trial_norm,
    config_from_broad_epsilon_pgd_hps,
    run_broad_epsilon_pgd_inner_maximizer,
)
from rlrmp.train.task_model import setup_task_model_pair


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_IDS = ("open_loop_small", "open_loop_moderate", "open_loop_stress")
BETA_VALUES = (0.95, 1.05, 1.2, 1.4, 1.8)
HVP_SOURCE_JSON = "results/06a4dc8/canonical_soft_lambda_hvp.json"
PRIMARY_LAMBDA_SUMMARY = "lambda_star_p90"
CAP_RADIUS_15CM = 0.004545500088363065
CAP_SOURCE = "ofb_6d_no_integrator_gamma_1p4_rollout_radius"
OBJECTIVE_GAIN_TOL = 1e-5
TASK_GAIN_TOL = 1e-9
NONZERO_NORM_TOL = 1e-12


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
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen c92 no-PGD direct-epsilon optima at HVP/Lanczos "
            "p90 soft-lambda beta scales."
        )
    )
    parser.add_argument("--experiment", default="c92ebd8")
    parser.add_argument("--issue", default="7180984")
    parser.add_argument("--run-ids", nargs="+", default=list(RUN_IDS))
    parser.add_argument("--hvp-source-json", default=HVP_SOURCE_JSON)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETA_VALUES))
    parser.add_argument("--pgd-steps", type=int, default=8)
    parser.add_argument("--pgd-step-size-fraction", type=float, default=0.25)
    parser.add_argument(
        "--output-json",
        default="results/7180984/direct_epsilon_soft_lambda_redo.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/7180984/direct_epsilon_soft_lambda_redo.csv",
    )
    parser.add_argument(
        "--output-md",
        default="results/7180984/notes/direct_epsilon_soft_lambda_redo.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = materialize(args)
    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    output_md = REPO_ROOT / args.output_md
    write_compact_json(output_json, payload)
    write_sweep_csv(output_csv, payload)
    update_marked_section(output_md, "direct_epsilon_soft_lambda_redo", render_markdown(payload))
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
    hvp_payload = read_hvp_payload(REPO_ROOT / args.hvp_source_json)
    rows = []
    source_rows = {row["run_id"]: row for row in hvp_payload["rows"]}
    for run_id in args.run_ids:
        if run_id not in source_rows:
            raise ValueError(f"Run {run_id!r} is missing from {args.hvp_source_json}")
        source_row = source_rows[run_id]
        frozen = load_frozen_batch(args, run_id)
        beta_rows = beta_mapping_from_source(source_row, args.betas)
        sweep_rows = [
            audit_open_loop_direct_epsilon(
                frozen,
                beta_row=beta_row,
                args=args,
            )
            for beta_row in beta_rows
        ]
        rows.append(
            {
                "run_id": run_id,
                "run_spec_path": f"results/{args.experiment}/runs/{run_id}.json",
                "artifact_dir": f"_artifacts/{args.experiment}/runs/{run_id}",
                "checkpoint": f"_artifacts/{args.experiment}/runs/{run_id}/trained_model.eqx",
                "hvp_source": hvp_row_source_summary(source_row, str(args.hvp_source_json)),
                "beta_mapping": beta_rows,
                "sweep": sweep_rows,
                "objective_classification_counts": classification_counts(sweep_rows),
            }
        )
    return {
        "schema_version": "rlrmp.direct_epsilon_soft_lambda_redo.v1",
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
        "beta_policy": {
            "formula": "lambda(beta) = beta^2 * substrate_p90(lambda_star_i)",
            "values": [float(beta) for beta in args.betas],
            "diagnostic_only_beta_values": [
                float(beta) for beta in args.betas if float(beta) < 1.0
            ],
            "primary_beta_values": [float(beta) for beta in args.betas if float(beta) >= 1.0],
        },
        "direct_epsilon_optimizer": {
            "objective": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            "pgd_steps": int(args.pgd_steps),
            "pgd_step_size_fraction": float(args.pgd_step_size_fraction),
            "optimized_variables": "epsilon sequence only; controller weights frozen",
            "objective_reduction": "mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]",
            "epsilon_dim": 6,
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
        "rows": rows,
    }


def read_hvp_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "rlrmp.canonical_soft_lambda_hvp.v1":
        raise ValueError(
            f"Unexpected HVP payload schema in {path}: {payload.get('schema_version')}"
        )
    return payload


def hvp_row_source_summary(source_row: dict[str, Any], source_path: str) -> dict[str, Any]:
    summary = source_row["summary"]
    return {
        "source_issue": "06a4dc8",
        "source_path": source_path,
        "run_id": source_row["run_id"],
        "n_trials_estimated": source_row["n_trials_estimated"],
        "n_trials_available": source_row["n_trials_available"],
        "replicate_index": source_row["replicate_index"],
        "primary_continuity_summary": PRIMARY_LAMBDA_SUMMARY,
        "lambda_star_median": summary["lambda_star_median"],
        "lambda_star_p75": summary["lambda_star_p75"],
        "lambda_star_p90": summary["lambda_star_p90"],
        "lambda_star_max": summary["lambda_star_max"],
        "finite_fraction": summary["finite_fraction"],
    }


def beta_mapping_from_source(
    source_row: dict[str, Any],
    betas: list[float],
) -> list[dict[str, Any]]:
    source_by_beta = {
        round(float(mapping["beta"]), 12): mapping for mapping in source_row["beta_mapping"]
    }
    lambda_star_p90 = float(source_row["summary"][PRIMARY_LAMBDA_SUMMARY])
    rows = []
    for beta in betas:
        beta_float = float(beta)
        source_mapping = source_by_beta.get(round(beta_float, 12))
        if source_mapping is None:
            lambda_value = float((beta_float**2) * lambda_star_p90)
            source = "computed_from_hvp_source_summary"
        else:
            lambda_value = float(source_mapping["lambda"])
            source = "copied_from_hvp_source_beta_mapping"
        rows.append(
            {
                "beta": beta_float,
                "lambda": lambda_value,
                "role": "diagnostic_only" if beta_float < 1.0 else "candidate_training_scale",
                "mapping": "lambda = beta^2 * substrate_p90(lambda_star_i)",
                "lambda_source": source,
                "lambda_star_summary": PRIMARY_LAMBDA_SUMMARY,
                "lambda_star_summary_value": lambda_star_p90,
            }
        )
    return rows


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


def audit_open_loop_direct_epsilon(
    frozen: FrozenBatch,
    *,
    beta_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lambda_value = float(beta_row["lambda"])
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
        beta_row=beta_row,
        delta=delta,
        radius=frozen.radius,
        diagnostics={key: np.asarray(jax.device_get(value)) for key, value in diagnostics.items()},
    )


def per_trial_energy(delta: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(delta), axis=tuple(range(1, delta.ndim)))


def audit_summary(
    *,
    beta_row: dict[str, Any],
    delta: jnp.ndarray,
    radius: jnp.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    lambda_value = float(beta_row["lambda"])
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
    finite_values = np.isfinite(
        [
            raw_loss_gain,
            energy_penalty,
            penalized_gain,
            scalar_diagnostic(diagnostics, "raw_task_loss_zero"),
            scalar_diagnostic(diagnostics, "raw_task_loss_selected"),
            float(lambda_value),
            float(jnp.mean(energy)),
        ]
    )
    finite_status = "finite" if bool(np.all(finite_values)) and not nonfinite_seen else "nonfinite"
    selected_norm_max = float(jnp.max(norm))
    selected_nonzero = selected_norm_max > NONZERO_NORM_TOL
    cap_fraction = float(jnp.mean((ratio >= 1.0 - 1e-4).astype(jnp.float32)))
    classification = classify_objective_behavior(
        finite_status=finite_status,
        selected_nonzero=selected_nonzero,
        penalized_gain=penalized_gain,
        raw_loss_gain=raw_loss_gain,
    )
    task_gain_denominator = max(abs(raw_loss_gain), 1e-30)
    return {
        "beta": float(beta_row["beta"]),
        "beta_role": beta_row["role"],
        "lambda": lambda_value,
        "lambda_mapping": beta_row["mapping"],
        "lambda_source": beta_row["lambda_source"],
        "lambda_star_summary": beta_row["lambda_star_summary"],
        "lambda_star_summary_value": float(beta_row["lambda_star_summary_value"]),
        "finite_status": finite_status,
        "selected_nonzero": bool(selected_nonzero),
        "classification": classification,
        "penalized_gain_over_zero": float(penalized_gain),
        "task_loss_gain": float(raw_loss_gain),
        "energy_mean": float(jnp.mean(energy)),
        "energy_max": float(jnp.max(energy)),
        "energy_penalty": float(energy_penalty),
        "penalty_minus_task_gain": float(energy_penalty - raw_loss_gain),
        "penalty_over_task_gain_abs": float(energy_penalty / task_gain_denominator),
        "selected_epsilon_norm_mean": float(jnp.mean(norm)),
        "selected_epsilon_norm_max": selected_norm_max,
        "old_cap_radius_mean": float(jnp.mean(radius)),
        "old_cap_radius_max": float(jnp.max(radius)),
        "old_cap_ratio_mean_sidecar": float(jnp.mean(ratio)),
        "old_cap_ratio_max_sidecar": float(jnp.max(ratio)),
        "old_cap_boundary_fraction_sidecar": cap_fraction,
        "old_cap_used_as_criterion": False,
        "direct_objective_reduction": "mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]",
        "diagnostics": plain_json(diagnostics),
    }


def classify_objective_behavior(
    *,
    finite_status: str,
    selected_nonzero: bool,
    penalized_gain: float,
    raw_loss_gain: float,
) -> str:
    if finite_status != "finite":
        return "optimizer_nonfinite"
    if selected_nonzero and penalized_gain > OBJECTIVE_GAIN_TOL:
        if raw_loss_gain > TASK_GAIN_TOL:
            return "nonzero_positive_penalized_and_task_gain"
        return "nonzero_positive_penalized_gain_without_task_gain"
    if selected_nonzero and penalized_gain < -OBJECTIVE_GAIN_TOL:
        return "nonzero_negative_penalized_gain"
    if selected_nonzero:
        return "nonzero_flat_penalized_gain"
    if penalized_gain > OBJECTIVE_GAIN_TOL:
        return "zero_selected_positive_penalized_gain"
    return "zero_selected_no_positive_penalized_gain"


def scalar_diagnostic(diagnostics: dict[str, Any], key: str) -> float:
    return float(np.asarray(diagnostics.get(key, np.nan)))


def plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain_json(item) for key, item in value.items()}
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, (np.bool_, bool)):
            return bool(scalar)
        if isinstance(scalar, (np.integer, int)):
            return int(scalar)
        return float(scalar)
    return array.tolist()


def classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = row["classification"]
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def write_sweep_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "beta",
        "beta_role",
        "lambda",
        "lambda_star_summary",
        "lambda_star_summary_value",
        "finite_status",
        "selected_nonzero",
        "classification",
        "penalized_gain_over_zero",
        "task_loss_gain",
        "energy_mean",
        "energy_max",
        "energy_penalty",
        "penalty_minus_task_gain",
        "penalty_over_task_gain_abs",
        "selected_epsilon_norm_mean",
        "selected_epsilon_norm_max",
        "old_cap_ratio_mean_sidecar",
        "old_cap_ratio_max_sidecar",
        "old_cap_boundary_fraction_sidecar",
        "old_cap_used_as_criterion",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in payload["rows"]:
            for sweep in row["sweep"]:
                writer.writerow(
                    {
                        "run_id": row["run_id"],
                        **{key: sweep[key] for key in fieldnames[1:]},
                    }
                )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Direct-epsilon soft-lambda redo",
        "",
        f"Issue: `{payload['issue']}`. Source no-PGD substrates: `c92ebd8`.",
        "",
        (
            "No controller weights were updated. This deterministic local materializer "
            "loads frozen c92 no-PGD substrates and evaluates direct-epsilon optima "
            "at beta-scaled lambda values from the corrected HVP/Lanczos p90 source."
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
            "Beta `0.95` is diagnostic only. Cap/interiority is not used as a "
            "criterion; old-cap ratios below are sidecars only."
        ),
        "",
        "## HVP/p90 beta mapping",
        "",
        "| substrate | beta | role | lambda_star p90 | lambda | source |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        for mapping in row["beta_mapping"]:
            lines.append(
                f"| `{row['run_id']}` | {mapping['beta']:.3g} | {mapping['role']} | "
                f"{mapping['lambda_star_summary_value']:.6g} | {mapping['lambda']:.6g} | "
                f"{mapping['lambda_source']} |"
            )
    lines.extend(
        [
            "",
            "## Direct-epsilon objective rows",
            "",
            "| substrate | beta | finite | selected | class | penalized gain | task gain | "
            "energy | penalty | norm | old-cap ratio |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        for sweep in row["sweep"]:
            lines.append(
                f"| `{row['run_id']}` | {sweep['beta']:.3g} | {sweep['finite_status']} | "
                f"{str(sweep['selected_nonzero']).lower()} | `{sweep['classification']}` | "
                f"{sweep['penalized_gain_over_zero']:.6g} | {sweep['task_loss_gain']:.6g} | "
                f"{sweep['energy_mean']:.6g} | {sweep['energy_penalty']:.6g} | "
                f"{sweep['selected_epsilon_norm_max']:.6g} | "
                f"{sweep['old_cap_ratio_max_sidecar']:.6g} |"
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
    for row in payload["rows"]:
        counts = ", ".join(
            f"`{label}`: {count}" for label, count in row["objective_classification_counts"].items()
        )
        lines.append(f"| `{row['run_id']}` | {counts} |")
    command = " ".join(
        [
            "PYTHONPATH=src",
            "uv run --no-sync python",
            "results/7180984/scripts/materialize_direct_epsilon_soft_lambda_redo.py",
        ]
    )
    lines.extend(
        [
            "",
            "## Deterministic local audit command",
            "",
            f"`{command}`",
            "",
            "Focused smoke example:",
            "",
            (
                "`PYTHONPATH=src uv run --no-sync python "
                "results/7180984/scripts/materialize_direct_epsilon_soft_lambda_redo.py "
                "--run-ids open_loop_small --pgd-steps 2 "
                "--output-json results/7180984/smoke/direct_epsilon_soft_lambda_redo.json "
                "--output-csv results/7180984/smoke/direct_epsilon_soft_lambda_redo.csv "
                "--output-md results/7180984/smoke/direct_epsilon_soft_lambda_redo.md`"
            ),
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
