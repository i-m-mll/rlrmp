"""Frozen direct-epsilon adaptive-lambda replay for issue 08483d5.

It loads existing local artifacts, freezes controller weights, reruns the
existing direct-epsilon inner maximizer, and writes outputs to a caller-provided
directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.runtime.batch import BatchInfo

from rlrmp.analysis.frozen_policy_gate import validate_direct_hvp_lambda_source
from rlrmp.paths import portable_repo_path
from rlrmp.train.cs_nominal_gru import (
    _args_values_from_run_spec,
    _is_replicate_axis_array,
    _with_single_replicate_state_initializers,
    build_hps,
    build_parser,
)
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
    BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    _ensure_broad_epsilon_input,
    run_broad_epsilon_pgd_inner_maximizer,
)
from rlrmp.train.task_model import setup_task_model_pair


D_REF = 6131.6906765
ROOT_KEY_SEED = 84_483_500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-issue", default="ae9f30f")
    parser.add_argument("--source-run", default="direct_epsilon_b1p05")
    parser.add_argument("--checkpoint-batches", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--adaptive-iterations", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--max-log-step", type=float, default=0.75)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_spec_path = repo_root / "results" / args.source_issue / "runs" / f"{args.source_run}.json"
    artifact_dir = repo_root / "_artifacts" / args.source_issue / "runs" / args.source_run
    checkpoint_dir = artifact_dir / "checkpoints" / f"checkpoint_{args.checkpoint_batches:07d}"
    metadata_path = checkpoint_dir / "metadata.json"
    model_path = checkpoint_dir / "model.eqx"
    lambda_source_path = repo_root / "results" / "06a4dc8" / "canonical_soft_lambda_hvp.json"

    exact_baseline_candidates = [
        {
            "path": "results/33b0dcb/runs/h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64.json",
            "status": "materialized_but_8d_48d_not_6d_no_integrator",
        },
        {
            "path": "results/ffff699/runs/delayed_no_integrator_no_pgd_lr3e-3_clip5_b64_seed42/run.json",
            "status": "6d_no_integrator_but_artifact_directory_empty_and_not_const_band16",
        },
    ]

    run_spec = _read_json(run_spec_path)
    if isinstance(run_spec, list):
        if len(run_spec) != 1:
            raise ValueError(f"unexpected run spec list length in {run_spec_path}")
        run_spec = run_spec[0]
    metadata = _read_json(metadata_path)
    lambda_source = _read_json(lambda_source_path)
    direct_lambda_source = validate_direct_hvp_lambda_source(lambda_source, beta=1.05)
    lambda0 = float(direct_lambda_source["candidate_lambda"])
    lambda_curv = float(lambda_source["pooled_summary"]["lambda_star_p90"])

    base_args = build_parser().parse_args([])
    for key, value in _args_values_from_run_spec(run_spec).items():
        setattr(base_args, key, value)
    base_args.batch_size = int(args.batch_size)
    base_args.broad_epsilon_pgd_training = True
    base_args.broad_epsilon_pgd_mechanism = BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
    base_args.broad_epsilon_pgd_objective = BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
    base_args.broad_epsilon_pgd_inner_optimizer_method = BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT
    base_args.broad_epsilon_pgd_steps = int(_pgd_value(run_spec, "inner_maximizer", "n_steps", 10))
    base_args.broad_epsilon_pgd_step_size_fraction = float(
        _pgd_value(run_spec, "inner_maximizer", "step_size_fraction_of_l2_radius", 0.25)
    )
    base_args.broad_epsilon_pgd_safety_cap_15cm = _pgd_value(
        run_spec, "safety_cap", "l2_radius_15cm", None
    )
    base_args.broad_epsilon_pgd_safety_cap_source = "replay_source_run_safety_cap"
    base_args.broad_epsilon_pgd_energy_lambda = float(lambda0)

    hps0 = build_hps(base_args)
    pair = setup_task_model_pair(hps0, key=jr.PRNGKey(int(run_spec.get("seed", 42))))
    model = eqx.tree_deserialise_leaves(model_path, pair.model)
    model = _replicate_model(model, hps0, int(args.replicate_index))

    root_key = jr.fold_in(jnp.asarray(metadata["next_prng_key"], dtype=jnp.uint32), ROOT_KEY_SEED)
    root_key = jr.fold_in(root_key, int(metadata["completed_batches"]))
    key_trials, key_model = jr.split(root_key, 2)
    keys_trials = jr.split(key_trials, int(args.batch_size))
    keys_model = jr.split(key_model, int(args.batch_size))
    batch_index = max(0, int(metadata["completed_batches"]) - 1)
    batch_info = BatchInfo(
        size=int(args.batch_size),
        start=jnp.asarray(0),
        current=jnp.asarray(batch_index),
        total=jnp.asarray(hps0.n_batches_condition),
    )
    trial_specs = eqx.filter_vmap(
        partial(pair.task.get_train_trial_with_intervenor_params, batch_info=batch_info)
    )(keys_trials)
    specs = _ensure_broad_epsilon_input(
        trial_specs,
        epsilon_dim=int(hps0.broad_epsilon_pgd_training.epsilon_dim),
    )

    def evaluate(lambda_value: float, label: str) -> dict[str, Any]:
        eval_args = copy.copy(base_args)
        eval_args.broad_epsilon_pgd_energy_lambda = float(lambda_value)
        hps = build_hps(eval_args)
        updated, diagnostics = run_broad_epsilon_pgd_inner_maximizer(
            pair.task,
            model,
            specs,
            pair.task.loss_func,
            keys_model,
            hps.broad_epsilon_pgd_training,
            return_diagnostics=True,
        )
        base_epsilon = jnp.asarray(specs.inputs["epsilon"])
        selected_epsilon = jnp.asarray(updated.inputs["epsilon"]) - base_epsilon
        row = _diagnostics_to_row(diagnostics)
        row.update(
            {
                "label": label,
                "lambda": float(lambda_value),
                "paired_damage": row["raw_task_loss_selected"] - row["raw_task_loss_zero"],
                "selected_epsilon_l2": float(np.linalg.norm(np.asarray(selected_epsilon))),
                "selected_epsilon_max_abs": float(np.max(np.abs(np.asarray(selected_epsilon)))),
            }
        )
        row["damage_ratio_to_ref"] = row["paired_damage"] / D_REF
        row["nonzero_selected"] = bool(
            row["epsilon_energy_mean"] > 1e-20 and row["selected_epsilon_l2"] > 1e-12
        )
        row["finite"] = not bool(row["inner_objective_nonfinite_seen"])
        row["guard_bound"] = bool(row["cap_boundary_fraction"] >= 0.05)
        return row

    bracket_multipliers = [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01]
    bracket_rows = [
        evaluate(lambda0 * multiplier, f"bracket_x{multiplier:g}")
        for multiplier in bracket_multipliers
    ]
    lambda_zero_bracket = _estimate_lambda_zero_bracket(bracket_rows)

    adaptive_rows: list[dict[str, Any]] = []
    lambda_value = float(lambda0)
    for iteration in range(int(args.adaptive_iterations)):
        row = evaluate(lambda_value, f"adaptive_{iteration:02d}")
        row["iteration"] = iteration
        damage_for_update = max(float(row["paired_damage"]), D_REF * 1e-12)
        unclipped_log_step = float(args.eta) * math.log(damage_for_update / D_REF)
        log_step = float(np.clip(unclipped_log_step, -float(args.max_log_step), float(args.max_log_step)))
        row["update_rule"] = {
            "formula": "log(lambda_next)=log(lambda)+clip(eta*log(max(damage,eps)/D_ref), +/-max_log_step)",
            "eta": float(args.eta),
            "max_log_step": float(args.max_log_step),
            "unclipped_log_step": unclipped_log_step,
            "applied_log_step": log_step,
        }
        lambda_next = float(lambda_value * math.exp(log_step))
        row["lambda_next"] = lambda_next
        adaptive_rows.append(row)
        lambda_value = lambda_next

    payload = {
        "schema_version": "rlrmp.08483d5.local_adaptive_lambda_replay.v1",
        "target": {
            "paired_nominal_noise_damage_ref": D_REF,
            "source_issue": "08483d5",
            "source_note": "results/08483d5/notes/output_feedback_damage_estimate.json",
        },
        "baseline_resolution": {
            "exact_requested": "6D no-PGD H0 const_band16 baseline/model",
            "exact_status": "not_materialized_as_a single local artifact",
            "candidates_checked": exact_baseline_candidates,
            "empirical_replay_source": {
                "status": "compatibility_replay_on_materialized_6d_direct_epsilon_checkpoint",
                "source_issue": args.source_issue,
                "source_run": args.source_run,
                "run_spec_path": portable_repo_path(run_spec_path, repo_root=repo_root),
                "checkpoint_path": portable_repo_path(checkpoint_dir, repo_root=repo_root),
                "checkpoint_batches": int(args.checkpoint_batches),
                "model_path": portable_repo_path(model_path, repo_root=repo_root),
                "replicate_index": int(args.replicate_index),
                "batch_size": int(args.batch_size),
                "batch_index": int(batch_index),
                "model_contract": {
                    "no_integrator_state": bool(hps0.model.no_integrator_state),
                    "physical_state_dim": int(hps0.model.physical_state_dim),
                    "state_dim": int(hps0.model.state_dim),
                    "epsilon_dim": int(hps0.broad_epsilon_pgd_training.epsilon_dim),
                    "initial_hidden_encoder": bool(hps0.model.initial_hidden_encoder),
                },
            },
        },
        "lambda_source": {
            "path": portable_repo_path(lambda_source_path, repo_root=repo_root),
            "lambda_curv_p90": lambda_curv,
            "lambda0_beta_1p05": lambda0,
            "validated_source": direct_lambda_source,
            "cap_or_interiority_used_as_criterion": False,
        },
        "optimizer_replay": {
            "mechanism": "direct_epsilon",
            "inner_optimizer_method": BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
            "n_steps": int(base_args.broad_epsilon_pgd_steps),
            "step_size_fraction": float(base_args.broad_epsilon_pgd_step_size_fraction),
            "safety_cap_l2_radius_15cm": base_args.broad_epsilon_pgd_safety_cap_15cm,
            "controller_updated": False,
            "approximation_note": (
                "Uses existing direct-epsilon inner maximizer diagnostics on a frozen "
                "materialized 6D direct-epsilon checkpoint because the exact no-PGD "
                "6D const_band16 baseline checkpoint was not found locally."
            ),
        },
        "lambda_zero_bracket": lambda_zero_bracket,
        "bracket_rows": bracket_rows,
        "adaptive_rows": adaptive_rows,
        "assessment": _assessment(adaptive_rows, lambda_zero_bracket),
    }

    json_path = output_dir / "adaptive_lambda_replay.json"
    md_path = output_dir / "adaptive_lambda_replay.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


def _diagnostics_to_row(diagnostics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in diagnostics.items():
        array = np.asarray(jax.device_get(value))
        if array.shape == ():
            if array.dtype == np.bool_:
                out[key] = bool(array)
            else:
                out[key] = float(array)
        else:
            out[key] = {
                "mean": float(np.mean(array.astype(np.float64))),
                "min": float(np.min(array.astype(np.float64))),
                "max": float(np.max(array.astype(np.float64))),
            }
    return out


def _estimate_lambda_zero_bracket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: float(row["lambda"]))
    active = [row for row in sorted_rows if row["nonzero_selected"]]
    inactive = [row for row in sorted_rows if not row["nonzero_selected"]]
    lower_active = max((float(row["lambda"]) for row in active), default=None)
    upper_inactive_candidates = [
        float(row["lambda"])
        for row in inactive
        if lower_active is None or float(row["lambda"]) > lower_active
    ]
    upper_inactive = min(upper_inactive_candidates, default=None)
    return {
        "method": "coarse frozen-grid selected-energy threshold",
        "active_threshold": "epsilon_energy_mean > 1e-20 and selected_l2 > 1e-12",
        "lower_active_lambda": lower_active,
        "upper_inactive_lambda": upper_inactive,
        "bracketed": lower_active is not None and upper_inactive is not None,
    }


def _assessment(rows: list[dict[str, Any]], bracket: dict[str, Any]) -> dict[str, Any]:
    finite_all = all(row["finite"] for row in rows)
    nonzero_all = all(row["nonzero_selected"] for row in rows)
    guard_any = any(row["guard_bound"] for row in rows)
    first = rows[0]["paired_damage"]
    last = rows[-1]["paired_damage"]
    moved_toward = abs(last - D_REF) < abs(first - D_REF)
    passed = finite_all and nonzero_all and not guard_any and moved_toward
    blockers: list[str] = []
    if not finite_all:
        blockers.append("nonfinite objective seen")
    if not nonzero_all:
        blockers.append("zero selected adversary occurred")
    if guard_any:
        blockers.append("selected adversary was guard/cap-bound in at least one adaptive row")
    if not moved_toward:
        blockers.append("adaptive damage did not move closer to D_ref on this frozen replay")
    if not bracket.get("bracketed"):
        blockers.append("lambda_zero was not bracketed by the coarse grid")
    return {
        "pass": bool(passed),
        "headline": (
            "fail: finite/nonzero may hold, but damage tracking or guard independence failed"
            if not passed
            else "pass: frozen replay stayed finite/nonzero, off guard, and moved toward target"
        ),
        "finite_all": finite_all,
        "nonzero_all": nonzero_all,
        "guard_bound_any": guard_any,
        "first_damage": first,
        "last_damage": last,
        "moved_toward_target": moved_toward,
        "blockers": blockers,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    rows = payload["adaptive_rows"]
    bracket_rows = payload["bracket_rows"]
    lines = [
        "# Adaptive-lambda frozen replay for 08483d5",
        "",
        "## Headline",
        "",
        f"{payload['assessment']['headline']}.",
        "",
        "This run did not update controller weights and wrote no repo files or ledger state.",
        "",
        "## Resolved model/baseline identity",
        "",
        f"- Requested exact baseline: `{payload['baseline_resolution']['exact_requested']}`.",
        f"- Exact local status: `{payload['baseline_resolution']['exact_status']}`.",
        "- Checked candidates:",
    ]
    for candidate in payload["baseline_resolution"]["candidates_checked"]:
        lines.append(f"  - `{candidate['path']}`: {candidate['status']}.")
    source = payload["baseline_resolution"]["empirical_replay_source"]
    contract = source["model_contract"]
    lines.extend(
        [
            "- Empirical replay source used:",
            f"  - `{source['source_issue']}/{source['source_run']}` at checkpoint {source['checkpoint_batches']}.",
            f"  - Run spec: `{source['run_spec_path']}`.",
            f"  - Model: `{source['model_path']}`.",
            f"  - Contract: no_integrator={contract['no_integrator_state']}, physical_state_dim={contract['physical_state_dim']}, state_dim={contract['state_dim']}, epsilon_dim={contract['epsilon_dim']}, initial_hidden_encoder={contract['initial_hidden_encoder']}.",
            "",
            "## Lambda source",
            "",
            f"- Source: `{payload['lambda_source']['path']}`.",
            f"- `lambda_curv_p90`: `{payload['lambda_source']['lambda_curv_p90']:.9g}`.",
            f"- Initial `lambda0 = beta^2 * lambda_curv_p90` for beta=1.05: `{payload['lambda_source']['lambda0_beta_1p05']:.9g}`.",
            "- The source validator accepted this as cap-independent; cap/trust radius was not used as the lambda criterion.",
            "",
            "## Lambda-zero bracket",
            "",
        ]
    )
    bracket = payload["lambda_zero_bracket"]
    if bracket["bracketed"]:
        lines.append(
            f"Coarse bracket: active at `{bracket['lower_active_lambda']:.9g}`, inactive at `{bracket['upper_inactive_lambda']:.9g}`."
        )
    else:
        lines.append(
            f"Not bracketed. Highest active lambda: `{bracket['lower_active_lambda']}`; lowest inactive above it: `{bracket['upper_inactive_lambda']}`."
        )
    lines.extend(
        [
            "",
            "| lambda | damage | energy_mean | objective_gain | nonzero | cap_boundary | finite |",
            "|---:|---:|---:|---:|:---:|---:|:---:|",
        ]
    )
    for row in bracket_rows:
        lines.append(
            f"| {row['lambda']:.6g} | {row['paired_damage']:.6g} | {row['epsilon_energy_mean']:.6g} | {row['selected_objective_gain_over_zero']:.6g} | {row['nonzero_selected']} | {row['cap_boundary_fraction']:.3g} | {row['finite']} |"
        )
    lines.extend(
        [
            "",
            "## Update rule",
            "",
            "For iteration `k`, rerun the frozen inner maximizer from zero epsilon at the current lambda, compute paired damage `D_k = J_selected - J_zero`, then update:",
            "",
            "`log(lambda_{k+1}) = log(lambda_k) + clip(eta * log(max(D_k, eps) / D_ref), +/- max_log_step)`",
            "",
            f"Here `D_ref = {payload['target']['paired_nominal_noise_damage_ref']:.10g}`, `eta = {rows[0]['update_rule']['eta']}`, `max_log_step = {rows[0]['update_rule']['max_log_step']}`. If damage is below target, lambda decreases; if damage is above target, lambda increases.",
            "",
            "## Adaptive replay diagnostics",
            "",
            "| iter | lambda | damage | damage/ref | energy_mean | objective | objective_gain | nonzero | cap_boundary | finite | next_lambda |",
            "|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['iteration']} | {row['lambda']:.6g} | {row['paired_damage']:.6g} | {row['damage_ratio_to_ref']:.4g} | {row['epsilon_energy_mean']:.6g} | {row['penalized_objective_selected']:.6g} | {row['selected_objective_gain_over_zero']:.6g} | {row['nonzero_selected']} | {row['cap_boundary_fraction']:.3g} | {row['finite']} | {row['lambda_next']:.6g} |"
        )
    assessment = payload["assessment"]
    lines.extend(
        [
            "",
            "## Pass/fail assessment",
            "",
            f"- Pass: `{assessment['pass']}`.",
            f"- Finite all iterations: `{assessment['finite_all']}`.",
            f"- Nonzero all iterations: `{assessment['nonzero_all']}`.",
            f"- Any guard/cap binding: `{assessment['guard_bound_any']}`.",
            f"- First damage: `{assessment['first_damage']:.6g}`; last damage: `{assessment['last_damage']:.6g}`; target: `{D_REF:.6g}`.",
            f"- Moved toward target: `{assessment['moved_toward_target']}`.",
            "",
            "Blockers and uncertainties:",
        ]
    )
    for blocker in assessment["blockers"]:
        lines.append(f"- {blocker}.")
    lines.extend(
        [
            "- Exact requested 6D no-PGD H0 const_band16 baseline checkpoint was not found as a runnable local artifact; the replay source is a materialized 6D direct-epsilon checkpoint.",
            "- The replay uses the existing projected direct-epsilon inner optimizer with its stabilization cap, so any cap-bound row fails the pure-soft interpretation.",
            "- This is a single small frozen batch and should be treated as a mechanism diagnostic, not a launch result.",
            "",
        ]
    )
    return "\n".join(lines)


def _pgd_value(run_spec: dict[str, Any], section: str, key: str, default: Any) -> Any:
    pgd = run_spec.get("hps", {}).get("broad_epsilon_pgd_training", {})
    value = pgd.get(section, {})
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def _replicate_model(model: Any, hps: Any, replicate_index: int) -> Any:
    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
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


if __name__ == "__main__":
    main()
