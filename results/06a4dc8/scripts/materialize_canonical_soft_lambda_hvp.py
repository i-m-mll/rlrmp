"""Materialize canonical HVP/Lanczos soft-lambda estimates for 06a4dc8."""

from __future__ import annotations
from rlrmp.analysis.soft_lambda import base_parser
from rlrmp.analysis.soft_lambda import load_frozen_batch as _load_frozen_batch
from rlrmp.analysis.soft_lambda import soft_pgd_config as soft_pgd_config

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
    _set_input,
    config_from_broad_epsilon_pgd_hps,
)
from rlrmp.train.task_model import setup_task_model_pair


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_IDS = ("open_loop_small", "open_loop_moderate", "open_loop_stress")
BETA_VALUES = (0.95, 1.05, 1.2, 1.4, 1.8)
FINITE_DIFFERENCE_STEPS = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5)
PRIMARY_PERCENTILE = 90.0
CAP_RADIUS_15CM = 0.004545500088363065
CAP_SOURCE = "ofb_6d_no_integrator_gamma_1p4_rollout_radius"
ANALYTIC_GAMMA_STAR = 9166.831285473823


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
class LanczosEstimate:
    eigmax: float
    top_vector: jnp.ndarray
    alphas: list[float]
    betas: list[float]
    ritz_values: list[float]
    residual_norm_estimate: float
    hvp_evaluations: int


def parse_args() -> argparse.Namespace:
    parser = base_parser(description='Estimate corrected per-trial soft lambda_star for c92 no-PGD frozen substrates using Hessian-vector products and Lanczos.', experiment='c92ebd8', issue='06a4dc8', batch_size=8, replicate_index=0)
    parser.add_argument('--run-ids', nargs='+', default=list(RUN_IDS))
    parser.add_argument('--max-trials-per-run', type=int, default=None)
    parser.add_argument('--lanczos-steps', type=int, default=12)
    parser.add_argument('--lanczos-seed', type=int, default=60931)
    parser.add_argument('--fd-trial-limit', type=int, default=1)
    parser.add_argument('--fd-steps', type=float, nargs='+', default=list(FINITE_DIFFERENCE_STEPS))
    parser.add_argument('--betas', type=float, nargs='+', default=list(BETA_VALUES))
    parser.add_argument('--analytic-gamma-star', type=float, default=ANALYTIC_GAMMA_STAR)
    parser.add_argument('--output-json', default='results/06a4dc8/canonical_soft_lambda_hvp.json')
    parser.add_argument('--output-csv', default='results/06a4dc8/canonical_soft_lambda_hvp_trials.csv')
    parser.add_argument('--output-md', default='results/06a4dc8/notes/canonical_soft_lambda_hvp.md')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = materialize(args)
    output_json = REPO_ROOT / args.output_json
    output_csv = REPO_ROOT / args.output_csv
    output_md = REPO_ROOT / args.output_md
    write_compact_json(output_json, payload)
    write_trial_csv(output_csv, payload)
    update_marked_section(output_md, "canonical_soft_lambda_hvp", render_markdown(payload))
    print(
        json.dumps(
            {
                "json": str(output_json),
                "csv": str(output_csv),
                "markdown": str(output_md),
                "primary_p90_lambda_star": payload["pooled_summary"]["lambda_star_p90"],
            },
            indent=2,
        )
    )
    return 0


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    all_trials: list[dict[str, Any]] = []
    for run_id in args.run_ids:
        frozen = load_frozen_batch(args, run_id)
        n_available = int(frozen.radius.shape[0])
        n_trials = n_available
        if args.max_trials_per_run is not None:
            n_trials = min(n_trials, int(args.max_trials_per_run))
        trial_rows = []
        for trial_index in range(n_trials):
            trial_row = estimate_trial(
                slice_frozen_batch(frozen, trial_index),
                run_id=run_id,
                trial_index=trial_index,
                args=args,
                validate_fd=trial_index < int(args.fd_trial_limit),
            )
            trial_rows.append(trial_row)
            all_trials.append(trial_row)
        summary = summarize_trials(trial_rows)
        rows.append(
            {
                "run_id": run_id,
                "run_spec_path": f"results/{args.experiment}/runs/{run_id}.json",
                "artifact_dir": f"_artifacts/{args.experiment}/runs/{run_id}",
                "checkpoint": f"_artifacts/{args.experiment}/runs/{run_id}/trained_model.eqx",
                "batch_size_requested": int(args.batch_size),
                "n_trials_estimated": n_trials,
                "n_trials_available": n_available,
                "replicate_index": int(args.replicate_index),
                "summary": summary,
                "beta_mapping": beta_mapping(summary["lambda_star_p90"], args.betas),
                "trials": trial_rows,
            }
        )
    pooled_summary = summarize_trials(all_trials)
    return {
        "schema_version": "rlrmp.canonical_soft_lambda_hvp.v1",
        "issue": str(args.issue),
        "parent_umbrella": "54389a4",
        "coordinator_issue": "31e67ff",
        "source_experiment": str(args.experiment),
        "command": {"argv": list(sys.argv), "cwd": "."},
        "estimator": {
            "method": "per_trial_hvp_lanczos_largest_algebraic",
            "ordinary_hessian_convention": "J ~= J0 + grad.T eps + 0.5 eps.T M eps",
            "lambda_star_convention": "lambda_star_i = 0.5 * eigmax_i",
            "eigenvalue_selection": "largest_algebraic_ritz_value",
            "full_hessian_materialized": False,
            "lanczos_steps": int(args.lanczos_steps),
            "lanczos_seed": int(args.lanczos_seed),
            "finite_difference_steps": [float(value) for value in args.fd_steps],
            "primary_continuity_summary": "p90(lambda_star_i)",
            "cap_or_interiority_used_as_criterion": False,
        },
        "objective_contract": {
            "soft_objective": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            "estimated_objective": "per-trial J_i(delta_i), not batch mean",
            "soft_training_reduction": "mean_i[J_i(delta_i) - lambda * E_i(delta_i)]",
            "energy": "E_i = sum_t,d delta_epsilon_i,t,d^2 over active mask",
            "epsilon_coordinates": "6D no-integrator process epsilon",
            "epsilon_shape": "trial x time x 6",
            "time_mask": "full-trial epsilon mask from broad-epsilon PGD config",
            "dt_factor": "none",
            "fixed_target_support": "15 cm const-band support inherited from c92 run specs",
            "safety_cap_radius_15cm": CAP_RADIUS_15CM,
            "safety_cap_source": CAP_SOURCE,
            "safety_cap_role": "provenance only; not used as lambda criterion",
        },
        "beta_policy": {
            "formula": "lambda(beta) = beta^2 * p90(lambda_star_i)",
            "primary_beta_values": [float(beta) for beta in args.betas if float(beta) >= 1.0],
            "diagnostic_only_beta_values": [
                float(beta) for beta in args.betas if float(beta) < 1.0
            ],
        },
        "analytic_gamma_comparison": analytic_gamma_comparison(
            pooled_summary["lambda_star_p90"],
            float(args.analytic_gamma_star),
        ),
        "pooled_summary": pooled_summary,
        "pooled_beta_mapping": beta_mapping(pooled_summary["lambda_star_p90"], args.betas),
        "rows": rows,
    }


def load_frozen_batch(args: argparse.Namespace, run_id: str) -> FrozenBatch:
    return _load_frozen_batch(args, run_id, repo_root=REPO_ROOT)




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


def slice_frozen_batch(frozen: FrozenBatch, trial_index: int) -> FrozenBatch:
    def take_trial(leaf: Any) -> Any:
        if (
            eqx.is_array(leaf)
            and leaf.ndim > 0
            and int(leaf.shape[0]) == int(frozen.radius.shape[0])
        ):
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


def trial_objective(frozen: FrozenBatch, delta: jnp.ndarray) -> jnp.ndarray:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    masked_delta = delta * frozen.time_mask
    candidate = _set_input(frozen.trial_specs, "epsilon", epsilon + masked_delta)
    states = frozen.task.eval_trials(frozen.model, candidate, frozen.keys_model)
    return jnp.asarray(frozen.task.loss_func(states, candidate, frozen.model).total)


def estimate_trial(
    frozen: FrozenBatch,
    *,
    run_id: str,
    trial_index: int,
    args: argparse.Namespace,
    validate_fd: bool,
) -> dict[str, Any]:
    epsilon = jnp.asarray(frozen.trial_specs.inputs["epsilon"])
    zero = jnp.zeros_like(epsilon)
    zero_loss, grad = jax.value_and_grad(lambda delta: trial_objective(frozen, delta))(zero)
    grad = grad * frozen.time_mask

    def hvp(vector: jnp.ndarray) -> jnp.ndarray:
        return (
            jax.jvp(
                jax.grad(lambda delta: trial_objective(frozen, delta)),
                (zero,),
                (vector * frozen.time_mask,),
            )[1]
            * frozen.time_mask
        )

    key = jr.fold_in(jr.PRNGKey(int(args.lanczos_seed)), stable_run_fold(run_id) + trial_index)
    lanczos = lanczos_largest_algebraic(
        hvp=hvp,
        shape=zero.shape,
        mask=frozen.time_mask,
        key=key,
        steps=int(args.lanczos_steps),
    )
    lambda_star = 0.5 * lanczos.eigmax
    fd_validation = []
    if validate_fd:
        fd_validation = finite_difference_validation(
            frozen,
            zero_loss=zero_loss,
            direction=lanczos.top_vector,
            eigmax=lanczos.eigmax,
            steps=args.fd_steps,
        )
    radius = jnp.asarray(frozen.radius)[0]
    return {
        "run_id": run_id,
        "trial_index": int(trial_index),
        "zero_loss": float(zero_loss),
        "gradient_norm": float(_flattened_per_trial_norm(grad)[0]),
        "radius": float(radius),
        "active_dimension": int(np.count_nonzero(np.asarray(jax.device_get(frozen.time_mask)))),
        "eigmax_largest_algebraic": float(lanczos.eigmax),
        "lambda_star": float(lambda_star),
        "lanczos": {
            "hvp_evaluations": lanczos.hvp_evaluations,
            "alphas": lanczos.alphas,
            "betas": lanczos.betas,
            "ritz_values": lanczos.ritz_values,
            "residual_norm_estimate": lanczos.residual_norm_estimate,
        },
        "finite_difference_validation": fd_validation,
    }


def lanczos_largest_algebraic(
    *,
    hvp: Any,
    shape: tuple[int, ...],
    mask: jnp.ndarray,
    key: jax.Array,
    steps: int,
) -> LanczosEstimate:
    vectors: list[jnp.ndarray] = []
    alphas: list[float] = []
    betas: list[float] = []
    v = normalize(jr.normal(key, shape, dtype=mask.dtype) * mask)
    beta_prev = jnp.asarray(0.0, dtype=mask.dtype)
    v_prev = jnp.zeros_like(v)
    terminal_beta = 0.0
    for step_index in range(max(int(steps), 1)):
        vectors.append(v)
        w = hvp(v) * mask
        alpha = dot(v, w)
        w = w - alpha * v - beta_prev * v_prev
        for basis in vectors:
            w = w - dot(basis, w) * basis
        beta = norm(w)
        alphas.append(float(alpha))
        terminal_beta = float(beta)
        if step_index == int(steps) - 1 or float(beta) < 1e-10:
            break
        betas.append(float(beta))
        v_prev = v
        v = w / jnp.maximum(beta, jnp.asarray(1e-30, dtype=beta.dtype))
        beta_prev = beta

    tmat = np.diag(np.asarray(alphas, dtype=float))
    if betas:
        offdiag = np.asarray(betas, dtype=float)
        tmat = tmat + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)
    ritz_values, ritz_vectors = np.linalg.eigh(tmat)
    top_index = int(np.argmax(ritz_values))
    coeffs = ritz_vectors[:, top_index]
    top_vector = jnp.zeros_like(vectors[0])
    for coeff, basis in zip(coeffs, vectors):
        top_vector = top_vector + float(coeff) * basis
    top_vector = normalize(top_vector * mask)
    residual = terminal_beta * abs(float(coeffs[-1])) if len(coeffs) else float("nan")
    return LanczosEstimate(
        eigmax=float(ritz_values[top_index]),
        top_vector=top_vector,
        alphas=[float(value) for value in alphas],
        betas=[float(value) for value in betas],
        ritz_values=[float(value) for value in ritz_values],
        residual_norm_estimate=float(residual),
        hvp_evaluations=len(alphas),
    )


def finite_difference_validation(
    frozen: FrozenBatch,
    *,
    zero_loss: jnp.ndarray,
    direction: jnp.ndarray,
    eigmax: float,
    steps: list[float],
) -> list[dict[str, float]]:
    rows = []
    for step in steps:
        step_value = jnp.asarray(float(step), dtype=direction.dtype)
        plus = trial_objective(frozen, step_value * direction)
        minus = trial_objective(frozen, -step_value * direction)
        curvature = (plus + minus - 2.0 * zero_loss) / jnp.maximum(step_value**2, 1e-30)
        curvature_float = float(curvature)
        rows.append(
            {
                "step": float(step),
                "central_curvature": curvature_float,
                "hvp_eigmax": float(eigmax),
                "absolute_error": float(abs(curvature_float - float(eigmax))),
                "relative_error": float(
                    abs(curvature_float - float(eigmax)) / max(abs(float(eigmax)), 1e-30)
                ),
                "loss_plus": float(plus),
                "loss_minus": float(minus),
            }
        )
    return rows


def normalize(value: jnp.ndarray) -> jnp.ndarray:
    return value / jnp.maximum(norm(value), jnp.asarray(1e-30, dtype=value.dtype))


def dot(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return jnp.vdot(left.reshape(-1), right.reshape(-1)).real


def norm(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(dot(value, value), jnp.asarray(0.0, dtype=value.dtype)))


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    lambdas = finite_array([trial["lambda_star"] for trial in trials])
    eigvals = finite_array([trial["eigmax_largest_algebraic"] for trial in trials])
    return {
        "n_trials": len(trials),
        "finite_fraction": float(len(lambdas) / len(trials)) if trials else 0.0,
        "lambda_star_median": percentile_or_nan(lambdas, 50.0),
        "lambda_star_p75": percentile_or_nan(lambdas, 75.0),
        "lambda_star_p90": percentile_or_nan(lambdas, PRIMARY_PERCENTILE),
        "lambda_star_max": max_or_nan(lambdas),
        "eigmax_median": percentile_or_nan(eigvals, 50.0),
        "eigmax_p75": percentile_or_nan(eigvals, 75.0),
        "eigmax_p90": percentile_or_nan(eigvals, PRIMARY_PERCENTILE),
        "eigmax_max": max_or_nan(eigvals),
        "primary_continuity_summary": "lambda_star_p90",
    }


def finite_array(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def max_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def beta_mapping(lambda_star: float, betas: list[float]) -> list[dict[str, Any]]:
    rows = []
    for beta in betas:
        beta_float = float(beta)
        rows.append(
            {
                "beta": beta_float,
                "lambda": float((beta_float**2) * lambda_star),
                "role": "diagnostic_only" if beta_float < 1.0 else "candidate_training_scale",
                "mapping": "lambda = beta^2 * p90(lambda_star_i)",
            }
        )
    return rows


def analytic_gamma_comparison(lambda_star: float, gamma_star: float) -> dict[str, Any]:
    gamma_squared = float(gamma_star) ** 2
    return {
        "analytic_gamma_star": float(gamma_star),
        "previous_lambda_equals_gamma_squared": gamma_squared,
        "lambda_star_over_gamma_star_squared": float(lambda_star / gamma_squared),
        "interpretation": "conversion factor from analytical gamma^2 to GRU-local p90 lambda_star",
    }


def write_trial_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "trial_index",
                "zero_loss",
                "gradient_norm",
                "radius",
                "active_dimension",
                "eigmax_largest_algebraic",
                "lambda_star",
                "hvp_evaluations",
                "lanczos_residual_norm_estimate",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in payload["rows"]:
            for trial in row["trials"]:
                writer.writerow(
                    {
                        "run_id": trial["run_id"],
                        "trial_index": trial["trial_index"],
                        "zero_loss": trial["zero_loss"],
                        "gradient_norm": trial["gradient_norm"],
                        "radius": trial["radius"],
                        "active_dimension": trial["active_dimension"],
                        "eigmax_largest_algebraic": trial["eigmax_largest_algebraic"],
                        "lambda_star": trial["lambda_star"],
                        "hvp_evaluations": trial["lanczos"]["hvp_evaluations"],
                        "lanczos_residual_norm_estimate": trial["lanczos"][
                            "residual_norm_estimate"
                        ],
                    }
                )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Canonical soft-lambda HVP estimate",
        "",
        f"Issue: `{payload['issue']}`. Source experiment: `{payload['source_experiment']}`.",
        "",
        "## Method",
        "",
        (
            "The materializer estimates each trial's largest algebraic Hessian "
            "eigenvalue with HVP-backed Lanczos and reports "
            "`lambda_star_i = 0.5 * eigmax_i` under the ordinary Hessian "
            "convention `J ~= J0 + grad.T eps + 0.5 eps.T M eps`."
        ),
        "",
        (
            "The objective is the corrected per-trial `J_i(delta_i)` for the "
            "6D no-integrator process-epsilon channel. The corresponding soft "
            "training convention is `mean_i[J_i - lambda * E_i]`; cap/interiority "
            "is recorded only as provenance and is not used as a criterion."
        ),
        "",
        "## Run provenance",
        "",
        f"Command: `{' '.join(payload['command']['argv'])}`",
        "",
        (
            f"Lanczos steps: `{payload['estimator']['lanczos_steps']}`. "
            f"Lanczos seed: `{payload['estimator']['lanczos_seed']}`. "
            "The JSON sidecar records per-trial Ritz values and residual estimates."
        ),
        "",
        "## Distribution summary",
        "",
        "| substrate | n | lambda median | lambda p75 | lambda p90 | lambda max | eigmax p90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        summary = row["summary"]
        lines.append(
            f"| `{row['run_id']}` | {summary['n_trials']} | "
            f"{summary['lambda_star_median']:.6g} | {summary['lambda_star_p75']:.6g} | "
            f"**{summary['lambda_star_p90']:.6g}** | {summary['lambda_star_max']:.6g} | "
            f"{summary['eigmax_p90']:.6g} |"
        )
    pooled = payload["pooled_summary"]
    lines.append(
        f"| `pooled` | {pooled['n_trials']} | {pooled['lambda_star_median']:.6g} | "
        f"{pooled['lambda_star_p75']:.6g} | **{pooled['lambda_star_p90']:.6g}** | "
        f"{pooled['lambda_star_max']:.6g} | {pooled['eigmax_p90']:.6g} |"
    )
    lines.extend(
        [
            "",
            "Primary continuity summary: p90 of `lambda_star_i`.",
            "",
            "## Beta mapping",
            "",
            "| source | beta | role | lambda |",
            "|---|---:|---|---:|",
        ]
    )
    for row in payload["rows"]:
        for mapping in row["beta_mapping"]:
            lines.append(
                f"| `{row['run_id']}` | {mapping['beta']:.3g} | "
                f"{mapping['role']} | {mapping['lambda']:.6g} |"
            )
    for mapping in payload["pooled_beta_mapping"]:
        lines.append(
            f"| `pooled` | {mapping['beta']:.3g} | {mapping['role']} | {mapping['lambda']:.6g} |"
        )
    gamma = payload["analytic_gamma_comparison"]
    lines.extend(
        [
            "",
            "Beta `0.95` is diagnostic only; beta values at or above `1.0` are "
            "candidate training scales pending later lanes.",
            "",
            "## Analytical gamma comparison",
            "",
            (
                f"`gamma_star = {gamma['analytic_gamma_star']:.12g}`, so the previous "
                f"`lambda = gamma^2` convention corresponds to "
                f"`{gamma['previous_lambda_equals_gamma_squared']:.6g}`. The pooled "
                f"p90 GRU-local conversion factor is "
                f"`{gamma['lambda_star_over_gamma_star_squared']:.6g}`."
            ),
            "",
            "## Finite-difference validation",
            "",
            "| substrate | trial | step | central curvature | HVP eigmax | rel error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        for trial in row["trials"]:
            for check in trial["finite_difference_validation"]:
                lines.append(
                    f"| `{row['run_id']}` | {trial['trial_index']} | "
                    f"{check['step']:.1e} | {check['central_curvature']:.6g} | "
                    f"{check['hvp_eigmax']:.6g} | {check['relative_error']:.3g} |"
                )
    lines.extend(
        [
            "",
            "## Deterministic smoke command",
            "",
            "```bash",
            "PYTHONPATH=src uv run --no-sync python results/06a4dc8/scripts/materialize_canonical_soft_lambda_hvp.py \\",
            "  --run-ids open_loop_small --batch-size 1 --max-trials-per-run 1 --lanczos-steps 2 \\",
            "  --output-json results/06a4dc8/smoke/canonical_soft_lambda_hvp.json \\",
            "  --output-csv results/06a4dc8/smoke/canonical_soft_lambda_hvp_trials.csv \\",
            "  --output-md results/06a4dc8/smoke/canonical_soft_lambda_hvp.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
