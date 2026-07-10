"""Aggregate minimax evaluation results across multiple replicate seeds.

Loads models from each seed_N/ subdirectory produced by train_minimax.py and
reports mean ± std across seeds for key metrics:
  - Peak velocity at SISU=0, 0.5, 1.0 (pert_scale=0)
  - Endpoint error at SISU=0.5
  - SISU velocity effect Δvel% = (vel(S=1) - vel(S=0)) / vel(S=0) * 100
  - Adversary force norms at SISU=0 and SISU=1
  - Final adversarial loss curve summary

Seeds with a missing adversarial_model.eqx are silently skipped (training
may not yet have completed for those replicates).

Usage:
    uv run python scripts/eval_minimax_aggregate.py \\
        --results-dir /workspace/results/minimax_single
    uv run python scripts/eval_minimax_aggregate.py \\
        --results-dir /workspace/results/minimax_single \\
        --eval-key 99
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax.random as jr
import numpy as np

from rlrmp.eval import eval_at_pert0
from rlrmp.eval.minimax_io import load_config, load_model
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.train.minimax_native import build_hps


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------


def _eval_single_seed(
    seed_dir: Path,
    eval_key_seed: int = 42,
) -> dict[str, Any] | None:
    """Evaluate warmup and adversarial models from one seed directory.

    Args:
        seed_dir: Path to a single seed directory (e.g. results/minimax_single/seed_0).
        eval_key_seed: Integer seed for the JAX evaluation PRNG key.

    Returns:
        A dict with keys ``"seed"``, ``"warmup"``, ``"adversarial"``, and
        ``"adversary_force"``, or None if adversarial_model.eqx is absent.
    """
    adv_model_path = seed_dir / "adversarial_model.eqx"
    if not adv_model_path.exists():
        return None

    print(f"  {seed_dir.name}: loading config ... ", end="", flush=True)
    config = load_config(seed_dir)
    hps = build_hps(
        argparse.Namespace(**{k: v for k, v in config.items() if k not in ("git", "output_dir")})
    )

    key = jr.PRNGKey(eval_key_seed)
    key, setup_key = jr.split(key)
    pair = setup_task_model_pair(hps, key=setup_key)
    task = pair.task
    print("task OK, loading models ... ", end="", flush=True)

    warmup_model = load_model(seed_dir, "warmup_model.eqx", hps, config)
    adv_model = load_model(seed_dir, "adversarial_model.eqx", hps, config)
    print("models OK")

    SISU_VALUES = [0.0, 0.5, 1.0]

    def _eval_model(model) -> dict[str, float] | None:
        if model is None:
            return None
        nonlocal key
        result: dict[str, float] = {}
        eval_key = jr.PRNGKey(eval_key_seed + 10)
        for sisu in SISU_VALUES:
            eval_key, k = jr.split(eval_key)
            km = eval_at_pert0(task, model, sisu=sisu, key=k)
            result[f"vel_{sisu}"] = float(np.mean(km["peak_velocity"]))
            result[f"ep_err_{sisu}"] = float(np.mean(km["endpoint_error"]))
            result[f"lat_dev_{sisu}"] = float(np.mean(km["max_lateral_deviation"]))
        return result

    warmup_row = _eval_model(warmup_model)
    adv_row = _eval_model(adv_model)

    # Adversary force norms from pre-saved profiles.
    force_norms: dict[str, float] = {}
    profile_path = seed_dir / "adversary_force_profiles.npz"
    if profile_path.exists():
        data = np.load(profile_path)
        for sisu in [0.0, 0.5, 1.0]:
            k_str = f"sisu_{sisu:.2f}"
            if k_str in data:
                force_norms[k_str] = float(np.linalg.norm(data[k_str]))

    # Adversarial loss summary.
    loss_summary: dict[str, float] = {}
    loss_path = seed_dir / "adversarial_losses.npz"
    if loss_path.exists():
        data = np.load(loss_path)
        ctrl = data["ctrl_losses"]
        adv = data["adv_losses"]
        loss_summary = {
            "ctrl_loss_initial": float(ctrl[0]),
            "ctrl_loss_final": float(ctrl[-1]),
            "adv_loss_initial": float(adv[0]),
            "adv_loss_final": float(adv[-1]),
            "n_batches": int(len(ctrl)),
        }

    return {
        "seed": seed_dir.name,
        "warmup": warmup_row,
        "adversarial": adv_row,
        "adversary_force": force_norms,
        "loss_summary": loss_summary,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _collect_metric(seed_results: list[dict], model_key: str, metric: str) -> np.ndarray:
    """Extract a scalar metric from each seed's model_key sub-dict.

    Args:
        seed_results: List of per-seed result dicts from _eval_single_seed.
        model_key: ``"warmup"`` or ``"adversarial"``.
        metric: Key in the per-model result dict (e.g. ``"vel_0.0"``).

    Returns:
        1-D numpy array of values across seeds (seeds where the value is absent are skipped).
    """
    vals = []
    for r in seed_results:
        sub = r.get(model_key)
        if sub is not None and metric in sub:
            vals.append(sub[metric])
    return np.array(vals, dtype=float)


def _mean_std(arr: np.ndarray) -> tuple[float, float]:
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=min(1, len(arr) - 1)))


def _fmt(mean: float, std: float, fmt: str = ".4f") -> str:
    return f"{mean:{fmt}} ± {std:{fmt}}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate minimax evaluation results across multiple seeds."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/workspace/_artifacts/minimax/minimax_single",
        help=(
            "Parent directory containing seed_0/, seed_1/, ... sub-directories "
            "(bulk artifact directory). "
            "Use rlrmp.paths.run_artifact_dir(exp, run) to construct this path."
        ),
    )
    parser.add_argument(
        "--eval-key",
        type=int,
        default=42,
        help="Integer seed for the JAX evaluation PRNG key (default: 42).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    # Discover seed directories.
    seed_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not seed_dirs:
        print(f"ERROR: no seed_N/ directories found under {results_dir}")
        sys.exit(1)

    print(f"Results directory : {results_dir}")
    print(f"Seed directories  : {[d.name for d in seed_dirs]}")
    print()

    # -----------------------------------------------------------------------
    # Per-seed evaluation
    # -----------------------------------------------------------------------
    print("Evaluating seeds:")
    seed_results: list[dict[str, Any]] = []
    for sd in seed_dirs:
        result = _eval_single_seed(sd, eval_key_seed=args.eval_key)
        if result is None:
            print(f"  {sd.name}: adversarial_model.eqx not found — skipping")
        else:
            seed_results.append(result)

    n_seeds = len(seed_results)
    if n_seeds == 0:
        print("\nNo completed seeds found — nothing to aggregate.")
        sys.exit(1)

    print(f"\n{n_seeds} seed(s) evaluated.\n")

    # -----------------------------------------------------------------------
    # Build aggregated metric table
    # -----------------------------------------------------------------------
    SISU_VALUES = [0.0, 0.5, 1.0]
    model_keys = ["warmup", "adversarial"]

    # Check which model types are actually available.
    available_models = [
        mk for mk in model_keys if any(r.get(mk) is not None for r in seed_results)
    ]

    agg: dict[str, dict[str, Any]] = {}
    per_seed_delta_pct: dict[str, list[float]] = {}

    for mk in available_models:
        row: dict[str, Any] = {}
        for sisu in SISU_VALUES:
            vel_arr = _collect_metric(seed_results, mk, f"vel_{sisu}")
            ep_err_arr = _collect_metric(seed_results, mk, f"ep_err_{sisu}")
            row[f"vel_{sisu}"] = _mean_std(vel_arr)
            row[f"ep_err_{sisu}"] = _mean_std(ep_err_arr)

        # Δvel%: compute per seed, then aggregate.
        delta_pcts: list[float] = []
        for r in seed_results:
            sub = r.get(mk)
            if sub is not None:
                v0 = sub.get(f"vel_{0.0}")
                v1 = sub.get(f"vel_{1.0}")
                if v0 is not None and v1 is not None and abs(v0) > 1e-8:
                    delta_pcts.append(100.0 * (v1 - v0) / v0)
        per_seed_delta_pct[mk] = delta_pcts
        row["delta_vel_pct"] = _mean_std(np.array(delta_pcts))
        agg[mk] = row

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    col_w = 16
    header_parts = [
        f"{'Model':<14}",
        f"{'vel(S=0)':>{col_w}}",
        f"{'vel(S=0.5)':>{col_w}}",
        f"{'vel(S=1)':>{col_w}}",
        f"{'Δvel%':>{col_w}}",
        f"{'ep_err(S=0.5)':>{col_w}}",
    ]
    header = " | ".join(header_parts)
    separator = "-" * len(header)

    print("=" * len(separator))
    print("AGGREGATE TABLE: SISU velocity effect (pert_scale=0, mean ± std across seeds)")
    print("=" * len(separator))
    print(header)
    print(separator)

    for mk in available_models:
        row = agg[mk]
        v0_str = _fmt(*row["vel_0.0"])
        v05_str = _fmt(*row["vel_0.5"])
        v1_str = _fmt(*row["vel_1.0"])
        dpct_mean, dpct_std = row["delta_vel_pct"]
        dpct_str = f"{dpct_mean:+.2f} ± {dpct_std:.2f}%"
        ep_str = _fmt(*row["ep_err_0.5"])
        print(
            f"  {mk:<12} | {v0_str:>{col_w}} | {v05_str:>{col_w}} | {v1_str:>{col_w}} | "
            f"{dpct_str:>{col_w}} | {ep_str:>{col_w}}"
        )

    print()

    # -----------------------------------------------------------------------
    # Per-seed Δvel%
    # -----------------------------------------------------------------------
    print("=" * len(separator))
    print("PER-SEED Δvel% (vel(S=1) - vel(S=0)) / vel(S=0) * 100")
    print("=" * len(separator))

    for mk in available_models:
        dpcts = per_seed_delta_pct[mk]
        print(f"  {mk}:")
        for r, dpct in zip(seed_results, dpcts):
            print(f"    {r['seed']:>8}: {dpct:+.3f}%")
    print()

    # -----------------------------------------------------------------------
    # Adversary force norms
    # -----------------------------------------------------------------------
    print("=" * len(separator))
    print("ADVERSARY FORCE NORMS (mean ± std across seeds)")
    print("=" * len(separator))
    force_keys = ["sisu_0.00", "sisu_0.50", "sisu_1.00"]
    force_available = any(r.get("adversary_force") for r in seed_results)
    if force_available:
        for fk in force_keys:
            vals = [
                r["adversary_force"][fk]
                for r in seed_results
                if fk in r.get("adversary_force", {})
            ]
            if vals:
                arr = np.array(vals)
                mean, std = _mean_std(arr)
                sisu_label = fk.replace("sisu_", "SISU=")
                print(f"  {sisu_label}: {_fmt(mean, std)}")

        # Ratio SISU=1 / SISU=0
        norm0_vals = np.array(
            [r["adversary_force"]["sisu_0.00"] for r in seed_results if "sisu_0.00" in r.get("adversary_force", {})]
        )
        norm1_vals = np.array(
            [r["adversary_force"]["sisu_1.00"] for r in seed_results if "sisu_1.00" in r.get("adversary_force", {})]
        )
        if len(norm0_vals) > 0 and len(norm1_vals) == len(norm0_vals):
            ratios = norm1_vals / (norm0_vals + 1e-8)
            print(f"\n  Force norm ratio SISU=1/SISU=0: {_fmt(*_mean_std(ratios))}")
    else:
        print("  No adversary force profiles found in any seed.")
    print()

    # -----------------------------------------------------------------------
    # Loss curve summary
    # -----------------------------------------------------------------------
    print("=" * len(separator))
    print("ADVERSARIAL LOSS CURVES (mean ± std across seeds)")
    print("=" * len(separator))
    loss_fields = [
        ("ctrl_loss_initial", "ctrl_loss initial"),
        ("ctrl_loss_final", "ctrl_loss final"),
        ("adv_loss_initial", "adv_loss initial"),
        ("adv_loss_final", "adv_loss final"),
    ]
    for field, label in loss_fields:
        vals = [
            r["loss_summary"][field]
            for r in seed_results
            if field in r.get("loss_summary", {})
        ]
        if vals:
            arr = np.array(vals)
            print(f"  {label:<24}: {_fmt(*_mean_std(arr))}")

    n_batches_vals = [
        r["loss_summary"]["n_batches"]
        for r in seed_results
        if "n_batches" in r.get("loss_summary", {})
    ]
    if n_batches_vals:
        print(f"  {'n_batches':<24}: {n_batches_vals} (per seed)")
    print()

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    output: dict[str, Any] = {
        "n_seeds_evaluated": n_seeds,
        "seeds": [r["seed"] for r in seed_results],
        "aggregate": {
            mk: {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in row.items()
            }
            for mk, row in agg.items()
        },
        "per_seed_delta_vel_pct": {mk: per_seed_delta_pct[mk] for mk in available_models},
        "per_seed_raw": [
            {
                "seed": r["seed"],
                "warmup": r["warmup"],
                "adversarial": r["adversarial"],
                "adversary_force": r["adversary_force"],
                "loss_summary": r["loss_summary"],
            }
            for r in seed_results
        ],
    }

    out_path = results_dir / "aggregate_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Aggregated results saved to: {out_path}")


if __name__ == "__main__":
    main()
